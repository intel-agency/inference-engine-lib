using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using InferenceEngine.Core.Worker;
using Microsoft.Extensions.Logging;

namespace InferenceEngine.Core;

/// <summary>
/// Host-side proxy that offers the same contract as <see cref="BaseInferenceEngine{TInput,TOutput}"/>
/// but runs the ONNX session in a <b>child worker process</b> over a length-prefixed JSON pipe.
/// The host never loads <c>libonnxruntime</c>/comgr; the worker never loads Mesa/GL — avoiding the
/// Mesa <c>libLLVM-19</c> ↔ comgr <c>libLLVM-20</c> collision that aborts an in-process MIGraphX +
/// GL-rendered UI. <see cref="BaseInferenceEngine{TInput,TOutput}"/>.EngineInfo is mirrored from the worker's Load response, so the
/// diagnostics surface (provider, fallback reason, ordered checks) is identical to the in-process engine.
/// </summary>
/// <remarks>
/// This base overrides <see cref="LoadAsync"/> and <see cref="PredictAsync"/> to marshal to the
/// worker; it never creates an in-process <see cref="Microsoft.ML.OnnxRuntime.InferenceSession"/>.
/// The consumer supplies the worker executable path (its own app launched with worker args) and an
/// <see cref="OutOfProcessEngineOptions"/>.
/// </remarks>
public abstract class OutOfProcessBaseInferenceEngine<TInput, TOutput> : BaseInferenceEngine<TInput, TOutput>
{
    private readonly OutOfProcessEngineOptions _oopOptions;
    private readonly IWorkerSerializer<TInput, TOutput> _serializer;

    private Process? _process;
    private Stream? _toWorker;   // host → worker stdin
    private Stream? _fromWorker; // worker stdout → host
    private bool _workerLoaded;
    private bool _disposed;

    protected OutOfProcessBaseInferenceEngine(OutOfProcessEngineOptions options, ILogger? logger = null)
        : base(options.EngineOptions, logger)
    {
        _oopOptions = options ?? throw new ArgumentNullException(nameof(options));
        _serializer = CreateSerializer();
    }

    /// <summary>Serializer used for the IPC payload. Override to customize; defaults to
    /// <see cref="SystemTextJsonWorkerSerializer{TInput,TOutput}"/>.</summary>
    protected virtual IWorkerSerializer<TInput, TOutput> CreateSerializer()
        => new SystemTextJsonWorkerSerializer<TInput, TOutput>();

    // The model lives in the worker; the host proxy owns no model.
    protected override string ModelName => _oopOptions.EngineId;

    /// <summary>
    /// Spawn the worker, send <c>load</c>, and mirror the worker's <see cref="BaseInferenceEngine{TInput,TOutput}"/>.EngineInfo onto this
    /// proxy via <see cref="BaseInferenceEngine{TInput,TOutput}.ApplyLoadedState"/>. Does NOT build an
    /// in-process session. Idempotent.
    /// </summary>
    public override async Task LoadAsync()
    {
        if (_workerLoaded) return;
        ThrowIfDisposed();

        StartWorker();

        var engineOpts = _oopOptions.EngineOptions;
        await WriteAsync(new WorkerEnvelope
        {
            Kind = "load",
            EngineId = _oopOptions.EngineId,
            Options = new WorkerEngineOptions
            {
                ModelPath = engineOpts.ModelPath,
                UseGpuAcceleration = engineOpts.UseGpuAcceleration,
                DeviceId = engineOpts.DeviceId,
                WarmupOnLoad = engineOpts.WarmupOnLoad,
            },
        }).ConfigureAwait(false);

        var loaded = await ReadAsync(_oopOptions.StartupTimeout).ConfigureAwait(false)
            ?? throw new InvalidOperationException("Worker closed the pipe during load.");

        if (loaded.Error is not null)
            throw new InvalidOperationException($"Worker failed to load the engine: {loaded.Error}");
        if (loaded.EngineInfo is null)
            throw new InvalidOperationException("Worker load response had no EngineInfo.");

        // Mirror the worker's loaded diagnostics. Metadata (NodeMetadata) is ORT-specific and not
        // serialized in this revision; it stays null on the proxy (a follow-up can carry name/dims).
        ApplyLoadedState(loaded.EngineInfo, inputMetadata: null, outputMetadata: null, primaryInputName: null);
        _workerLoaded = true;
    }

    /// <summary>Marshal a <c>predict</c> to the worker and return its deserialized result.</summary>
    public override async Task<TOutput> PredictAsync(TInput input)
    {
        if (!_workerLoaded) await LoadAsync().ConfigureAwait(false);
        ThrowIfDisposed();

        await WriteAsync(new WorkerEnvelope { Kind = "predict", Payload = _serializer.SerializeInput(input) })
            .ConfigureAwait(false);

        var result = await ReadAsync(_oopOptions.InferenceTimeout).ConfigureAwait(false)
            ?? throw new InvalidOperationException("Worker closed the pipe during predict.");

        if (result.Error is not null)
            throw new InvalidOperationException($"Worker predict failed: {result.Error}");
        if (result.Payload is null)
            throw new InvalidOperationException("Worker predict returned no payload.");

        return _serializer.DeserializeOutput(result.Payload);
    }

    // ---- plumbing -----------------------------------------------------------

    private void StartWorker()
    {
        // Clean up any previously-started worker (e.g. a Load retry after a failure) BEFORE spawning
        // a new one, to avoid leaking worker processes / GPU memory.
        if (_process is not null)
        {
            try { if (!_process.HasExited) _process.Kill(); }
            catch { /* best-effort */ }
            _toWorker?.Dispose();
            _fromWorker?.Dispose();
            _process.Dispose();
            _process = null;
            _toWorker = null;
            _fromWorker = null;
        }

        var psi = new ProcessStartInfo(_oopOptions.WorkerExecutablePath)
        {
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (var arg in _oopOptions.WorkerArguments)
            psi.ArgumentList.Add(arg);
        foreach (var kv in _oopOptions.WorkerEnvironment)
            psi.Environment[kv.Key] = kv.Value;

        _process = new Process { StartInfo = psi };
        // Worker logs pass through to the host logger (stderr only; stdout is protocol-only).
        _process.ErrorDataReceived += (_, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
                _logger?.LogWarning("worker stderr: {Line}", e.Data);
        };

        if (!_process.Start())
            throw new InvalidOperationException($"Failed to start worker process: {_oopOptions.WorkerExecutablePath}");

        _process.BeginErrorReadLine();
        _toWorker = _process.StandardInput.BaseStream;
        _fromWorker = _process.StandardOutput.BaseStream;
    }

    private async Task WriteAsync(WorkerEnvelope envelope)
    {
        ThrowIfDisposed();
        await WorkerFrame.WriteAsync(_toWorker!, envelope, CancellationToken.None).ConfigureAwait(false);
    }

    private async Task<WorkerEnvelope?> ReadAsync(TimeSpan timeout)
    {
        ThrowIfDisposed();
        using var cts = new CancellationTokenSource(timeout);
        try
        {
            return await WorkerFrame.ReadAsync(_fromWorker!, cts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw new TimeoutException($"Worker did not respond within {timeout}.");
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OutOfProcessBaseInferenceEngine<TInput, TOutput>));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing && !_disposed)
        {
            _disposed = true;

            // Best-effort graceful shutdown — only when we actually have a live pipe + process.
            try
            {
                if (_toWorker is not null && _process is not null && !_process.HasExited)
                {
                    try { WorkerFrame.Write(_toWorker, new WorkerEnvelope { Kind = "shutdown" }); }
                    catch { /* best-effort */ }
                }
            }
            catch { /* swallow teardown errors */ }

            // Dispose the streams FIRST: closing stdin signals EOF to the worker (unblocking any
            // pending stdout/stderr write) so WaitForExit cannot deadlock; closing stdout ends our read side.
            _toWorker?.Dispose();
            _fromWorker?.Dispose();
            _toWorker = null;
            _fromWorker = null;

            try
            {
                if (_process is not null && !_process.HasExited)
                {
                    if (!_process.WaitForExit(2000))
                        try { _process.Kill(); } catch { /* best-effort */ }
                }
            }
            catch { /* swallow teardown errors */ }

            _process?.Dispose();
            _process = null;
        }
        base.Dispose(disposing);
    }

    // The one-shot pre/post hooks are unused: this engine marshals to the worker.
    protected sealed override (List<Microsoft.ML.OnnxRuntime.NamedOnnxValue> InputTensors, object? Context)
        PreProcessWithContext(TInput input) =>
        throw new NotSupportedException(
            "Out-of-process engines marshal Load/Predict to a worker; PreProcessWithContext is not used.");
    protected sealed override TOutput PostProcessWithContext(
        Microsoft.ML.OnnxRuntime.IDisposableReadOnlyCollection<Microsoft.ML.OnnxRuntime.DisposableNamedOnnxValue> output,
        object? context) =>
        throw new NotSupportedException(
            "Out-of-process engines marshal Load/Predict to a worker; PostProcessWithContext is not used.");
}
