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
/// <para>
/// This base overrides <see cref="LoadAsync"/> and <see cref="PredictAsync"/> to marshal to the
/// worker; it never creates an in-process <see cref="Microsoft.ML.OnnxRuntime.InferenceSession"/>.
/// The consumer supplies the worker executable path (its own app launched with worker args) and an
/// <see cref="OutOfProcessEngineOptions"/>.
/// </para>
/// <para>
/// <b>Congcurrency (H1):</b> all load/predict IPC is serialized by <see cref="_inFlight"/> so that
/// concurrent calls never interleave write/read on the shared pipe (which would silently pair the
/// wrong request with the wrong response). <see cref="Dispose"/> (M6) also takes that lock so it
/// cannot tear down the pipe while a call is mid-flight.
/// </para>
/// </remarks>
public abstract class OutOfProcessBaseInferenceEngine<TInput, TOutput> : BaseInferenceEngine<TInput, TOutput>, IAsyncDisposable
{
    private readonly OutOfProcessEngineOptions _oopOptions;
    private readonly IWorkerSerializer<TInput, TOutput> _serializer;

    // Serializes all host↔worker IPC so concurrent PredictAsync calls cannot interleave their
    // write/read pairs on the shared stdin/stdout pipe (H1). Dispose acquires it too (M6).
    private readonly SemaphoreSlim _inFlight = new(1, 1);

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
    /// in-process session. Idempotent and thread-safe.
    /// </summary>
    public override async Task LoadAsync()
    {
        await _inFlight.WaitAsync().ConfigureAwait(false);
        try
        {
            if (_workerLoaded) return;
            await LoadLockedAsync().ConfigureAwait(false);
        }
        finally
        {
            _inFlight.Release();
        }
    }

    /// <summary>
    /// Pre-loads the worker without holding a caller-visible lock re-entrancy. Assumes the caller
    /// already holds <see cref="_inFlight"/>. Spawns the worker (once), sends <c>load</c>, and mirrors
    /// the worker's diagnostics onto this proxy.
    /// </summary>
    private async Task LoadLockedAsync()
    {
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
        }, _oopOptions.StartupTimeout).ConfigureAwait(false);

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
        // Serialize the whole operation (auto-load + write + read) so two concurrent callers each
        // pair their own request with their own response (H1). A second caller simply waits.
        await _inFlight.WaitAsync().ConfigureAwait(false);
        try
        {
            if (!_workerLoaded)
                await LoadLockedAsync().ConfigureAwait(false);
            ThrowIfDisposed();

            await WriteAsync(new WorkerEnvelope { Kind = "predict", Payload = _serializer.SerializeInput(input) },
                _oopOptions.InferenceTimeout).ConfigureAwait(false);

            var result = await ReadAsync(_oopOptions.InferenceTimeout).ConfigureAwait(false)
                ?? throw new InvalidOperationException("Worker closed the pipe during predict.");

            if (result.Error is not null)
                throw new InvalidOperationException($"Worker predict failed: {result.Error}");
            if (result.Payload is null)
                throw new InvalidOperationException("Worker predict returned no payload.");

            return _serializer.DeserializeOutput(result.Payload);
        }
        finally
        {
            _inFlight.Release();
        }
    }

    // ---- plumbing -----------------------------------------------------------

    /// <summary>
    /// Spawn the worker process and wire its stdin/stdout. Override in tests to bypass process
    /// creation and inject in-memory streams (call <see cref="SetWorkerStreamsForTesting"/>).
    /// </summary>
    protected virtual void StartWorker()
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

    /// <summary>Test seam: wire the host↔worker streams directly without spawning a process.</summary>
    internal void SetWorkerStreamsForTesting(Stream toWorker, Stream fromWorker)
    {
        _toWorker = toWorker;
        _fromWorker = fromWorker;
    }

    private async Task WriteAsync(WorkerEnvelope envelope, TimeSpan timeout)
    {
        ThrowIfDisposed();
        // Race the write against a timeout. Redirected stdin pipes do not reliably honor
        // cancellation, so on timeout we kill the worker (closing the read end of stdin) to unblock
        // the pending write and then surface a TimeoutException (H2).
        var writeTask = WorkerFrame.WriteAsync(_toWorker!, envelope, CancellationToken.None);
        if (await Task.WhenAny(writeTask, Task.Delay(timeout)).ConfigureAwait(false) != writeTask)
        {
            KillWorker();
            throw new TimeoutException($"Worker did not accept a write within {timeout}.");
        }
        await writeTask.ConfigureAwait(false); // propagate any write fault
    }

    private async Task<WorkerEnvelope?> ReadAsync(TimeSpan timeout)
    {
        ThrowIfDisposed();
        // Redirected process stdout pipes do not promptly honor CancellationToken, so race the read
        // against a delay; on expiry kill the worker and dispose the read stream to unblock the
        // pending read, then surface a TimeoutException (H2).
        var readTask = WorkerFrame.ReadAsync(_fromWorker!, CancellationToken.None);
        if (await Task.WhenAny(readTask, Task.Delay(timeout)).ConfigureAwait(false) != readTask)
        {
            KillWorker();
            throw new TimeoutException($"Worker did not respond within {timeout}.");
        }
        return await readTask.ConfigureAwait(false);
    }

    /// <summary>
    /// Forcefully terminates the worker process and disposes the host-side streams so any pending
    /// blocked read/write on the pipe unblocks. Best-effort; swallow teardown errors.
    /// </summary>
    private void KillWorker()
    {
        try { if (_process is not null && !_process.HasExited) _process.Kill(); }
        catch { /* best-effort */ }
        try { _fromWorker?.Dispose(); } catch { /* best-effort */ }
        try { _toWorker?.Dispose(); } catch { /* best-effort */ }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(OutOfProcessBaseInferenceEngine<TInput, TOutput>));
    }

    /// <summary>
    /// Asynchronously tears down the worker: sends a graceful <c>shutdown</c> frame without
    /// blocking, then disposes the pipe/process and base session. Prefer
    /// <c>await using</c> over the synchronous <see cref="Dispose"/> — the synchronous path cannot
    /// (and must not) perform a sync-over-async write.
    /// </summary>
    public virtual async ValueTask DisposeAsync()
    {
        if (!_disposed)
        {
            await _inFlight.WaitAsync().ConfigureAwait(false);
            try
            {
                if (!_disposed)
                {
                    _disposed = true;
                    await SendShutdownFrameAsync().ConfigureAwait(false);
                    TearDownPipeAndProcess();
                }
            }
            finally
            {
                _inFlight.Release();
            }
            _inFlight.Dispose();
        }

        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>Best-effort asynchronous graceful-shutdown frame, only when a live pipe + process exist.</summary>
    private async ValueTask SendShutdownFrameAsync()
    {
        try
        {
            if (_toWorker is not null && _process is not null && !_process.HasExited)
            {
                try
                {
                    await WorkerFrame.WriteAsync(_toWorker, new WorkerEnvelope { Kind = "shutdown" }, CancellationToken.None)
                        .ConfigureAwait(false);
                }
                catch { /* best-effort */ }
            }
        }
        catch { /* swallow teardown errors */ }
    }

    /// <summary>
    /// Disposes the host side of the pipe and the worker process. Disposing stdin FIRST closes the
    /// worker's read end, which signals EOF and unblocks any pending stdout/stderr write so
    /// <see cref="Process.WaitForExit(int)"/> cannot deadlock; then the process is reaped/killed.
    /// </summary>
    private void TearDownPipeAndProcess()
    {
        try { _toWorker?.Dispose(); } catch { /* best-effort */ }
        try { _fromWorker?.Dispose(); } catch { /* best-effort */ }
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

    protected override void Dispose(bool disposing)
    {
        if (disposing && !_disposed)
        {
            // Take the in-flight lock (M6) so Dispose cannot tear down the pipe while a Load/Predict is
            // mid-write/read. In-flight IPC either completes or times out (H2) before releasing the lock.
            _inFlight.Wait();
            try
            {
                if (!_disposed)
                {
                    _disposed = true;
                    // Sync path: NO sync-over-async shutdown write (avoids thread-pool starvation /
                    // deadlock). The explicit "shutdown" frame is sent only on the async path
                    // (DisposeAsync). Here, TearDownPipeAndProcess closes stdin, which signals EOF to
                    // the worker; InferenceWorkerHost.RunAsync breaks its loop on the null message and
                    // releases the GPU/session.
                    TearDownPipeAndProcess();
                }
            }
            finally
            {
                _inFlight.Release();
            }

            _inFlight.Dispose();
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
