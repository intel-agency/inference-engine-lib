using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace InferenceEngine.Core.Worker;

/// <summary>
/// The child-process message loop. The consumer's worker executable calls
/// <see cref="RunAsync{TInput,TOutput}"/> with a factory that builds ITS engine (a normal
/// <see cref="BaseInferenceEngine{TInput,TOutput}"/> — exactly like the headless
/// <c>--diag inference</c> path). The loop reads length-prefixed JSON requests on stdin and writes
/// responses on stdout; logging passes through on stderr. Because this process loads no
/// Avalonia/GL, comgr's LLVM is the only LLVM in it → no Mesa/comgr collision.
/// </summary>
public static class InferenceWorkerHost
{
    /// <param name="engineFactory">Builds the consumer's engine from the host-supplied options.</param>
    /// <param name="serializer">Optional serializer; defaults to <see cref="SystemTextJsonWorkerSerializer{TInput,TOutput}"/>.</param>
    /// <param name="input">Optional input stream (stdin). Tests inject a pipe; production uses console stdin.</param>
    /// <param name="output">Optional output stream (stdout). Tests inject a pipe; production uses console stdout.</param>
    /// <param name="cancellationToken">Observed between requests.</param>
    /// <param name="logger">Optional logger for the worker loop itself (frame tracing, load
    /// outcomes, timings). stdout stays protocol-only; logger output goes to whatever sink the
    /// consumer configured (typically stderr, which the host proxy forwards).</param>
    public static async Task<int> RunAsync<TInput, TOutput>(
        Func<InferenceEngineOptions, BaseInferenceEngine<TInput, TOutput>> engineFactory,
        IWorkerSerializer<TInput, TOutput>? serializer = null,
        Stream? input = null,
        Stream? output = null,
        CancellationToken cancellationToken = default,
        ILogger? logger = null)
    {
        var log = logger ?? NullLogger.Instance;
        serializer ??= new SystemTextJsonWorkerSerializer<TInput, TOutput>();

        // Redirect engine logging away from stdout (stdout is protocol-only); use stderr.
        var stdin = input ?? Console.OpenStandardInput();
        var stdout = output ?? Console.OpenStandardOutput();

        // In production (no injected streams), redirect Console.Out to Console.Error so any stray
        // Console.WriteLine from managed code or libraries cannot corrupt the stdout IPC channel.
        // (When streams are injected — i.e. in tests — leave the console untouched.)
        if (output is null)
        {
            Console.SetOut(Console.Error);
        }
        // Keep stdout free of any stray text writes — protocol frames only.

        log.LogInformation("worker loop: started (pid={Pid}, {Mode})",
            Environment.ProcessId, output is null ? "console stdio" : "injected streams");

        BaseInferenceEngine<TInput, TOutput>? engine = null;
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                WorkerEnvelope? msg;
                try
                {
                    msg = await WorkerFrame.ReadAsync(stdin, cancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    log.LogInformation("worker loop: cancelled; exiting");
                    break;
                }
                if (msg is null) // host closed stdin → exit
                {
                    log.LogInformation("worker loop: stdin closed by host; exiting");
                    break;
                }

                switch (msg.Kind)
                {
                    case "load":
                        engine = await HandleLoadAsync(engineFactory, serializer, msg, stdout, log, cancellationToken).ConfigureAwait(false);
                        break;
                    case "predict" when engine is not null:
                        await HandlePredictAsync(engine, serializer, msg, stdout, log, cancellationToken).ConfigureAwait(false);
                        break;
                    case "shutdown":
                        log.LogInformation("worker loop: shutdown received; exiting 0");
                        return 0;
                    default:
                        log.LogWarning("worker loop: unknown or unexpected message kind '{Kind}'", msg.Kind);
                        await WorkerFrame.WriteAsync(stdout,
                            new WorkerEnvelope { Kind = "result", Error = $"Unknown or unexpected message kind: '{msg.Kind}'." },
                            cancellationToken).ConfigureAwait(false);
                        break;
                }
            }
        }
        finally
        {
            (engine as IDisposable)?.Dispose();
        }
        return 0;
    }

    private static async Task<BaseInferenceEngine<TInput, TOutput>?> HandleLoadAsync<TInput, TOutput>(
        Func<InferenceEngineOptions, BaseInferenceEngine<TInput, TOutput>> engineFactory,
        IWorkerSerializer<TInput, TOutput> serializer,
        WorkerEnvelope msg,
        Stream stdout,
        ILogger log,
        CancellationToken ct)
    {
        var o = msg.Options ?? new WorkerEngineOptions();
        var engineOptions = new InferenceEngineOptions
        {
            ModelPath = o.ModelPath,
            UseGpuAcceleration = o.UseGpuAcceleration,
            DeviceId = o.DeviceId,
            WarmupOnLoad = o.WarmupOnLoad,
            MIGraphXModelCacheDirectory = o.MIGraphXModelCacheDirectory,
        };

        var sw = Stopwatch.StartNew();
        BaseInferenceEngine<TInput, TOutput> engine;
        try
        {
            engine = engineFactory(engineOptions);
        }
        catch (Exception ex)
        {
            log.LogError(ex, "worker load: engine factory threw for engine '{EngineId}'", msg.EngineId);
            await WorkerFrame.WriteAsync(stdout, new WorkerEnvelope { Kind = "loaded", Error = ex.ToString() }, ct).ConfigureAwait(false);
            return null;
        }

        try
        {
            await engine.LoadAsync().ConfigureAwait(false); // real EP selection happens here
            log.LogInformation("worker load: engine '{EngineId}' loaded in {Elapsed}ms (provider={Provider}, fallback={IsFallback}, reason={FallbackReason})",
                msg.EngineId, sw.ElapsedMilliseconds, engine.EngineInfo.Provider, engine.EngineInfo.IsFallback, engine.EngineInfo.FallbackReason);
            await WorkerFrame.WriteAsync(stdout, new WorkerEnvelope { Kind = "loaded", EngineInfo = engine.EngineInfo }, ct).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            log.LogError(ex, "worker load: engine '{EngineId}' LoadAsync failed after {Elapsed}ms",
                msg.EngineId, sw.ElapsedMilliseconds);
            await WorkerFrame.WriteAsync(stdout,
                new WorkerEnvelope { Kind = "loaded", EngineInfo = engine.EngineInfo, Error = ex.ToString() },
                ct).ConfigureAwait(false);
        }
        return engine;
    }

    private static async Task HandlePredictAsync<TInput, TOutput>(
        BaseInferenceEngine<TInput, TOutput> engine,
        IWorkerSerializer<TInput, TOutput> serializer,
        WorkerEnvelope msg,
        Stream stdout,
        ILogger log,
        CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            var input = serializer.DeserializeInput(msg.Payload ?? string.Empty);
            var output = await engine.PredictAsync(input).ConfigureAwait(false);
            log.LogDebug("worker predict: ok in {Elapsed}ms", sw.ElapsedMilliseconds);
            await WorkerFrame.WriteAsync(stdout,
                new WorkerEnvelope { Kind = "result", Payload = serializer.SerializeOutput(output) },
                ct).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            log.LogError(ex, "worker predict: failed after {Elapsed}ms", sw.ElapsedMilliseconds);
            await WorkerFrame.WriteAsync(stdout,
                new WorkerEnvelope { Kind = "result", Error = ex.ToString() },
                ct).ConfigureAwait(false);
        }
    }
}
