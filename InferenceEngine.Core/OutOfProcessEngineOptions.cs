using System;
using System.Collections.Generic;

namespace InferenceEngine.Core;

/// <summary>Options for an <see cref="OutOfProcessBaseInferenceEngine{TInput,TOutput}"/>.</summary>
public sealed class OutOfProcessEngineOptions
{
    /// <summary>Absolute path to the consumer's worker executable (the same app, launched with
    /// <see cref="WorkerArguments"/> so it enters <see cref="Worker.InferenceWorkerHost"/>).</summary>
    public required string WorkerExecutablePath { get; init; }

    /// <summary>Arguments appended after <see cref="WorkerExecutablePath"/> to enter the worker mode.
    /// Default is <c>["--worker", "inference"]</c>; match the consumer's <c>Program.cs</c> entry point.
    /// A list (not a single string) avoids fragility with arguments containing spaces.</summary>
    public IReadOnlyList<string> WorkerArguments { get; init; } = new[] { "--worker", "inference" };

    /// <summary>Consumer-defined id identifying which engine the worker should construct.</summary>
    public string EngineId { get; init; } = string.Empty;

    /// <summary>Environment variables applied to the WORKER process only (e.g.
    /// <c>HSA_OVERRIDE_GFX_VERSION=10.3.0</c>). Never applied to the host (which must stay GL-free
    /// of ROCm/comgr). Inherits the host's environment by default; these augment/override it.</summary>
    public IReadOnlyDictionary<string, string> WorkerEnvironment { get; init; } = new Dictionary<string, string>();

    /// <summary>In-process engine options mirrored to the worker (model path, GPU toggle, warmup).
    /// <see cref="InferenceEngineOptions.OnSessionInitialized"/> does not cross the boundary; the
    /// proxy mirrors <see cref="BaseInferenceEngine{TInput,TOutput}.EngineInfo"/> from the worker's
    /// Load response and fires the callback on the host via <see cref="BaseInferenceEngine{TInput,TOutput}"/>.</summary>
    public InferenceEngineOptions EngineOptions { get; init; } = new();

    /// <summary>Timeout for the worker to start, handle <c>load</c>, and emit its EngineInfo.
    /// The first MIGraphX program compile can take ~85s; default is generous.</summary>
    public TimeSpan StartupTimeout { get; init; } = TimeSpan.FromMinutes(3);

    /// <summary>Timeout for a single <c>predict</c> round-trip.</summary>
    public TimeSpan InferenceTimeout { get; init; } = TimeSpan.FromSeconds(60);
}
