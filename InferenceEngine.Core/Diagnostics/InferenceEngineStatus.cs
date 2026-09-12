namespace InferenceEngine.Core.Diagnostics;

/// <summary>Lifecycle position of an inference engine.</summary>
public enum EngineLoadState
{
    /// <summary>Never loaded, disposed, or no load attempt yet.</summary>
    NotLoaded = 0,

    /// <summary><c>LoadAsync</c> is in flight (model resolution, provider selection, session
    /// construction). For out-of-process engines this spans worker spawn + IPC handshake.</summary>
    Loading,

    /// <summary>Session ready (possibly via CPU fallback — inspect
    /// <see cref="InferenceEngineStatus.Info"/> for provider/fallback).</summary>
    Loaded,

    /// <summary><c>LoadAsync</c> threw. <see cref="InferenceEngineStatus.Info"/> carries the
    /// partial provider-selection trail when available; <see cref="InferenceEngineStatus.LoadError"/>
    /// carries the error.</summary>
    Failed,
}

/// <summary>
/// The consumer-facing object model for one engine's state: a single immutable snapshot covering
/// lifecycle (<see cref="State"/>), runtime details (<see cref="Info"/>) and failure details
/// (<see cref="LoadError"/>). Delivered both as a pollable property
/// (<c>BaseInferenceEngine{TInput,TOutput}.Status</c>, mirrored onto out-of-process proxies) and
/// via the <c>InferenceEngineOptions.OnStatusChanged</c> callback on every transition.
/// </summary>
/// <remarks>
/// The graph consumers program against:
/// <code>
/// InferenceEngineOptions                    // configuration (model, GPU flags, callbacks)
///   └─ OnStatusChanged(InferenceEngineStatus)
///        ├─ State: EngineLoadState          // NotLoaded → Loading → Loaded | Failed
///        ├─ Info: InferenceEngineInfo       // Provider, DeviceId, ModelName/Path/Source,
///        │                                    IsFallback + FallbackReason, Checks trail
///        │                                    (partial — checks only — when State=Failed)
///        └─ LoadError: string?              // populated when State=Failed
/// </code>
/// <see cref="InferenceEngineOptions.OnSessionInitialized"/> remains as the success-only hook;
/// new consumers should prefer <c>OnStatusChanged</c>, which also covers failures.
/// </remarks>
public sealed record InferenceEngineStatus
{
    public EngineLoadState State { get; init; } = EngineLoadState.NotLoaded;

    /// <summary>Runtime details. Full when <see cref="EngineLoadState.Loaded"/>; partial
    /// (provider-selection checks, <c>Provider=Unknown</c>) when <see cref="EngineLoadState.Failed"/>;
    /// null while NotLoaded/Loading.</summary>
    public InferenceEngineInfo? Info { get; init; }

    /// <summary>The error that failed the load (exception message, or the worker-reported error
    /// for out-of-process engines). Null unless <see cref="State"/> is Failed.</summary>
    public string? LoadError { get; init; }

    /// <summary>When this snapshot was produced (UTC).</summary>
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>True when the engine is loaded AND running on a hardware provider
    /// (not CPU, not unknown).</summary>
    public bool IsGpuActive =>
        State == EngineLoadState.Loaded
        && Info is not null
        && Info.Provider is not (InferenceProvider.Cpu or InferenceProvider.Unknown);

    public static InferenceEngineStatus NotLoaded { get; } = new();
}
