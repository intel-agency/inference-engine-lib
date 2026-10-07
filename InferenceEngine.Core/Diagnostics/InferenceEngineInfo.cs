using System.Collections.Immutable;

namespace InferenceEngine.Core.Diagnostics;

/// <summary>How the engine resolved the model it loaded.</summary>
public enum ModelSource
{
    /// <summary>Not reported (engine not loaded, or an older engine version).</summary>
    Unspecified = 0,

    /// <summary>The consumer supplied the location explicitly
    /// (<see cref="InferenceEngineOptions.ModelPath"/> or a custom <c>ModelProvider</c>).</summary>
    ExplicitPath,

    /// <summary>The engine's default registry/provider resolved it (downloading to the model
    /// cache on first use) — i.e. the consumer did not pin a location.</summary>
    RegistryManaged,
}

/// <summary>Immutable snapshot of a loaded inference engine's provider, platform, and diagnostic trail.</summary>
public sealed record InferenceEngineInfo
{
    public InferenceProvider Provider { get; init; } = InferenceProvider.Unknown;
    public HostPlatform Platform { get; init; } = HostPlatformDetector.Current();
    public int DeviceId { get; init; }
    public bool GpuAccelerationRequested { get; init; }

    /// <summary>Model file name the engine loaded (e.g. "yolov8n-face.onnx"). Empty until loaded.</summary>
    public string ModelName { get; init; } = string.Empty;

    /// <summary>Absolute path the model was actually loaded from (consumer-supplied asset path or
    /// the registry's cache location). Empty until loaded.</summary>
    public string ModelPath { get; init; } = string.Empty;

    /// <summary>How <see cref="ModelPath"/> was determined.</summary>
    public ModelSource ModelSource { get; init; } = ModelSource.Unspecified;

    /// <summary>True when a fall-back occurred (FallbackReason != None).</summary>
    public bool IsFallback { get; init; }
    public InferenceFallbackReason FallbackReason { get; init; } = InferenceFallbackReason.None;

    /// <summary>Per-provider key/value options actually applied.</summary>
    public IReadOnlyDictionary<string, string> ProviderOptions { get; init; }
        = ImmutableDictionary<string, string>.Empty;

    /// <summary>Not populated in v1; always <see cref="string.Empty"/>.</summary>
    public string OnnxRuntimeVersion { get; init; } = string.Empty;

    public DateTimeOffset SessionCreatedAt { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>Ordered diagnostic trail of every check performed during provider selection.</summary>
    public IReadOnlyList<ProviderDiagnosticCheck> Checks { get; init; } = Array.Empty<ProviderDiagnosticCheck>();

    public static InferenceEngineInfo Empty { get; } = new();
}
