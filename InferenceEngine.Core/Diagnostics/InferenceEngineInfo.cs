using System.Collections.Immutable;

namespace InferenceEngine.Core.Diagnostics;

/// <summary>Immutable snapshot of a loaded inference engine's provider, platform, and diagnostic trail.</summary>
public sealed record InferenceEngineInfo
{
    public InferenceProvider Provider { get; init; } = InferenceProvider.Unknown;
    public HostPlatform Platform { get; init; } = HostPlatformDetector.Current();
    public int DeviceId { get; init; }
    public bool GpuAccelerationRequested { get; init; }

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
