namespace InferenceEngine.Core.Diagnostics;

/// <summary>Reason the engine fell back to CPU (or <see cref="None"/> if no fallback occurred).</summary>
public enum InferenceFallbackReason
{
    None = 0,
    UserDisabledGpu,
    UnsupportedPlatform,
    PlatformProbeFailed,
    KfdDeviceMissing,
    HsaRuntimeUnloadable,
    ProviderAppendFailed,
    SessionConstructionFailed,
    NativeLibraryLoadFailed,
}
