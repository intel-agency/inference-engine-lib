using System.Runtime.InteropServices;

namespace InferenceEngine.Core.Diagnostics;

/// <summary>Identifies the host platform and architecture.</summary>
public enum HostPlatform
{
    Unknown = 0,
    LinuxX64,
    LinuxArm64,
    WinX64,
    WinArm64,
    OSX64,
    OSXArm64,
}

/// <summary>
/// Detects the host platform at runtime. Provided for convenience; the primary
/// API for platform information is <see cref="InferenceEngineInfo.Platform"/>.
/// </summary>
public static class HostPlatformDetector
{
    public static HostPlatform Current()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            return RuntimeInformation.ProcessArchitecture == Architecture.X64
                ? HostPlatform.LinuxX64
                : HostPlatform.LinuxArm64;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return RuntimeInformation.ProcessArchitecture == Architecture.X64
                ? HostPlatform.OSX64
                : HostPlatform.OSXArm64;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return RuntimeInformation.ProcessArchitecture == Architecture.X64
                ? HostPlatform.WinX64
                : HostPlatform.WinArm64;
        return HostPlatform.Unknown;
    }
}
