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
    // Test seam: the OS/architecture probes are injectable so tests can drive every
    // branch of Current() hermetically on any host (the real RuntimeInformation calls
    // answer for the host OS only, leaving the other branches permanently uncovered).
    // Internal and nullable-probe-free: production callers always go through Current(),
    // whose public surface is unchanged. Tests must restore the defaults afterwards.
    internal static Func<string, bool> IsOSPlatformProbe { get; set; } =
        static name => RuntimeInformation.IsOSPlatform(OSPlatform.Create(name));
    internal static Func<Architecture> ProcessArchitectureProbe { get; set; } =
        static () => RuntimeInformation.ProcessArchitecture;

    /// <summary>Detects the current host platform. See <see cref="HostPlatform"/>.</summary>
    public static HostPlatform Current()
    {
        if (IsOSPlatformProbe("Linux"))
        {
            return ProcessArchitectureProbe() switch
            {
                Architecture.X64 => HostPlatform.LinuxX64,
                Architecture.Arm64 => HostPlatform.LinuxArm64,
                _ => HostPlatform.Unknown,
            };
        }
        if (IsOSPlatformProbe("OSX"))
        {
            return ProcessArchitectureProbe() switch
            {
                Architecture.X64 => HostPlatform.OSX64,
                Architecture.Arm64 => HostPlatform.OSXArm64,
                _ => HostPlatform.Unknown,
            };
        }
        if (IsOSPlatformProbe("Windows"))
        {
            return ProcessArchitectureProbe() switch
            {
                Architecture.X64 => HostPlatform.WinX64,
                Architecture.Arm64 => HostPlatform.WinArm64,
                _ => HostPlatform.Unknown,
            };
        }
        return HostPlatform.Unknown;
    }
}
