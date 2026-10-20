using System.Runtime.InteropServices;
using InferenceEngine.Core.Diagnostics;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// Removes the parallelization that xUnit otherwise applies across test classes: these tests
/// swap <see cref="HostPlatformDetector"/>'s internal probes, and other engines under test
/// read <see cref="HostPlatformDetector.Current"/> (via <c>InferenceEngineInfo</c>'s default
/// platform) concurrently. Tests restore the real probes in a finally block.
/// </summary>
[CollectionDefinition("HostPlatformDetector", DisableParallelization = true)]
public sealed class HostPlatformDetectorCollection { }

[Collection("HostPlatformDetector")]
public sealed class HostPlatformDetectorTests
{
    [Theory]
    [InlineData("Linux", Architecture.X64, HostPlatform.LinuxX64)]
    [InlineData("Linux", Architecture.Arm64, HostPlatform.LinuxArm64)]
    [InlineData("Linux", Architecture.X86, HostPlatform.Unknown)]   // unsupported Linux arch
    [InlineData("Linux", Architecture.Arm, HostPlatform.Unknown)]
    [InlineData("OSX", Architecture.X64, HostPlatform.OSX64)]
    [InlineData("OSX", Architecture.Arm64, HostPlatform.OSXArm64)]
    [InlineData("OSX", Architecture.Wasm, HostPlatform.Unknown)]    // unsupported macOS arch
    [InlineData("Windows", Architecture.X64, HostPlatform.WinX64)]
    [InlineData("Windows", Architecture.Arm64, HostPlatform.WinArm64)]
    [InlineData("Windows", Architecture.LoongArch64, HostPlatform.Unknown)] // unsupported Windows arch
    [InlineData("FreeBSD", Architecture.X64, HostPlatform.Unknown)]  // no supported OS matches
    [InlineData("", Architecture.X64, HostPlatform.Unknown)]         // OS probe answers false for everything
    public void Current_MapsOsAndArchitectureToPlatform(string os, Architecture arch, HostPlatform expected)
    {
        try
        {
            HostPlatformDetector.IsOSPlatformProbe = name => name == os;
            HostPlatformDetector.ProcessArchitectureProbe = () => arch;

            Assert.Equal(expected, HostPlatformDetector.Current());
        }
        finally
        {
            RestoreDefaultProbes();
        }
    }

    [Fact]
    public void Current_LinuxProbeWins_WhenMultipleOsProbesMatch()
    {
        // The OS checks short-circuit in Linux → macOS → Windows order; a probe that
        // answers true for several names must resolve to the first (Linux) branch.
        try
        {
            HostPlatformDetector.IsOSPlatformProbe = _ => true;
            HostPlatformDetector.ProcessArchitectureProbe = () => Architecture.X64;

            Assert.Equal(HostPlatform.LinuxX64, HostPlatformDetector.Current());
        }
        finally
        {
            RestoreDefaultProbes();
        }
    }

    [Fact]
    public void Current_WithDefaultProbes_ReportsTheRealHostPlatform()
    {
        var expected = RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
            ? RuntimeInformation.ProcessArchitecture == Architecture.X64 ? HostPlatform.LinuxX64
                : RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? HostPlatform.LinuxArm64
                : HostPlatform.Unknown
            : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                ? RuntimeInformation.ProcessArchitecture == Architecture.X64 ? HostPlatform.OSX64
                    : RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? HostPlatform.OSXArm64
                    : HostPlatform.Unknown
            : RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? RuntimeInformation.ProcessArchitecture == Architecture.X64 ? HostPlatform.WinX64
                    : RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? HostPlatform.WinArm64
                    : HostPlatform.Unknown
            : HostPlatform.Unknown;

        Assert.Equal(expected, HostPlatformDetector.Current());
    }

    private static void RestoreDefaultProbes()
    {
        HostPlatformDetector.IsOSPlatformProbe =
            static name => RuntimeInformation.IsOSPlatform(OSPlatform.Create(name));
        HostPlatformDetector.ProcessArchitectureProbe =
            static () => RuntimeInformation.ProcessArchitecture;
    }
}
