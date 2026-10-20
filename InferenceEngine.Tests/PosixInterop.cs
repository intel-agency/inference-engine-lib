namespace InferenceEngine.Tests;

/// <summary>POSIX interop used by filesystem-permission tests (no-ops conceptually on Windows).</summary>
internal static class PosixInterop
{
    [System.Runtime.InteropServices.DllImport("libc", SetLastError = true)]
    private static extern int chmod(string path, ushort mode);

    public static bool IsLinux => System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux);

    /// <summary>Sets POSIX permissions; returns false when the syscall fails or the host is not Linux.</summary>
    public static bool TryChmod(string path, ushort mode)
        => IsLinux && chmod(path, mode) == 0;

    public const ushort OwnerReadWriteExecute = 0b111_101_101; // 0755
    public const ushort OwnerReadOnlyDir = 0b101_101_101;      // 0555
    public const ushort None = 0b000_000_000;                  // 0000

    /// <summary>Permission-based tests are meaningless for root (chmod does not restrict root).</summary>
    public static bool RunningAsRoot => Environment.UserName == "root" || Environment.GetEnvironmentVariable("USER") == "root";
}
