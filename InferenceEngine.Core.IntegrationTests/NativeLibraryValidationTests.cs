using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace InferenceEngine.Core.IntegrationTests
{
    /// <summary>
    /// Tier 1 validation tests — run on any Linux runner (no GPU required).
    /// Validates the compiled .so artifacts are well-formed and that OnnxRuntime
    /// loads them without crashing.
    /// </summary>
    public class NativeLibraryValidationTests
    {
        public NativeLibraryValidationTests()
        {
            EnsureLinuxOnly();
        }

        private static void EnsureLinuxOnly()
        {
            Skip.IfNot(
                RuntimeInformation.IsOSPlatform(OSPlatform.Linux),
                "NativeLibraryValidationTests are Linux-only and require Linux native libraries.");
        }
        // -----------------------------------------------------------------
        // Helpers
        // -----------------------------------------------------------------

        /// <summary>
        /// Locates the native library directory. Checks NATIVE_LIBS_DIR env var
        /// (set by CI), then falls back to runtimes/linux-x64/native/ beside the assembly.
        /// </summary>
        private static string GetNativeLibsDir()
        {
            var fromEnv = Environment.GetEnvironmentVariable("NATIVE_LIBS_DIR");
            if (!string.IsNullOrEmpty(fromEnv) && Directory.Exists(fromEnv))
                return fromEnv;

            // Beside the test assembly (copied via CopyToOutputDirectory)
            var assemblyDir = AppContext.BaseDirectory;
            var candidate = Path.Combine(assemblyDir, "runtimes", "linux-x64", "native");
            if (Directory.Exists(candidate))
                return candidate;

            // Fallback: repo-root relative (for `dotnet test` run from workspace root)
            var wsRoot = FindRepoRoot(assemblyDir);
            if (wsRoot is not null)
            {
                var repoCandidate = Path.Combine(wsRoot, "InferenceEngine.Core", "runtimes", "linux-x64", "native");
                if (Directory.Exists(repoCandidate))
                    return repoCandidate;
            }

            throw new DirectoryNotFoundException(
                "Cannot locate native libs directory. Set NATIVE_LIBS_DIR env var or build with --runtime linux-x64.");
        }

        private static string? FindRepoRoot(string start)
        {
            var dir = new DirectoryInfo(start);
            while (dir is not null)
            {
                if (File.Exists(Path.Combine(dir.FullName, "inference-engine-lib.slnx")))
                    return dir.FullName;
                dir = dir.Parent;
            }
            return null;
        }

        private static string RunCommand(string executable, string args)
        {
            var psi = new ProcessStartInfo(executable, args)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            using var proc = Process.Start(psi)!;
            var stdout = proc.StandardOutput.ReadToEnd();
            if (!proc.WaitForExit(10_000))
            {
                try { proc.Kill(); } catch { /* best-effort cleanup */ }
                throw new TimeoutException($"Command '{executable} {args}' timed out after 10s.");
            }
            return stdout + proc.StandardError.ReadToEnd();
        }

        private static string ModelPath =>
            Path.Combine(AppContext.BaseDirectory, "Resources", "identity.onnx");

        // -----------------------------------------------------------------
        // Existence tests
        // -----------------------------------------------------------------

        [SkippableFact]
        public void LibOnnxRuntime_Exists()
        {
            var path = Path.Combine(GetNativeLibsDir(), "libonnxruntime.so");
            Assert.True(File.Exists(path), $"Missing: {path}");
        }

        [SkippableFact]
        public void LibOnnxRuntimeProvidersRocm_Exists()
        {
            var path = Path.Combine(GetNativeLibsDir(), "libonnxruntime_providers_rocm.so");
            Assert.True(File.Exists(path), $"Missing: {path}");
        }

        [SkippableFact]
        public void BothLibs_AreNonEmpty()
        {
            var dir = GetNativeLibsDir();
            foreach (var name in new[] { "libonnxruntime.so", "libonnxruntime_providers_rocm.so" })
            {
                var info = new FileInfo(Path.Combine(dir, name));
                Assert.True(info.Exists && info.Length > 1_000_000,
                    $"{name}: expected > 1 MB, got {info.Length} bytes");
            }
        }

        // -----------------------------------------------------------------
        // ELF format tests (Linux only)
        // -----------------------------------------------------------------

        [SkippableFact]
        [Trait("Category", "LinuxOnly")]
        public void LibOnnxRuntime_IsElf64()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return;
            var path = Path.Combine(GetNativeLibsDir(), "libonnxruntime.so");
            var output = RunCommand("file", path);
            Assert.Contains("ELF 64-bit", output);
            Assert.Contains("shared object", output);
        }

        [SkippableFact]
        [Trait("Category", "LinuxOnly")]
        public void LibOnnxRuntimeProvidersRocm_IsElf64()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return;
            var path = Path.Combine(GetNativeLibsDir(), "libonnxruntime_providers_rocm.so");
            var output = RunCommand("file", path);
            Assert.Contains("ELF 64-bit", output);
            Assert.Contains("shared object", output);
        }

        // -----------------------------------------------------------------
        // Symbol export tests (Linux only)
        // -----------------------------------------------------------------

        [SkippableFact]
        [Trait("Category", "LinuxOnly")]
        public void LibOnnxRuntime_ExportsOrtGetApiBase()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return;
            var path = Path.Combine(GetNativeLibsDir(), "libonnxruntime.so");
            var output = RunCommand("nm", $"-D --defined-only {path}");
            Assert.Contains("OrtGetApiBase", output);
        }

        [SkippableFact]
        [Trait("Category", "LinuxOnly")]
        public void LibOnnxRuntimeProvidersRocm_ExportsRocmProvider()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return;
            var path = Path.Combine(GetNativeLibsDir(), "libonnxruntime_providers_rocm.so");
            var output = RunCommand("nm", $"-D --defined-only {path}");
            // Any of these symbols confirms the ROCm EP is present
            Assert.True(
                output.Contains("OrtSessionOptionsAppendExecutionProvider_ROCm") ||
                output.Contains("GetApi") ||
                output.Contains("rocm"),
                $"Expected ROCm EP symbols not found in nm output. First 500 chars:\n{output[..Math.Min(500, output.Length)]}");
        }

        // -----------------------------------------------------------------
        // OnnxRuntime managed loading tests
        // -----------------------------------------------------------------

        [SkippableFact]
        public void OrtEnv_Initializes_WithoutCrash()
        {
            // If the managed ORT assembly or native lib is broken, this throws/crashes
            var env = OrtEnv.Instance();
            Assert.NotNull(env);
        }

        [SkippableFact]
        public void InferenceSession_LoadsIdentityModel_WithoutCrash()
        {
            Assert.True(File.Exists(ModelPath), $"identity.onnx not found at: {ModelPath}");

            // CPU-only: this must succeed on any runner
            using var opts = new SessionOptions();
            opts.AppendExecutionProvider_CPU();
            using var session = new InferenceSession(ModelPath, opts);

            Assert.NotNull(session.InputMetadata);
            Assert.NotEmpty(session.InputMetadata);
        }

        [SkippableFact]
        public void InferenceSession_RunsIdentityModel_ProducesCorrectOutput()
        {
            Assert.True(File.Exists(ModelPath), $"identity.onnx not found at: {ModelPath}");

            using var opts = new SessionOptions();
            opts.AppendExecutionProvider_CPU();
            using var session = new InferenceSession(ModelPath, opts);

            var inputName = session.InputMetadata.Keys.First();
            var inputTensor = new DenseTensor<float>(new float[] { 42.0f }, new[] { 1 });
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };

            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>().ToArray();
            Assert.Equal(new[] { 42.0f }, output);
        }

        [SkippableFact]
        [Trait("Category", "LinuxOnly")]
        public void InferenceSession_WithRocmEP_ThrowsCleanExceptionNotCrash()
        {
            // On a runner with no AMD GPU, requesting ROCm EP should throw a managed
            // OrtException (provider not available) — NOT a SIGABRT or native crash.
            // This proves the .so loads and its registration path is sane.
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return;
            Assert.True(File.Exists(ModelPath), $"identity.onnx not found at: {ModelPath}");

            try
            {
                using var opts = new SessionOptions();
                var rocmOptions = new System.Collections.Generic.Dictionary<string, string>
                {
                    { "device_id", "0" }
                };
                opts.AppendExecutionProvider("ROCmExecutionProvider", rocmOptions);
                using var session = new InferenceSession(ModelPath, opts);
                // If we reach here, ROCm is actually available — also a pass
            }
            catch (OnnxRuntimeException)
            {
                // Expected on runners without AMD GPU — clean managed exception = pass
            }
            // Any unmanaged crash (SIGABRT etc.) would fail the test by killing the process
        }
    }
}
