using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using InferenceEngine.Core.Diagnostics;
using InferenceEngine.Core.ModelProviders;

namespace InferenceEngine.Core
{
    /// <summary>
    /// The core abstract base class that handles ONNX Runtime session management,
    /// cross-platform hardware acceleration detection, and resource cleanup.
    /// Supports state passing between Pre and Post processing steps via a context object.
    /// </summary>
    public abstract class BaseInferenceEngine<TInput, TOutput> : IInferenceEngine<TInput, TOutput>
    {
        protected readonly InferenceEngineOptions _options;
        protected readonly ILogger _logger;
        protected InferenceSession? _session;
        protected RunOptions? _runOptions;

        private bool _isLoaded = false;
        private IModelProvider? _modelProvider;
        private List<ProviderDiagnosticCheck> _pendingChecks = new();
        private int _nextOrder = 0;

        public IReadOnlyDictionary<string, NodeMetadata>? InputMetadata { get; private set; }
        public IReadOnlyDictionary<string, NodeMetadata>? OutputMetadata { get; private set; }
        public string? PrimaryInputName { get; private set; }

        /// <summary>
        /// Returns <see langword="true"/> when the ONNX session was successfully initialized
        /// with the MIGraphX execution provider on AMD ROCm hardware. <see langword="false"/> on non-Linux-x64 platforms,
        /// when MIGraphX initialization failed, or when CPU fallback was used.
        /// </summary>
        public bool IsRocmSessionActive { get; private set; }

        /// <summary>
        /// Snapshot of the engine's loaded state. Returns <see cref="InferenceEngineInfo.Empty"/>
        /// until <see cref="LoadAsync"/> completes. Thread-safe: reference assignment is atomic
        /// and <see cref="InferenceEngineInfo"/> is an immutable record.
        /// </summary>
        public InferenceEngineInfo EngineInfo { get; private set; } = InferenceEngineInfo.Empty;

        // --- Model Specification ---

        /// <summary>
        /// The name of the model file (e.g., "yolo11n.onnx").
        /// Derived classes must implement this to specify which model they require.
        /// </summary>
        protected abstract string ModelName { get; }

        /// <summary>
        /// URLs where the model can be downloaded from, in priority order.
        /// Derived classes can override to provide custom model sources.
        /// If not overridden, the base engine will use the default ModelRegistry to find URLs.
        /// </summary>
        protected virtual string[]? ModelDownloadUrls => null;

        /// <summary>
        /// Model registry for looking up download URLs by model name.
        /// Override to use a custom registry; defaults to the built-in registry.
        /// </summary>
        protected virtual ModelRegistry ModelRegistry => _lazyRegistry.Value;

        private readonly Lazy<ModelRegistry> _lazyRegistry;

        /// <summary>
        /// Base directory for model cache. Defaults to LocalApplicationData.
        /// Derived classes can override to use a custom cache location.
        /// </summary>
        protected virtual string ModelCacheDirectory => Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "InferenceEngine",
            "Models");

        protected BaseInferenceEngine(InferenceEngineOptions options, ILogger? logger = null)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger.Instance;
            _lazyRegistry = new Lazy<ModelRegistry>(() => new ModelRegistry(logger: _logger), LazyThreadSafetyMode.ExecutionAndPublication);
        }

        // --- Abstract Methods ---

        /// <summary>
        /// Takes raw input and returns ONNX tensors plus an optional context object.
        /// The context object is passed to PostProcessWithContext, allowing you to share state
        /// without overriding the main pipeline.
        /// </summary>
        protected abstract (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(TInput input);

        /// <summary>
        /// Takes raw ONNX output and the context object from PreProcess, returning the final result.
        /// </summary>
        protected abstract TOutput PostProcessWithContext(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context);

        // --- Public API ---

        /// <summary>
        /// Gets or creates the model provider.
        /// Uses options.ModelProvider if set, otherwise creates from ModelName/ModelDownloadUrls/ModelRegistry.
        /// </summary>
        private IModelProvider GetModelProvider()
        {
            if (_options.ModelProvider is not null)
            {
                return _options.ModelProvider;
            }

            if (_modelProvider is null)
            {
                var urls = ModelDownloadUrls;

                if (urls is null || urls.Length == 0)
                {
                    urls = ModelRegistry.GetModelUrls(ModelName);

                    if (urls is null || urls.Length == 0)
                    {
                        throw new InvalidOperationException(
                            $"Engine '{GetType().Name}' does not specify ModelDownloadUrls and model '{ModelName}' " +
                            $"was not found in the ModelRegistry. Either: " +
                            $"1) Override ModelDownloadUrls in your engine class, " +
                            $"2) Add '{ModelName}' to the model registry (models.json), or " +
                            $"3) Set {nameof(InferenceEngineOptions)}.{nameof(InferenceEngineOptions.ModelProvider)} explicitly.");
                    }

                    _logger.LogInformation("Found model '{ModelName}' in registry with {Count} URL(s)", ModelName, urls.Length);
                }

                _modelProvider = new DownloadingModelProvider(
                    modelName: ModelName,
                    downloadUrls: urls,
                    cacheDirectory: ModelCacheDirectory,
                    logger: _logger);
            }

            return _modelProvider;
        }

        /// <summary>
        /// Initializes the inference session with the best available hardware provider.
        /// </summary>
        public virtual async Task LoadAsync()
        {
            if (_isLoaded) return;

            _logger.LogInformation("Initializing Inference Engine...");

            var modelPath = await GetModelProvider().GetModelPathAsync(CancellationToken.None);

            await Task.Run(() =>
            {
                try
                {
                    _pendingChecks = new List<ProviderDiagnosticCheck>();
                    _nextOrder = 0;

                    InferenceProvider provider = InferenceProvider.Unknown;
                    InferenceFallbackReason fallback;
                    int deviceId;

                    SessionOptions? sessionOptions = null;

                    try
                    {
                        sessionOptions = ConfiguredSessionOptions(out provider, out fallback, out deviceId);

                        try
                        {
                            _session = new InferenceSession(modelPath, sessionOptions);
                            if (fallback != InferenceFallbackReason.None)
                            {
                                _pendingChecks.Add(ProviderDiagnosticCheck.Pass("cpu.fallback-options", "CreateCpuOnlySessionOptions() succeeded", "Fallback", NextOrder()));
                                _pendingChecks.Add(ProviderDiagnosticCheck.Pass("cpu.fallback-session", "CPU InferenceSession constructed", "Fallback", NextOrder()));
                            }
                            else
                            {
                                AppendSessionCheck(provider, passed: true);
                            }
                        }
                        catch (Exception ex)
                        {
                            AppendSessionCheck(provider, passed: false, ex.Message);
                            throw;
                        }
                    }
                    catch (Exception ex) when (
                        _options.UseGpuAcceleration &&
                        provider != InferenceProvider.Cpu &&
                        IsRecoverableNativeFailure(ex))
                    {
                        _logger.LogWarning(ex, "Native session initialization failed. Falling back to CPU.");
                        _pendingChecks.Add(ProviderDiagnosticCheck.Pass(
                            "session.native-recoverable",
                            "IsRecoverableNativeFailure detected, attempting CPU fallback",
                            "Fallback",
                            NextOrder()));

                        sessionOptions?.Dispose();
                        fallback = InferenceFallbackReason.NativeLibraryLoadFailed;
                        provider = InferenceProvider.Cpu;
                        deviceId = 0;

                        sessionOptions = CreateCpuOnlySessionOptions();
                        AppendFallbackChecks(sessionOptions, modelPath);
                    }
                    finally
                    {
                        sessionOptions?.Dispose();
                    }

                    _runOptions = new RunOptions();

                    InputMetadata = _session!.InputMetadata;
                    OutputMetadata = _session!.OutputMetadata;
                    PrimaryInputName = _session!.InputMetadata.Keys.FirstOrDefault();

                    IsRocmSessionActive = provider == InferenceProvider.MIGraphX;

                    var info = new InferenceEngineInfo
                    {
                        Provider = provider,
                        Platform = HostPlatformDetector.Current(),
                        DeviceId = deviceId,
                        GpuAccelerationRequested = _options.UseGpuAcceleration,
                        IsFallback = fallback != InferenceFallbackReason.None,
                        FallbackReason = fallback,
                        ProviderOptions = ExtractProviderOptions(provider),
                        SessionCreatedAt = DateTimeOffset.UtcNow,
                        Checks = _pendingChecks.ToArray(),
                    };

#if DEBUG
                    for (int i = 0; i < info.Checks.Count; i++)
                        Debug.Assert(info.Checks[i].Order == i, $"Check order mismatch at index {i}: expected {i}, got {info.Checks[i].Order}");
#endif

                    EngineInfo = info;

                    _logger.LogInformation(
                        "Model loaded with {Provider} provider (fallback={IsFallback}, reason={FallbackReason}). Primary Input: {PrimaryInputName}",
                        provider, info.IsFallback, fallback, PrimaryInputName);
                    _isLoaded = true;

                    if (_options.WarmupOnLoad)
                    {
                        PerformWarmup();
                    }

                    FireOnSessionInitialized(info);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to initialize Inference Engine.");
                    throw;
                }
            });
        }

        public virtual async Task<TOutput> PredictAsync(TInput input)
        {
            if (!_isLoaded) await LoadAsync();

            return await Task.Run(() =>
            {
                var (inputTensors, context) = PreProcessWithContext(input);

                try
                {
                    using var outputTensors = _session!.Run(inputTensors, _session.OutputNames, _runOptions);
                    return PostProcessWithContext(outputTensors, context);
                }
                finally
                {
                    foreach(var t in inputTensors)
                    {
                        if (t is IDisposable disposable)
                        {
                            disposable.Dispose();
                        }
                    }
                }
            });
        }

        // --- Internal/Protected Helpers ---

        private int NextOrder() => _nextOrder++;

        private SessionOptions ConfiguredSessionOptions(
            out InferenceProvider provider,
            out InferenceFallbackReason fallback,
            out int deviceId)
        {
            if (!_options.UseGpuAcceleration)
            {
                _pendingChecks.Add(ProviderDiagnosticCheck.Fail(
                    "gpu.user-enabled",
                    "UseGpuAcceleration is enabled",
                    "Config",
                    NextOrder(),
                    "UseGpuAcceleration=false"));
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("rid.platform", "Host platform detection", "Platform", NextOrder(), "UseGpuAcceleration=false"));
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("provider.append", "Execution provider append", "Provider", NextOrder(), "UseGpuAcceleration=false"));
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("session.construct", "Session construction", "Session", NextOrder(), "UseGpuAcceleration=false"));

                provider = InferenceProvider.Cpu;
                fallback = InferenceFallbackReason.UserDisabledGpu;
                deviceId = 0;
                var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                options.AppendExecutionProvider_CPU();
                return options;
            }

            _pendingChecks.Add(ProviderDiagnosticCheck.Pass("gpu.user-enabled", "UseGpuAcceleration is enabled", "Config", NextOrder()));

            var platform = HostPlatformDetector.Current();
            var (sessionOptions, prov, fb, dev) = platform switch
            {
                HostPlatform.LinuxX64 => TryRocmBranch(),
                HostPlatform.WinX64 or HostPlatform.WinArm64 => TryDirectMlBranch(),
                HostPlatform.OSX64 or HostPlatform.OSXArm64 => TryCoreMlBranch(),
                _ => TryCpuBranch(InferenceFallbackReason.UnsupportedPlatform),
            };

            provider = prov;
            fallback = fb;
            deviceId = dev;
            return sessionOptions;
        }

        private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryRocmBranch()
        {
            _pendingChecks.Add(ProviderDiagnosticCheck.Pass("rid.platform", "Host is Linux x64", "Platform", NextOrder()));
            var deviceId = _options.DeviceId;

            if (!File.Exists("/dev/kfd"))
            {
                _pendingChecks.Add(ProviderDiagnosticCheck.Fail("kfd.present", "/dev/kfd exists", "Provider", NextOrder(), "/dev/kfd not found"));
                SkipRocmProviderChecks("precondition kfd.present failed");
                return BuildCpuFallback(InferenceFallbackReason.KfdDeviceMissing, deviceId);
            }
            _pendingChecks.Add(ProviderDiagnosticCheck.Pass("kfd.present", "/dev/kfd exists", "Provider", NextOrder()));

            IntPtr hsaHandle = IntPtr.Zero;
            bool hsaLoaded = false;
            string? hsaError = null;
            try
            {
                hsaLoaded = NativeLibrary.TryLoad("libhsa-runtime64.so", typeof(InferenceSession).Assembly,
                    DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32, out hsaHandle);
            }
            catch (Exception ex)
            {
                hsaError = ex.Message;
            }

            if (!hsaLoaded)
            {
                _pendingChecks.Add(ProviderDiagnosticCheck.Fail("hsa.dlopen", "libhsa-runtime64.so is loadable", "Provider", NextOrder(), hsaError ?? "NativeLibrary.TryLoad returned false"));
                SkipRocmProviderChecks("precondition hsa.dlopen failed", skipHsa: false);
                return BuildCpuFallback(InferenceFallbackReason.HsaRuntimeUnloadable, deviceId);
            }
            if (hsaHandle != IntPtr.Zero) NativeLibrary.Free(hsaHandle);
            _pendingChecks.Add(ProviderDiagnosticCheck.Pass("hsa.dlopen", "libhsa-runtime64.so is loadable", "Provider", NextOrder()));

            IntPtr hipHandle = IntPtr.Zero;
            bool hipLoaded = false;
            string? hipError = null;
            try
            {
                hipLoaded = NativeLibrary.TryLoad("libamdhip64.so.7", typeof(InferenceSession).Assembly,
                    DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32, out hipHandle);
            }
            catch (Exception ex)
            {
                hipError = ex.Message;
            }

            if (!hipLoaded)
            {
                _pendingChecks.Add(ProviderDiagnosticCheck.Fail("hip.dlopen", "libamdhip64.so.7 is loadable", "Provider", NextOrder(),
                    hipError ?? "libamdhip64.so.7 not found. HIP runtime is required for MIGraphX GPU acceleration. " +
                    "Install: sudo apt install hip-runtime-amd (from AMD ROCm repository). " +
                    "See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"));
                SkipRocmProviderChecks("precondition hip.dlopen failed", skipHsa: false, skipHip: false);
                return BuildCpuFallback(InferenceFallbackReason.HipRuntimeMissing, deviceId);
            }
            if (hipHandle != IntPtr.Zero) NativeLibrary.Free(hipHandle);
            _pendingChecks.Add(ProviderDiagnosticCheck.Pass("hip.dlopen", "libamdhip64.so.7 is loadable", "Provider", NextOrder()));

            IntPtr mgxHandle = IntPtr.Zero;
            bool mgxLoaded = false;
            string? mgxError = null;
            try
            {
                mgxLoaded = NativeLibrary.TryLoad("libmigraphx_c.so.3", typeof(InferenceSession).Assembly,
                    DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32, out mgxHandle);
            }
            catch (Exception ex)
            {
                mgxError = ex.Message;
            }

            if (!mgxLoaded)
            {
                _pendingChecks.Add(ProviderDiagnosticCheck.Fail("migraphx.dlopen", "libmigraphx_c.so.3 is loadable", "Provider", NextOrder(),
                    mgxError ?? "libmigraphx_c.so.3 not found. MIGraphX runtime is required for GPU-accelerated inference. " +
                    "Install: sudo apt install migraphx (from AMD ROCm repository). " +
                    "See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"));
                SkipRocmProviderChecks("precondition migraphx.dlopen failed", skipHsa: false, skipHip: false, skipMigraphxDlopen: false);
                return BuildCpuFallback(InferenceFallbackReason.MIGraphXRuntimeMissing, deviceId);
            }
            if (mgxHandle != IntPtr.Zero) NativeLibrary.Free(mgxHandle);
            _pendingChecks.Add(ProviderDiagnosticCheck.Pass("migraphx.dlopen", "libmigraphx_c.so.3 is loadable", "Provider", NextOrder()));

            var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            try
            {
                options.AppendExecutionProvider_MIGraphX(deviceId);
                _pendingChecks.Add(ProviderDiagnosticCheck.Pass("migraphx.append", "AppendExecutionProvider_MIGraphX succeeded", "Provider", NextOrder()));
                return (options, InferenceProvider.MIGraphX, InferenceFallbackReason.None, deviceId);
            }
            catch (Exception ex)
            {
                _pendingChecks.Add(ProviderDiagnosticCheck.Fail("migraphx.append", "AppendExecutionProvider_MIGraphX succeeded", "Provider", NextOrder(), ex.Message));
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("migraphx.session", "MIGraphX InferenceSession construction", "Session", NextOrder(), "precondition migraphx.append failed"));
                options.Dispose();
                return BuildCpuFallback(InferenceFallbackReason.ProviderAppendFailed, deviceId);
            }
        }

        private void SkipRocmProviderChecks(string reason,
            bool skipHsa = true, bool skipHip = true, bool skipMigraphxDlopen = true,
            bool skipMigraphxAppend = true, bool skipMigraphxSession = true)
        {
            if (skipHsa)
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("hsa.dlopen", "libhsa-runtime64.so is loadable", "Provider", NextOrder(), reason));
            if (skipHip)
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("hip.dlopen", "libamdhip64.so.7 is loadable", "Provider", NextOrder(), reason));
            if (skipMigraphxDlopen)
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("migraphx.dlopen", "libmigraphx_c.so.3 is loadable", "Provider", NextOrder(), reason));
            if (skipMigraphxAppend)
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("migraphx.append", "AppendExecutionProvider_MIGraphX succeeded", "Provider", NextOrder(), reason));
            if (skipMigraphxSession)
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("migraphx.session", "MIGraphX InferenceSession construction", "Session", NextOrder(), reason));
        }

        private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryDirectMlBranch()
        {
            _pendingChecks.Add(ProviderDiagnosticCheck.Pass("rid.platform", "Host is Windows", "Platform", NextOrder()));
            var deviceId = _options.DeviceId;
            var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };

            try
            {
                options.AppendExecutionProvider_DML(deviceId);
                _pendingChecks.Add(ProviderDiagnosticCheck.Pass("dml.append", "AppendExecutionProvider_DML succeeded", "Provider", NextOrder()));
                return (options, InferenceProvider.DirectML, InferenceFallbackReason.None, deviceId);
            }
            catch (Exception ex)
            {
                _pendingChecks.Add(ProviderDiagnosticCheck.Fail("dml.append", "AppendExecutionProvider_DML succeeded", "Provider", NextOrder(), ex.Message));
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("dml.session", "DirectML InferenceSession construction", "Session", NextOrder(), "precondition dml.append failed"));
                options.Dispose();
                return BuildCpuFallback(InferenceFallbackReason.ProviderAppendFailed, deviceId);
            }
        }

        private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryCoreMlBranch()
        {
            _pendingChecks.Add(ProviderDiagnosticCheck.Pass("rid.platform", "Host is macOS", "Platform", NextOrder()));
            var deviceId = _options.DeviceId;
            var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };

            try
            {
                options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                _pendingChecks.Add(ProviderDiagnosticCheck.Pass("coreml.append", "AppendExecutionProvider_CoreML succeeded", "Provider", NextOrder()));
                return (options, InferenceProvider.CoreML, InferenceFallbackReason.None, deviceId);
            }
            catch (Exception ex)
            {
                _pendingChecks.Add(ProviderDiagnosticCheck.Fail("coreml.append", "AppendExecutionProvider_CoreML succeeded", "Provider", NextOrder(), ex.Message));
                _pendingChecks.Add(ProviderDiagnosticCheck.Skip("coreml.session", "CoreML InferenceSession construction", "Session", NextOrder(), "precondition coreml.append failed"));
                options.Dispose();
                return BuildCpuFallback(InferenceFallbackReason.ProviderAppendFailed, deviceId);
            }
        }

        private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryCpuBranch(InferenceFallbackReason reason)
        {
            _pendingChecks.Add(ProviderDiagnosticCheck.Fail("rid.platform", "Host platform detection", "Platform", NextOrder(), $"Unsupported platform: {HostPlatformDetector.Current()}"));
            _pendingChecks.Add(ProviderDiagnosticCheck.Skip("provider.append", "Execution provider append", "Provider", NextOrder(), "unsupported platform"));
            _pendingChecks.Add(ProviderDiagnosticCheck.Skip("session.construct", "Session construction", "Session", NextOrder(), "unsupported platform"));
            var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            options.AppendExecutionProvider_CPU();
            return (options, InferenceProvider.Cpu, reason, 0);
        }

        private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) BuildCpuFallback(InferenceFallbackReason reason, int deviceId)
        {
            var options = CreateCpuOnlySessionOptions();
            return (options, InferenceProvider.Cpu, reason, deviceId);
        }

        private void AppendSessionCheck(InferenceProvider provider, bool passed, string? detail = null)
        {
            var name = provider switch
            {
                InferenceProvider.MIGraphX => "migraphx.session",
                InferenceProvider.DirectML => "dml.session",
                InferenceProvider.CoreML => "coreml.session",
                _ => "cpu.session",
            };
            var desc = $"{provider} InferenceSession construction";
            _pendingChecks.Add(passed
                ? ProviderDiagnosticCheck.Pass(name, desc, "Session", NextOrder(), detail)
                : ProviderDiagnosticCheck.Fail(name, desc, "Session", NextOrder(), detail ?? "unknown failure"));
        }

        private void AppendFallbackChecks(SessionOptions cpuOptions, string modelPath)
        {
            _pendingChecks.Add(ProviderDiagnosticCheck.Pass("cpu.fallback-options", "CreateCpuOnlySessionOptions() succeeded", "Fallback", NextOrder()));
            try
            {
                _session = new InferenceSession(modelPath, cpuOptions);
                _pendingChecks.Add(ProviderDiagnosticCheck.Pass("cpu.fallback-session", "CPU InferenceSession constructed", "Fallback", NextOrder()));
            }
            catch (Exception ex)
            {
                _pendingChecks.Add(ProviderDiagnosticCheck.Fail("cpu.fallback-session", "CPU InferenceSession construction", "Fallback", NextOrder(), ex.Message));
                throw;
            }
        }

        private void FireOnSessionInitialized(InferenceEngineInfo info)
        {
            if (_options.OnSessionInitialized is null) return;
            try
            {
                _options.OnSessionInitialized(info);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "OnSessionInitialized callback threw; engine is still usable.");
            }
        }

        /// <summary>
        /// Apply engine state that was loaded by an out-of-process worker (no in-process
        /// <see cref="InferenceSession"/>). Used by <c>OutOfProcessBaseInferenceEngine</c> to mirror
        /// the worker's <see cref="EngineInfo"/>/metadata so the diagnostics surface (provider,
        /// fallback reason, ordered checks) behaves identically to the in-process engine. Does not
        /// create a session; the proxy's <see cref="PredictAsync"/> marshals to the worker instead.
        /// </summary>
        protected void ApplyLoadedState(
            InferenceEngineInfo engineInfo,
            IReadOnlyDictionary<string, NodeMetadata>? inputMetadata,
            IReadOnlyDictionary<string, NodeMetadata>? outputMetadata,
            string? primaryInputName)
        {
            EngineInfo = engineInfo;
            InputMetadata = inputMetadata;
            OutputMetadata = outputMetadata;
            PrimaryInputName = primaryInputName;
            _isLoaded = true;
            FireOnSessionInitialized(engineInfo);
        }

        private static IReadOnlyDictionary<string, string> ExtractProviderOptions(InferenceProvider provider)
        {
            _ = provider;
            return ImmutableDictionary<string, string>.Empty;
        }

        private static SessionOptions CreateCpuOnlySessionOptions()
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            };
            options.AppendExecutionProvider_CPU();
            return options;
        }

        /// <summary>
        /// Determines whether an exception thrown during ONNX session initialization represents a
        /// recoverable native library loading failure.
        /// </summary>
        private static bool IsRecoverableNativeFailure(Exception ex)
        {
            if (ex is OnnxRuntimeException or DllNotFoundException or BadImageFormatException) return true;
            if (ex is TypeInitializationException tiex)
                return tiex.InnerException is OnnxRuntimeException or DllNotFoundException or BadImageFormatException;
            return false;
        }

        /// <summary>
        /// Generates dummy input for model warmup.
        /// Default implementation assumes float tensors. Override this for int64/string inputs.
        /// </summary>
        protected virtual List<NamedOnnxValue> GetWarmupInput()
        {
            var inputs = new List<NamedOnnxValue>();
            if (InputMetadata == null) return inputs;

            foreach (var inputMeta in InputMetadata)
            {
                var shape = inputMeta.Value.Dimensions.Select(d => d == -1 ? 1 : d).ToArray();
                var tensorData = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(shape);
                inputs.Add(NamedOnnxValue.CreateFromTensor(inputMeta.Key, tensorData));
            }
            return inputs;
        }

        private void PerformWarmup()
        {
            try
            {
                _logger.LogInformation("Performing warmup inference...");
                var inputs = GetWarmupInput();

                if (inputs.Count > 0)
                {
                    using var results = _session!.Run(inputs);
                    _logger.LogInformation("Warmup complete.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning($"Warmup failed (non-critical). Error: {ex.Message}");
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                _session?.Dispose();
                _runOptions?.Dispose();
                _session = null;
                _isLoaded = false;
            }
        }
    }
}
