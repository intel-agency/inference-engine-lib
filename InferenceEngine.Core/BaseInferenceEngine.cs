using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
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
        private readonly SemaphoreSlim _loadLock = new(1, 1);
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

        /// <summary>
        /// Lifecycle + runtime snapshot (<see cref="EngineLoadState"/> + partial-or-full
        /// <see cref="InferenceEngineInfo"/> + load error). Transitions fire
        /// <see cref="InferenceEngineOptions.OnStatusChanged"/> when set. Out-of-process proxies
        /// mirror the worker's terminal states onto the same property.
        /// </summary>
        public InferenceEngineStatus Status { get; private set; } = InferenceEngineStatus.NotLoaded;

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
            // Serialize concurrent loads (H5): two concurrent LoadAsync calls must not both pass the
            // _isLoaded guard and double-construct the session / corrupt the check list.
            await _loadLock.WaitAsync().ConfigureAwait(false);
            try
            {
                if (_isLoaded) return;

                SetStatus(new InferenceEngineStatus { State = EngineLoadState.Loading, Detail = "Resolving model…" });
                try
                {

            _logger.LogInformation("Initializing Inference Engine (engine={EngineType}, gpu={UseGpu}, deviceId={DeviceId})...",
                GetType().Name, _options.UseGpuAcceleration, _options.DeviceId);

            var modelPath = await GetModelProvider().GetModelPathAsync(CancellationToken.None);
            _logger.LogInformation("Model resolved to '{ModelPath}' (exists={Exists})", modelPath, File.Exists(modelPath));

            SetStatus(new InferenceEngineStatus { State = EngineLoadState.Loading, Detail = "Selecting execution provider…" });

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

                        // Session construction is where a hardware EP compiles the model's
                        // kernels — including the multi-minute MIGraphX first-run compile.
                        SetStatus(new InferenceEngineStatus { State = EngineLoadState.Loading, Detail = "Constructing inference session…" });

                        try
                        {
                            _session = CreateInferenceSession(modelPath, sessionOptions);
                            if (fallback != InferenceFallbackReason.None)
                            {
                                AddCheck(ProviderDiagnosticCheck.Pass("cpu.fallback-options", "CreateCpuOnlySessionOptions() succeeded", "Fallback", NextOrder()));
                                AddCheck(ProviderDiagnosticCheck.Pass("cpu.fallback-session", "CPU InferenceSession constructed", "Fallback", NextOrder()));
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
                        AddCheck(ProviderDiagnosticCheck.Pass(
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
                        ModelName = ModelName,
                        ModelPath = modelPath,
                        ModelSource = _options.ModelProvider is not null
                            ? Diagnostics.ModelSource.ExplicitPath
                            : Diagnostics.ModelSource.RegistryManaged,
                        IsFallback = fallback != InferenceFallbackReason.None,
                        FallbackReason = fallback,
                        ProviderOptions = ExtractProviderOptions(provider, deviceId),
                        SessionCreatedAt = DateTimeOffset.UtcNow,
                        Checks = _pendingChecks.ToArray(),
                    };

#if DEBUG
                    for (int i = 0; i < info.Checks.Count; i++)
                        Debug.Assert(info.Checks[i].Order == i, $"Check order mismatch at index {i}: expected {i}, got {info.Checks[i].Order}");
#endif

                    EngineInfo = info;
                    SetStatus(new InferenceEngineStatus { State = EngineLoadState.Loaded, Info = info });

                    _logger.LogInformation(
                        "Model loaded with {Provider} provider (fallback={IsFallback}, reason={FallbackReason}). Primary Input: {PrimaryInputName}",
                        provider, info.IsFallback, fallback, PrimaryInputName);
                    _isLoaded = true;

                    // M4: fire the session-initialized callback BEFORE warmup (it documents that it
                    // "fires once after the session is created"), so a long first-compile warmup does
                    // not delay it and a warmup problem cannot prevent it.
                    FireOnSessionInitialized(info);

                    if (_options.WarmupOnLoad)
                    {
                        PerformWarmup();
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to initialize Inference Engine.");
                    throw;
                }
            });
                }
                catch (Exception ex)
                {
                    // Surface the failure through the status object model: the partial
                    // provider-selection trail (checks accumulated before the failure) explains
                    // WHY, and the error string says WHAT threw.
                    SetStatus(new InferenceEngineStatus
                    {
                        State = EngineLoadState.Failed,
                        Info = new InferenceEngineInfo
                        {
                            Provider = InferenceProvider.Unknown,
                            Checks = _pendingChecks.ToArray(),
                        },
                        LoadError = ex.Message,
                    });
                    throw;
                }
            }
            finally
            {
                _loadLock.Release();
            }
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

        // Test seams (T1/T2): virtualize the platform/probe/append/session-construction steps so the
        // most complex fallback branches can be exercised hermetically (on any CI host, with no GPU).
        // Defaults reproduce the original behavior exactly.

        /// <summary>Platform used to select the execution-provider branch. Override to force a branch in tests.</summary>
        protected virtual HostPlatform EffectivePlatform => HostPlatformDetector.Current();

        /// <summary>True when the ROCm KFD device node is present. Override in tests.</summary>
        protected virtual bool IsKfdPresent() => File.Exists("/dev/kfd");

        /// <summary>Attempts to dlopen a native library; returns success and an error message on failure.</summary>
        protected virtual bool TryLoadNativeLibrary(string name, out string? error)
        {
            try
            {
                bool loaded = NativeLibrary.TryLoad(name, typeof(InferenceSession).Assembly,
                    DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32, out var handle);
                if (loaded && handle != IntPtr.Zero) NativeLibrary.Free(handle);
                error = loaded ? null : "NativeLibrary.TryLoad returned false";
                return loaded;
            }
            catch (Exception ex)
            {
                error = ex.Message;
                return false;
            }
        }

        /// <summary>Appends the MIGraphX execution provider. Override to inject a throw in tests.</summary>
        protected virtual void AppendMIGraphXProvider(SessionOptions options, int deviceId)
            => options.AppendExecutionProvider_MIGraphX(deviceId);

        /// <summary>Appends the DirectML execution provider. Override to inject a throw in tests.</summary>
        protected virtual void AppendDirectMlProvider(SessionOptions options, int deviceId)
            => options.AppendExecutionProvider_DML(deviceId);

        /// <summary>Appends the CoreML execution provider. Override to inject a throw in tests.</summary>
        protected virtual void AppendCoreMlProvider(SessionOptions options, CoreMLFlags flags)
            => options.AppendExecutionProvider_CoreML(flags);

        /// <summary>Constructs the ONNX session. Override to inject a recoverable native failure in tests.</summary>
        protected virtual InferenceSession CreateInferenceSession(string modelPath, SessionOptions options)
            => new(modelPath, options);

        private int NextOrder() => _nextOrder++;

        /// <summary>
        /// Single choke point for appending to the provider-selection check trail: keeps the list
        /// behavior identical while mirroring every check to the logger, so the forensic trail
        /// (which probe ran, what it found) lands in the log file even before EngineInfo reaches a
        /// consumer. Failed checks escalate to Warning; passes/skips log at Debug.
        /// </summary>
        private void AddCheck(ProviderDiagnosticCheck check)
        {
            _pendingChecks.Add(check);
            _logger.Log(
                check.Result == Diagnostics.CheckResult.Failed ? LogLevel.Warning : LogLevel.Debug,
                "check [{Order}] {Result} {Name} ({Category}): {Detail}",
                check.Order, check.Result, check.Name, check.Category, check.Detail ?? check.Description);
        }

        private SessionOptions ConfiguredSessionOptions(
            out InferenceProvider provider,
            out InferenceFallbackReason fallback,
            out int deviceId)
        {
            if (!_options.UseGpuAcceleration)
            {
                AddCheck(ProviderDiagnosticCheck.Fail(
                    "gpu.user-enabled",
                    "UseGpuAcceleration is enabled",
                    "Config",
                    NextOrder(),
                    "UseGpuAcceleration=false"));
                AddCheck(ProviderDiagnosticCheck.Skip("rid.platform", "Host platform detection", "Platform", NextOrder(), "UseGpuAcceleration=false"));
                AddCheck(ProviderDiagnosticCheck.Skip("provider.append", "Execution provider append", "Provider", NextOrder(), "UseGpuAcceleration=false"));
                AddCheck(ProviderDiagnosticCheck.Skip("session.construct", "Session construction", "Session", NextOrder(), "UseGpuAcceleration=false"));

                provider = InferenceProvider.Cpu;
                fallback = InferenceFallbackReason.UserDisabledGpu;
                deviceId = 0;
                var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                options.AppendExecutionProvider_CPU();
                return options;
            }

            AddCheck(ProviderDiagnosticCheck.Pass("gpu.user-enabled", "UseGpuAcceleration is enabled", "Config", NextOrder()));

            var platform = EffectivePlatform;
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
            AddCheck(ProviderDiagnosticCheck.Pass("rid.platform", "Host is Linux x64", "Platform", NextOrder()));
            var deviceId = _options.DeviceId;

            if (!IsKfdPresent())
            {
                AddCheck(ProviderDiagnosticCheck.Fail("kfd.present", "/dev/kfd exists", "Provider", NextOrder(), "/dev/kfd not found"));
                SkipRocmProviderChecks("precondition kfd.present failed");
                return BuildCpuFallback(InferenceFallbackReason.KfdDeviceMissing, deviceId);
            }
            AddCheck(ProviderDiagnosticCheck.Pass("kfd.present", "/dev/kfd exists", "Provider", NextOrder()));

            if (!TryLoadNativeLibrary("libhsa-runtime64.so", out var hsaError))
            {
                AddCheck(ProviderDiagnosticCheck.Fail("hsa.dlopen", "libhsa-runtime64.so is loadable", "Provider", NextOrder(), hsaError ?? "NativeLibrary.TryLoad returned false"));
                SkipRocmProviderChecks("precondition hsa.dlopen failed", skipHsa: false);
                return BuildCpuFallback(InferenceFallbackReason.HsaRuntimeUnloadable, deviceId);
            }
            AddCheck(ProviderDiagnosticCheck.Pass("hsa.dlopen", "libhsa-runtime64.so is loadable", "Provider", NextOrder()));

            if (!TryLoadNativeLibrary("libamdhip64.so.7", out var hipError))
            {
                AddCheck(ProviderDiagnosticCheck.Fail("hip.dlopen", "libamdhip64.so.7 is loadable", "Provider", NextOrder(),
                    hipError ?? "libamdhip64.so.7 not found. HIP runtime is required for MIGraphX GPU acceleration. " +
                    "Install: sudo apt install hip-runtime-amd (from AMD ROCm repository). " +
                    "See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"));
                SkipRocmProviderChecks("precondition hip.dlopen failed", skipHsa: false, skipHip: false);
                return BuildCpuFallback(InferenceFallbackReason.HipRuntimeMissing, deviceId);
            }
            AddCheck(ProviderDiagnosticCheck.Pass("hip.dlopen", "libamdhip64.so.7 is loadable", "Provider", NextOrder()));

            if (!TryLoadNativeLibrary("libmigraphx_c.so.3", out var mgxError))
            {
                AddCheck(ProviderDiagnosticCheck.Fail("migraphx.dlopen", "libmigraphx_c.so.3 is loadable", "Provider", NextOrder(),
                    mgxError ?? "libmigraphx_c.so.3 not found. MIGraphX runtime is required for GPU-accelerated inference. " +
                    "Install: sudo apt install migraphx (from AMD ROCm repository). " +
                    "See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"));
                SkipRocmProviderChecks("precondition migraphx.dlopen failed", skipHsa: false, skipHip: false, skipMigraphxDlopen: false);
                return BuildCpuFallback(InferenceFallbackReason.MIGraphXRuntimeMissing, deviceId);
            }
            AddCheck(ProviderDiagnosticCheck.Pass("migraphx.dlopen", "libmigraphx_c.so.3 is loadable", "Provider", NextOrder()));

            var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            try
            {
                AppendMIGraphXProvider(options, deviceId);
                AddCheck(ProviderDiagnosticCheck.Pass("migraphx.append", "AppendExecutionProvider_MIGraphX", "Provider", NextOrder()));
                return (options, InferenceProvider.MIGraphX, InferenceFallbackReason.None, deviceId);
            }
            catch (Exception ex)
            {
                AddCheck(ProviderDiagnosticCheck.Fail("migraphx.append", "AppendExecutionProvider_MIGraphX", "Provider", NextOrder(), ex.Message));
                AddCheck(ProviderDiagnosticCheck.Skip("migraphx.session", "MIGraphX InferenceSession construction", "Session", NextOrder(), "precondition migraphx.append failed"));
                options.Dispose();
                return BuildCpuFallback(InferenceFallbackReason.ProviderAppendFailed, deviceId);
            }
        }

        private void SkipRocmProviderChecks(string reason,
            bool skipHsa = true, bool skipHip = true, bool skipMigraphxDlopen = true,
            bool skipMigraphxAppend = true, bool skipMigraphxSession = true)
        {
            if (skipHsa)
                AddCheck(ProviderDiagnosticCheck.Skip("hsa.dlopen", "libhsa-runtime64.so is loadable", "Provider", NextOrder(), reason));
            if (skipHip)
                AddCheck(ProviderDiagnosticCheck.Skip("hip.dlopen", "libamdhip64.so.7 is loadable", "Provider", NextOrder(), reason));
            if (skipMigraphxDlopen)
                AddCheck(ProviderDiagnosticCheck.Skip("migraphx.dlopen", "libmigraphx_c.so.3 is loadable", "Provider", NextOrder(), reason));
            if (skipMigraphxAppend)
                AddCheck(ProviderDiagnosticCheck.Skip("migraphx.append", "AppendExecutionProvider_MIGraphX succeeded", "Provider", NextOrder(), reason));
            if (skipMigraphxSession)
                AddCheck(ProviderDiagnosticCheck.Skip("migraphx.session", "MIGraphX InferenceSession construction", "Session", NextOrder(), reason));
        }

        private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryDirectMlBranch()
        {
            AddCheck(ProviderDiagnosticCheck.Pass("rid.platform", "Host is Windows", "Platform", NextOrder()));
            var deviceId = _options.DeviceId;
            var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };

            try
            {
                AppendDirectMlProvider(options, deviceId);
                AddCheck(ProviderDiagnosticCheck.Pass("dml.append", "AppendExecutionProvider_DML", "Provider", NextOrder()));
                return (options, InferenceProvider.DirectML, InferenceFallbackReason.None, deviceId);
            }
            catch (Exception ex)
            {
                AddCheck(ProviderDiagnosticCheck.Fail("dml.append", "AppendExecutionProvider_DML", "Provider", NextOrder(), ex.Message));
                AddCheck(ProviderDiagnosticCheck.Skip("dml.session", "DirectML InferenceSession construction", "Session", NextOrder(), "precondition dml.append failed"));
                options.Dispose();
                return BuildCpuFallback(InferenceFallbackReason.ProviderAppendFailed, deviceId);
            }
        }

        private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryCoreMlBranch()
        {
            AddCheck(ProviderDiagnosticCheck.Pass("rid.platform", "Host is macOS", "Platform", NextOrder()));
            var deviceId = _options.DeviceId;
            var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };

            try
            {
                AppendCoreMlProvider(options, CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                AddCheck(ProviderDiagnosticCheck.Pass("coreml.append", "AppendExecutionProvider_CoreML", "Provider", NextOrder()));
                return (options, InferenceProvider.CoreML, InferenceFallbackReason.None, deviceId);
            }
            catch (Exception ex)
            {
                AddCheck(ProviderDiagnosticCheck.Fail("coreml.append", "AppendExecutionProvider_CoreML", "Provider", NextOrder(), ex.Message));
                AddCheck(ProviderDiagnosticCheck.Skip("coreml.session", "CoreML InferenceSession construction", "Session", NextOrder(), "precondition coreml.append failed"));
                options.Dispose();
                return BuildCpuFallback(InferenceFallbackReason.ProviderAppendFailed, deviceId);
            }
        }

        private (SessionOptions Options, InferenceProvider Provider, InferenceFallbackReason Fallback, int DeviceId) TryCpuBranch(InferenceFallbackReason reason)
        {
            AddCheck(ProviderDiagnosticCheck.Fail("rid.platform", "Host platform detection", "Platform", NextOrder(), $"Unsupported platform: {HostPlatformDetector.Current()}"));
            AddCheck(ProviderDiagnosticCheck.Skip("provider.append", "Execution provider append", "Provider", NextOrder(), "unsupported platform"));
            AddCheck(ProviderDiagnosticCheck.Skip("session.construct", "Session construction", "Session", NextOrder(), "unsupported platform"));
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
            AddCheck(passed
                ? ProviderDiagnosticCheck.Pass(name, desc, "Session", NextOrder(), detail)
                : ProviderDiagnosticCheck.Fail(name, desc, "Session", NextOrder(), detail ?? "unknown failure"));
        }

        private void AppendFallbackChecks(SessionOptions cpuOptions, string modelPath)
        {
            AddCheck(ProviderDiagnosticCheck.Pass("cpu.fallback-options", "CreateCpuOnlySessionOptions() succeeded", "Fallback", NextOrder()));
            try
            {
                _session = CreateInferenceSession(modelPath, cpuOptions);
                AddCheck(ProviderDiagnosticCheck.Pass("cpu.fallback-session", "CPU InferenceSession constructed", "Fallback", NextOrder()));
            }
            catch (Exception ex)
            {
                AddCheck(ProviderDiagnosticCheck.Fail("cpu.fallback-session", "CPU InferenceSession construction", "Fallback", NextOrder(), ex.Message));
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
        /// Publishes a status transition: updates <see cref="Status"/> then fires
        /// <see cref="InferenceEngineOptions.OnStatusChanged"/> (callback exceptions are
        /// swallowed and logged — a broken consumer callback must not kill the engine).
        /// Private protected: out-of-process proxies drive their own transitions with the
        /// worker-reported terminal states.
        /// </summary>
        private protected void SetStatus(InferenceEngineStatus status)
        {
            Status = status;
            if (_options.OnStatusChanged is null) return;
            try
            {
                _options.OnStatusChanged(status);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "OnStatusChanged callback threw; engine is still usable.");
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

        /// <summary>
        /// Per-provider key/value options actually applied. At minimum surfaces <c>device_id</c> for
        /// hardware-accelerated providers (MIGraphX/DirectML/CoreML); empty for CPU/Unknown.
        /// </summary>
        private static IReadOnlyDictionary<string, string> ExtractProviderOptions(InferenceProvider provider, int deviceId)
        {
            if (provider == InferenceProvider.Cpu || provider == InferenceProvider.Unknown)
                return ImmutableDictionary<string, string>.Empty;

            // Build the single-entry dictionary directly on the immutable tree instead of allocating
            // a transient Dictionary<string,string> + ToImmutableDictionary() on every session init.
            return ImmutableDictionary<string, string>.Empty.Add("device_id", deviceId.ToString());
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
                // Silent reset (no OnStatusChanged fire): a disposed engine's prior state is not
                // a transition consumers need to react to.
                Status = InferenceEngineStatus.NotLoaded;
                _loadLock.Dispose();
            }
        }
    }
}
