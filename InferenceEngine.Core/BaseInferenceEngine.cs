using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
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

        // Tracks initialization state to prevent usage before loading
        private bool _isLoaded = false;

        // Model Provider (created lazily from model spec)
        private IModelProvider? _modelProvider;

        // Exposed Metadata
        public IReadOnlyDictionary<string, NodeMetadata>? InputMetadata { get; private set; }
        public IReadOnlyDictionary<string, NodeMetadata>? OutputMetadata { get; private set; }
        public string? PrimaryInputName { get; private set; }

        /// <summary>
        /// Returns <see langword="true"/> when the ONNX session was successfully initialized
        /// with the ROCm execution provider. <see langword="false"/> on non-Linux-x64 platforms,
        /// when ROCm initialization failed, or when CPU fallback was used.
        /// </summary>
        public bool IsRocmSessionActive { get; private set; }

        // --- Model Specification (to be implemented by derived classes) ---

        /// <summary>
        /// The name of the model file (e.g., "yolo11n.onnx").
        /// Derived classes must implement this to specify which model they require.
        /// </summary>
        protected abstract string ModelName { get; }

        /// <summary>
        /// URLs where the model can be downloaded from, in priority order.
        /// Derived classes can override to provide custom model sources (e.g., on-prem servers).
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
        /// (like image scaling factors) without overriding the main pipeline.
        /// </summary>
        /// <param name="input">Raw input data</param>
        /// <returns>A tuple of NamedOnnxValues (inputs) and a Context object (state)</returns>
        protected abstract (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(TInput input);

        /// <summary>
        /// Takes raw ONNX output and the context object from PreProcess, returning the final result.
        /// </summary>
        /// <param name="output">Disposable collection of output tensors</param>
        /// <param name="context">The state object returned by PreProcessWithContext</param>
        protected abstract TOutput PostProcessWithContext(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context);

        // --- Public API ---

        /// <summary>
        /// Gets or creates the model provider.
        /// Uses options.ModelProvider if set, otherwise creates from ModelName/ModelDownloadUrls/ModelRegistry.
        /// </summary>
        private IModelProvider GetModelProvider()
        {
            // Caller can always override with explicit provider
            if (_options.ModelProvider is not null)
            {
                return _options.ModelProvider;
            }

            // Lazily create provider from model spec
            if (_modelProvider is null)
            {
                var urls = ModelDownloadUrls;
                
                // If no URLs specified, try to look up in registry
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

            // Acquire model through provider (may download if necessary)
            var modelPath = await GetModelProvider().GetModelPathAsync(CancellationToken.None);

            await Task.Run(() =>
            {
                try
                {
                    // Combined try: ConfiguredSessionOptions and new InferenceSession share one try and one catch.
                    // A TypeInitializationException thrown while loading the base libonnxruntime (for example
                    // a missing libonnxruntime.so) is unwrapped via IsRecoverableNativeFailure so the CPU
                    // fallback path is reached instead of bubbling out as an opaque load failure.
                    try
                    {
                        using var sessionOptions = ConfiguredSessionOptions(out var rocmProviderAppended);
                        // Create the session
                        _session = new InferenceSession(modelPath, sessionOptions);
                        IsRocmSessionActive = rocmProviderAppended;
                    }
                    catch (Exception ex) when (
                        _options.UseGpuAcceleration &&
                        IsRecoverableNativeFailure(ex))
                    {
                        _logger.LogWarning(ex, "Native session initialization failed. Falling back to CPU.");
                        using var cpuSessionOptions = CreateCpuOnlySessionOptions();
                        _session = new InferenceSession(modelPath, cpuSessionOptions);
                    }

                    _runOptions = new RunOptions();

                    // Populate Metadata properties
                    InputMetadata = _session.InputMetadata;
                    OutputMetadata = _session.OutputMetadata;
                    PrimaryInputName = InputMetadata.Keys.FirstOrDefault();

                    _logger.LogInformation($"Model loaded. Primary Input: {PrimaryInputName}");
                    _isLoaded = true;

                    // Handle First Run Responsibility: Warmup
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

        public virtual async Task<TOutput> PredictAsync(TInput input)
        {
            if (!_isLoaded) await LoadAsync();

            return await Task.Run(() =>
            {
                // 1. Pre-process (with context)
                var (inputTensors, context) = PreProcessWithContext(input);

                try
                {
                    // 2. Inference
                    // Using null-forgiving operator (!) as LoadAsync ensures _session is not null here
                    using var outputTensors = _session!.Run(inputTensors, _session.OutputNames, _runOptions);

                    // 3. Post-process (with context)
                    return PostProcessWithContext(outputTensors, context);
                }
                finally
                {
                    // Dispose input tensors if they wrap native memory
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

        private SessionOptions ConfiguredSessionOptions(out bool rocmProviderAppended)
        {
            rocmProviderAppended = false;
            var options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            if (!_options.UseGpuAcceleration)
            {
                _logger.LogInformation("GPU Acceleration disabled by config. Using CPU.");
                options.AppendExecutionProvider_CPU();
                return options;
            }

            // 1. Windows Strategy (DirectML)
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                try
                {
                    _logger.LogInformation("Platform detected: Windows. Attempting DirectML...");
                    options.AppendExecutionProvider_DML(_options.DeviceId);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning($"DirectML failed to load. Falling back to CPU. Error: {ex.Message}");
                    options.AppendExecutionProvider_CPU();
                }
            }
            // 2. macOS Strategy (CoreML)
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                try
                {
                    _logger.LogInformation("Platform detected: macOS. Attempting CoreML...");
                    options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning($"CoreML failed to load. Falling back to CPU. Error: {ex.Message}");
                    options.AppendExecutionProvider_CPU();
                }
            }
            // 3. Linux Strategy (ROCm on x64, CPU on ARM64)
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
                && RuntimeInformation.ProcessArchitecture == Architecture.X64)
            {
                if (!IsRocmRuntimeUsable())
                {
                    _logger.LogInformation("Platform detected: Linux x64. ROCm runtime not available on this host — using CPU provider.");
                    options.AppendExecutionProvider_CPU();
                    return options;
                }

                _logger.LogInformation("Platform detected: Linux x64. ROCm runtime detected. Attempting ROCm provider...");

                try
                {
                    using var rocmOptions = new OrtROCMProviderOptions();
                    rocmOptions.UpdateOptions(new Dictionary<string, string>
                    {
                        ["device_id"] = _options.DeviceId.ToString(),
                        ["miopen_conv_exhaustive_search"] = "1",
                        ["arena_extend_strategy"] = "kSameAsRequested",
                    });

                    options.AppendExecutionProvider_ROCm(rocmOptions);
                    rocmProviderAppended = true;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "ROCm provider failed to load. Falling back to CPU.");
                    options.AppendExecutionProvider_CPU();
                }
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
                && RuntimeInformation.ProcessArchitecture == Architecture.Arm64)
            {
                _logger.LogInformation("Platform detected: Linux ARM64. Using CPU provider.");
                options.AppendExecutionProvider_CPU();
            }
            // 4. Other Platform Strategy (CPU Fallback)
            else
            {
                _logger.LogInformation("Platform detected: Other. Using default CPU provider.");
                options.AppendExecutionProvider_CPU();
            }

            return options;
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
        /// recoverable native library loading failure (for example, a missing or unloadable
        /// <c>libonnxruntime.so</c>). When GPU acceleration was requested, such failures should
        /// trigger a CPU fallback; otherwise the exception should propagate so the outer
        /// <c>LogError</c> handler can report it.
        /// </summary>
        /// <param name="ex">The exception thrown while building or opening the ONNX session.</param>
        /// <returns>
        /// <see langword="true"/> if <paramref name="ex"/> is or wraps an
        /// <see cref="OnnxRuntimeException"/> or <see cref="DllNotFoundException"/>;
        /// otherwise, <see langword="false"/>.
        /// </returns>
        private static bool IsRecoverableNativeFailure(Exception ex)
        {
            if (ex is OnnxRuntimeException || ex is DllNotFoundException) return true;
            if (ex is TypeInitializationException tiex)
                return tiex.InnerException is OnnxRuntimeException or DllNotFoundException;
            return false;
        }

        /// <summary>
        /// Determines whether the ROCm execution provider is usable on the current host.
        /// On Linux, checks that the kernel fusion driver (<c>/dev/kfd</c>) is exposed AND that
        /// the ROCm HSA runtime library (<c>libhsa-runtime64.so</c>) can be resolved via dlopen.
        /// On other platforms, returns <see langword="false"/>.
        /// </summary>
        /// <returns>
        /// <see langword="true"/> when both the ROCm kernel driver and HSA runtime are reachable;
        /// <see langword="false"/> otherwise. The reason for a <see langword="false"/> result is
        /// logged at <c>Debug</c> level.
        /// </returns>
        private bool IsRocmRuntimeUsable()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return false;
            }

            if (!File.Exists("/dev/kfd"))
            {
                _logger.LogDebug("ROCm kernel fusion driver not present (/dev/kfd missing). ROCm provider unavailable.");
                return false;
            }

            try
            {
                var handle = NativeLibrary.Load(
                    "libhsa-runtime64.so",
                    typeof(InferenceSession).Assembly,
                    DllImportSearchPath.UseDllDirectoryForDependencies | DllImportSearchPath.System32);
                if (handle != IntPtr.Zero)
                {
                    NativeLibrary.Free(handle);
                }
                return true;
            }
            catch (DllNotFoundException ex)
            {
                _logger.LogDebug(ex, "ROCm HSA runtime library (libhsa-runtime64.so) not resolvable. ROCm provider unavailable.");
                return false;
            }
            catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or InvalidOperationException)
            {
                // Belt-and-suspenders: some platforms wrap DllNotFoundException in TypeInitializationException
                // or surface it as InvalidOperationException from NativeLibrary.Load.
                _logger.LogDebug(ex, "ROCm HSA runtime probe failed. ROCm provider unavailable.");
                return false;
            }
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
                // Handle dynamic dimensions (-1) by defaulting to 1
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
