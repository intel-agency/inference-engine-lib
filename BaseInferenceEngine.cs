using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

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

        // Exposed Metadata
        public IReadOnlyDictionary<string, NodeMetadata>? InputMetadata { get; private set; }
        public IReadOnlyDictionary<string, NodeMetadata>? OutputMetadata { get; private set; }
        public string? PrimaryInputName { get; private set; }

        protected BaseInferenceEngine(InferenceEngineOptions options, ILogger? logger = null)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger.Instance;
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
        /// Initializes the inference session with the best available hardware provider.
        /// </summary>
        public virtual async Task LoadAsync()
        {
            if (_isLoaded) return;

            if (!File.Exists(_options.ModelPath))
            {
                throw new FileNotFoundException($"Model not found at: {_options.ModelPath}");
            }

            await Task.Run(() =>
            {
                try
                {
                    _logger.LogInformation("Initializing Inference Engine...");
                    var sessionOptions = ConfiguredSessionOptions();

                    // Create the session
                    _session = new InferenceSession(_options.ModelPath, sessionOptions);
                    _runOptions = new RunOptions();

                    // Populate Metadata properties
                    InputMetadata = _session.InputMetadata;
                    OutputMetadata = _session.OutputMetadata;
                    PrimaryInputName = InputMetadata.Keys.FirstOrDefault();

                    _logger.LogInformation($"Model loaded. Provider: {_session.GetExecutionProviders().FirstOrDefault()}. Primary Input: {PrimaryInputName}");
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
                    using var outputTensors = _session!.Run(inputTensors, _runOptions);

                    // 3. Post-process (with context)
                    return PostProcessWithContext(outputTensors, context);
                }
                finally
                {
                    // Dispose input tensors if they wrap native memory
                    foreach (var t in inputTensors)
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

        private SessionOptions ConfiguredSessionOptions()
        {
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
            // 3. Linux/Other Strategy (CPU Fallback)
            else
            {
                _logger.LogInformation("Platform detected: Linux/Other. Using default CPU provider.");
                options.AppendExecutionProvider_CPU();
            }

            return options;
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
