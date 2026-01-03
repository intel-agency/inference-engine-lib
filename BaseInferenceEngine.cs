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
    /// Client apps should subclass this (e.g., FaceDetectionEngine, ChatBotEngine).
    /// </summary>
    public abstract class BaseInferenceEngine<TInput, TOutput> : IInferenceEngine<TInput, TOutput>
    {
        protected readonly InferenceEngineOptions _options;
        protected readonly ILogger _logger;
        protected InferenceSession _session;
        protected RunOptions _runOptions;

        // Tracks initialization state to prevent usage before loading
        private bool _isLoaded = false;

        protected BaseInferenceEngine(InferenceEngineOptions options, ILogger? logger = null)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger.Instance;
        }

        /// <summary>
        /// Template Method: Takes raw input, converts to ONNX tensors.
        /// Must be implemented by specific app subclasses.
        /// </summary>
        protected abstract List<NamedOnnxValue> PreProcess(TInput input);

        /// <summary>
        /// Template Method: Takes raw ONNX output, converts to domain objects.
        /// Must be implemented by specific app subclasses.
        /// </summary>
        protected abstract TOutput PostProcess(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output);

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

                    // Create the session (This is where the heavy lifting happens)
                    _session = new InferenceSession(_options.ModelPath, sessionOptions);
                    _runOptions = new RunOptions();

                    _logger.LogInformation("Model loaded successfully.");
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

        /// <summary>
        /// Configures the session options based on the OS and Hardware capabilities.
        /// </summary>
        private SessionOptions ConfiguredSessionOptions()
        {
            var options = new SessionOptions();

            // Optimization: heavily reduces CPU usage for standard models
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
                    // Note: Requires Microsoft.ML.OnnxRuntime.DirectML NuGet package
                    options.AppendExecutionProvider_DML(_options.DeviceId);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning($"DirectML failed to load. Falling back to CPU. Error: {ex.Message}");
                    options.AppendExecutionProvider_CPU();
                }
            }
            // 2. macOS Strategy (CPU Fallback)
            // NOTE: ONNX Runtime's CoreML execution provider bindings are not currently available
            // as a supported Microsoft.ML.OnnxRuntime.* NuGet package.
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                _logger.LogInformation("Platform detected: macOS. Using CPU provider.");
                options.AppendExecutionProvider_CPU();
            }
            // 3. Linux/Other Strategy (CPU Fallback)
            else
            {
                _logger.LogInformation("Platform detected: Linux/Other. Using default CPU provider.");
                options.AppendExecutionProvider_CPU();
            }

            return options;
        }

        private void PerformWarmup()
        {
            try
            {
                _logger.LogInformation("Performing warmup inference to compile shaders...");

                // Get input metadata to construct a valid dummy tensor
                // This is generic; it creates a dummy tensor based on the model's actual expected shape
                var inputs = new List<NamedOnnxValue>();
                foreach (var inputMeta in _session.InputMetadata)
                {
                    var shape = inputMeta.Value.Dimensions.Select(d => d == -1 ? 1 : d).ToArray(); // Handle dynamic axes
                    var tensorData = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(shape);
                    // Note: This assumes float input. For highly specific models (like int64 text models), 
                    // subclasses might need to override a specific 'Warmup' method.
                    inputs.Add(NamedOnnxValue.CreateFromTensor(inputMeta.Key, tensorData));
                }

                using var results = _session.Run(inputs);
                _logger.LogInformation("Warmup complete.");
            }
            catch (Exception ex)
            {
                _logger.LogWarning($"Warmup failed (non-critical). Error: {ex.Message}");
            }
        }

        public virtual async Task<TOutput> PredictAsync(TInput input)
        {
            if (!_isLoaded) await LoadAsync();

            return await Task.Run(() =>
            {
                // 1. Pre-process (Client code)
                var inputTensors = PreProcess(input);

                try
                {
                    // 2. Inference (Engine code)
                    // The 'using' block ensures native memory held by output tensors is freed immediately
                    var outputNames = _session.OutputMetadata.Keys.ToArray();
                    using var outputTensors = _session.Run(inputTensors, outputNames, _runOptions);

                    // 3. Post-process (Client code)
                    return PostProcess(outputTensors);
                }
                finally
                {
                    // Clean up input tensors if they hold native memory (rare but possible in some bindings)
                    foreach (var t in inputTensors)
                    {
                        // Some advanced implementations might need explicit disposal here
                    }
                }
            });
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
