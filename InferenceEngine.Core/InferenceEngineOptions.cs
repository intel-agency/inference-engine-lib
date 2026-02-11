using InferenceEngine.Core.ModelProviders;

namespace InferenceEngine.Core
{
    /// <summary>
    /// Configuration options for inference engine initialization.
    /// </summary>
    public class InferenceEngineOptions
    {
        /// <summary>
        /// Optional model provider. If not set, the engine will use its default provider
        /// (as defined by CreateDefaultModelProvider() in the specific engine class).
        /// </summary>
        public IModelProvider? ModelProvider { get; set; }

        /// <summary>
        /// Convenience property for local model paths.
        /// Setting this creates a LocalModelProvider internally.
        /// </summary>
        public string? ModelPath
        {
            get => ModelProvider is LocalModelProvider local ? local.ModelPath : null;
            set => ModelProvider = value is not null ? new LocalModelProvider(value) : null;
        }

        /// <summary>
        /// If true, attempts to use DirectML (Windows) or CoreML (macOS).
        /// If false, forces CPU execution.
        /// </summary>
        public bool UseGpuAcceleration { get; set; } = true;

        /// <summary>
        /// Specific GPU Device ID (default 0). Useful for multi-GPU setups.
        /// </summary>
        public int DeviceId { get; set; } = 0;

        /// <summary>
        /// If true, runs a dummy inference during LoadAsync to force 
        /// shader compilation/caching immediately.
        /// </summary>
        public bool WarmupOnLoad { get; set; } = true;
    }
}
