namespace InferenceEngine.Core
{
    public class InferenceEngineOptions
    {
        /// <summary>
        /// Absolute path to the .onnx model file.
        /// </summary>
        public string ModelPath { get; set; } = string.Empty;

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
