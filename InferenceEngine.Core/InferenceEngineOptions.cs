using InferenceEngine.Core.Diagnostics;
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

        /// <summary>
        /// Directory for the MIGraphX compiled-model (.mxr) cache (Linux x64 only). When
        /// enabled, the MIGraphX execution provider persists each compiled program to disk and
        /// reloads it on later startups, cutting engine load from a ~1-2 minute kernel compile
        /// to a ~1-2 second cache load. Cache files are keyed by MIGraphX version, graph
        /// identity, GPU architecture, and input shapes, so model swaps, ROCm/MIGraphX
        /// upgrades, or a different GPU naturally miss the cache and recompile.
        /// <list type="bullet">
        /// <item><c>null</c> (default): use <c>%LocalAppData%/InferenceEngine/MIGraphXModelCache</c>.</item>
        /// <item><c>""</c> (empty string): disable .mxr caching (every startup recompiles).</item>
        /// <item>non-empty: use that directory verbatim (created if missing).</item>
        /// </list>
        /// Note: quantization flags (fp16/int8) are not part of the cache key — delete the
        /// directory after changing quantization settings.
        /// </summary>
        public string? MIGraphXModelCacheDirectory { get; set; }

        /// <summary>
        /// Optional callback invoked on every engine status transition (Loading → Loaded or
        /// Failed), on the LoadAsync thread pool thread. Receives the full
        /// <see cref="Diagnostics.InferenceEngineStatus"/> object model — on failure its
        /// <c>Info</c> carries the partial provider-selection trail and <c>LoadError</c> the
        /// error. Exceptions thrown by the callback are caught and logged as warnings; they do
        /// not abort the engine.
        /// <para>
        /// This is the recommended hook; <see cref="OnSessionInitialized"/> remains as the
        /// success-only equivalent.
        /// </para>
        /// </summary>
        public Action<Diagnostics.InferenceEngineStatus>? OnStatusChanged { get; set; }

        /// <summary>
        /// Optional callback invoked once, on the LoadAsync thread pool thread, after the session
        /// is created (including fallback paths). Receives the fully-populated InferenceEngineInfo.
        /// Exceptions thrown by the callback are caught and logged as warnings; they do not
        /// abort the engine.
        /// <para>
        /// Invoked at most once per LoadAsync call. If LoadAsync is called again after the engine
        /// is already loaded, the callback does not fire.
        /// </para>
        /// <para>
        /// Runs on a Task.Run thread pool thread, not the caller's synchronization context.
        /// Consumers that touch UI state must marshal to the UI thread themselves.
        /// </para>
        /// </summary>
        public Action<InferenceEngineInfo>? OnSessionInitialized { get; set; }
    }
}
