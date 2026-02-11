using System;
using System.Threading.Tasks;

namespace InferenceEngine.Core
{
    /// <summary>
    /// A generic contract for an AI Inference Engine.
    /// Implementation details (ONNX, TensorFlow, Mock) are abstracted away.
    /// </summary>
    /// <typeparam name="TInput">The type of input data (e.g., byte[] for images, string for text).</typeparam>
    /// <typeparam name="TOutput">The type of output data (e.g., List&lt;Prediction&gt;, string response).</typeparam>
    public interface IInferenceEngine<TInput, TOutput> : IDisposable
    {
        /// <summary>
        /// Initializes the engine, loads the model, and performs hardware warmup.
        /// This handles the "First Run" latency (e.g., DirectML graph compilation).
        /// </summary>
        Task LoadAsync();

        /// <summary>
        /// Runs the inference pipeline: PreProcess -> Inference -> PostProcess.
        /// </summary>
        /// <param name="input">The raw input data.</param>
        /// <returns>The processed prediction result.</returns>
        Task<TOutput> PredictAsync(TInput input);
    }
}
