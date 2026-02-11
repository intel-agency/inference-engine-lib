using System.Threading;
using System.Threading.Tasks;

namespace InferenceEngine.Core.ModelProviders
{
    /// <summary>
    /// Abstraction for model acquisition. Implementations can load models from
    /// local files, download from URLs, fetch from cloud storage, etc.
    /// </summary>
    public interface IModelProvider
    {
        /// <summary>
        /// Gets the local path to the model file.
        /// Implementations should ensure the model is available at the returned path
        /// (downloading if necessary) before returning.
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for the operation</param>
        /// <returns>Absolute path to the .onnx model file</returns>
        Task<string> GetModelPathAsync(CancellationToken cancellationToken = default);
    }
}
