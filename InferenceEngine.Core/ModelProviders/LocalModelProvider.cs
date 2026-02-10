using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace InferenceEngine.Core.ModelProviders
{
    /// <summary>
    /// Model provider for locally stored model files.
    /// Validates that the file exists but does not perform any network operations.
    /// </summary>
    public class LocalModelProvider : IModelProvider
    {
        /// <summary>
        /// The model file path (absolute or relative).
        /// </summary>
        public string ModelPath { get; }

        /// <summary>
        /// Creates a provider for a local model file.
        /// </summary>
        /// <param name="modelPath">Absolute or relative path to the .onnx model file</param>
        public LocalModelProvider(string modelPath)
        {
            ModelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
        }

        /// <inheritdoc />
        public Task<string> GetModelPathAsync(CancellationToken cancellationToken = default)
        {
            // Resolve relative paths to absolute
            var fullPath = Path.GetFullPath(ModelPath);

            if (!File.Exists(fullPath))
            {
                throw new FileNotFoundException($"Model file not found: {fullPath}", fullPath);
            }

            return Task.FromResult(fullPath);
        }
    }
}
