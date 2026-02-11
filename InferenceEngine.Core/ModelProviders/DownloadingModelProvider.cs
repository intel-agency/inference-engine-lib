using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace InferenceEngine.Core.ModelProviders
{
    /// <summary>
    /// Model provider that downloads models from URLs with local caching.
    /// If the model already exists in the cache, returns the cached path immediately.
    /// </summary>
    public class DownloadingModelProvider : IModelProvider
    {
        private readonly string _modelName;
        private readonly string[] _downloadUrls;
        private readonly string _cacheDirectory;
        private readonly ILogger _logger;
        private static readonly HttpClient Http = new();

        /// <summary>
        /// Creates a provider that downloads models from URLs with caching.
        /// </summary>
        /// <param name="modelName">Name of the model file (e.g., "yolov8n.onnx")</param>
        /// <param name="downloadUrls">One or more URLs to try downloading from (tried in order)</param>
        /// <param name="cacheDirectory">Directory to cache downloaded models. Defaults to local app data.</param>
        /// <param name="logger">Optional logger</param>
        public DownloadingModelProvider(
            string modelName,
            string[] downloadUrls,
            string? cacheDirectory = null,
            ILogger? logger = null)
        {
            _modelName = modelName ?? throw new ArgumentNullException(nameof(modelName));
            _downloadUrls = downloadUrls ?? throw new ArgumentNullException(nameof(downloadUrls));
            _cacheDirectory = cacheDirectory ?? GetDefaultCacheDirectory();
            _logger = logger ?? NullLogger.Instance;

            if (_downloadUrls.Length == 0)
            {
                throw new ArgumentException("At least one download URL must be provided", nameof(downloadUrls));
            }
        }

        /// <inheritdoc />
        public async Task<string> GetModelPathAsync(CancellationToken cancellationToken = default)
        {
            Directory.CreateDirectory(_cacheDirectory);
            var modelPath = Path.Combine(_cacheDirectory, _modelName);

            // Return cached model if available
            if (File.Exists(modelPath))
            {
                _logger.LogDebug("Using cached model: {ModelPath}", modelPath);
                return modelPath;
            }

            _logger.LogInformation("Model not found in cache. Downloading {ModelName}...", _modelName);

            // Try each URL until one succeeds
            Exception? lastError = null;
            foreach (var url in _downloadUrls)
            {
                try
                {
                    _logger.LogDebug("Attempting download from: {Url}", url);
                    await DownloadModelAsync(url, modelPath, cancellationToken);
                    _logger.LogInformation("Successfully downloaded {ModelName} from {Url}", _modelName, url);
                    return modelPath;
                }
                catch (Exception ex)
                {
                    lastError = ex;
                    _logger.LogWarning("Failed to download from {Url}: {Error}", url, ex.Message);
                    // Try next URL
                }
            }

            throw new InvalidOperationException(
                $"Failed to download model '{_modelName}' from all sources. Last error: {lastError?.Message}",
                lastError);
        }

        private static async Task DownloadModelAsync(string url, string destinationPath, CancellationToken ct)
        {
            using var response = await Http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct);
            response.EnsureSuccessStatusCode();

            await using var fs = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None);
            await response.Content.CopyToAsync(fs, ct);
        }

        private static string GetDefaultCacheDirectory()
        {
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "InferenceEngine",
                "Models");
        }
    }
}
