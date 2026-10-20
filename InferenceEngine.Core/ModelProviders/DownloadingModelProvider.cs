using System;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace InferenceEngine.Core.ModelProviders
{
    /// <summary>
    /// Model provider that downloads models from URLs with local caching.
    /// If the model already exists in the cache, returns the cached path immediately.
    /// Downloads are hardened (security assessment MEDIUM-1): model names are validated
    /// against path traversal, files are downloaded atomically via a temp file, an optional
    /// SHA-256 checksum is enforced, a size cap aborts runaway downloads, and redirects
    /// are capped.
    /// </summary>
    public class DownloadingModelProvider : IModelProvider
    {
        /// <summary>Default cap applied when <c>maxDownloadBytes</c> is not specified (2 GiB).</summary>
        public const long DefaultMaxDownloadBytes = 2L * 1024 * 1024 * 1024;

        private const int MaxAutomaticRedirections = 5;

        private readonly string _modelName;
        private readonly string[] _downloadUrls;
        private readonly string _cacheDirectory;
        private readonly string? _expectedSha256;
        private readonly long _maxDownloadBytes;
        private readonly ILogger _logger;
        private readonly HttpClient _http;
        private static readonly HttpClient DefaultHttp = CreateHttpClient();

        /// <summary>
        /// Creates a provider that downloads models from URLs with caching.
        /// </summary>
        /// <param name="modelName">Name of the model file (e.g., "yolov8n.onnx"). Must be a plain
        /// file name: directory separators, rooted paths, ".." segments, and invalid filename
        /// characters are rejected.</param>
        /// <param name="downloadUrls">One or more URLs to try downloading from (tried in order)</param>
        /// <param name="cacheDirectory">Directory to cache downloaded models. Defaults to local app data.</param>
        /// <param name="logger">Optional logger</param>
        /// <param name="sha256">
        /// Optional expected SHA-256 of the model file as 64 lowercase hex characters. When set, the
        /// downloaded file is verified against it and a mismatch rejects (and deletes) the download.
        /// When null, the download is accepted but logged as unverified.
        /// </param>
        /// <param name="maxDownloadBytes">
        /// Maximum allowed download size in bytes (validated against Content-Length and the running
        /// byte count). Must be positive; defaults to <see cref="DefaultMaxDownloadBytes"/> (2 GiB).
        /// </param>
        /// <param name="httpClient">Optional HTTP client to use (mainly for tests). Defaults to a
        /// shared client with automatic redirects capped at 5.</param>
        /// <exception cref="ArgumentNullException">A required argument is null.</exception>
        /// <exception cref="ArgumentException">
        /// <paramref name="modelName"/> contains path traversal or invalid characters, no download
        /// URL is provided, <paramref name="sha256"/> is malformed, or
        /// <paramref name="maxDownloadBytes"/> is not positive.
        /// </exception>
        public DownloadingModelProvider(
            string modelName,
            string[] downloadUrls,
            string? cacheDirectory = null,
            ILogger? logger = null,
            string? sha256 = null,
            long maxDownloadBytes = DefaultMaxDownloadBytes,
            HttpClient? httpClient = null)
        {
            _modelName = ValidateModelName(modelName);
            _downloadUrls = downloadUrls ?? throw new ArgumentNullException(nameof(downloadUrls));
            _cacheDirectory = ValidateCacheDirectory(cacheDirectory ?? GetDefaultCacheDirectory());
            _expectedSha256 = ValidateSha256(sha256);
            _maxDownloadBytes = maxDownloadBytes > 0
                ? maxDownloadBytes
                : throw new ArgumentException("Maximum download size must be positive.", nameof(maxDownloadBytes));
            _logger = logger ?? NullLogger.Instance;
            _http = httpClient ?? DefaultHttp;

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
            var tempPath = modelPath + ".downloading";

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
                    await DownloadModelAsync(url, tempPath, cancellationToken).ConfigureAwait(false);
                    File.Move(tempPath, modelPath, overwrite: true);
                    _logger.LogInformation("Successfully downloaded {ModelName} from {Url}", _modelName, url);
                    return modelPath;
                }
                catch (Exception ex)
                {
                    DeleteFileBestEffort(tempPath);
                    lastError = ex;
                    _logger.LogWarning("Failed to download from {Url}: {Error}", url, ex.Message);
                    // Try next URL
                }
            }

            throw new InvalidOperationException(
                $"Failed to download model '{_modelName}' from all sources. Last error: {lastError?.Message}",
                lastError);
        }

        private async Task DownloadModelAsync(string url, string tempPath, CancellationToken ct)
        {
            using var response = await _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            if (response.Content.Headers.ContentLength is long declared && declared > _maxDownloadBytes)
            {
                throw new InvalidOperationException(
                    $"Download from '{url}' declares {declared} bytes, which exceeds the maximum of {_maxDownloadBytes} bytes.");
            }

            await using var fs = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None);
            await using var source = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);

            var buffer = new byte[81920];
            long total = 0;
            int read;
            while ((read = await source.ReadAsync(buffer.AsMemory(0, buffer.Length), ct).ConfigureAwait(false)) > 0)
            {
                total += read;
                if (total > _maxDownloadBytes)
                {
                    throw new InvalidOperationException(
                        $"Download from '{url}' exceeded the maximum of {_maxDownloadBytes} bytes.");
                }

                await fs.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
            }

            if (_expectedSha256 is null)
            {
                _logger.LogWarning(
                    "Model '{ModelName}' downloaded from '{Url}' without integrity verification: no sha256 is configured for it.",
                    _modelName, url);
                return;
            }

            fs.Close();
            var actual = await ComputeFileSha256Async(tempPath, ct).ConfigureAwait(false);
            if (!string.Equals(actual, _expectedSha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    $"SHA-256 mismatch for model '{_modelName}' downloaded from '{url}': " +
                    $"expected {_expectedSha256}, got {actual}.");
            }

            _logger.LogInformation("SHA-256 verified for model '{ModelName}' downloaded from '{Url}'.", _modelName, url);
        }

        private static async Task<string> ComputeFileSha256Async(string path, CancellationToken ct)
        {
            await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var sha = SHA256.Create();
            var hash = await sha.ComputeHashAsync(stream, ct).ConfigureAwait(false);
            var sb = new StringBuilder(hash.Length * 2);
            foreach (var b in hash)
            {
                sb.Append(b.ToString("x2"));
            }

            return sb.ToString();
        }

        private static HttpClient CreateHttpClient()
        {
            var handler = new HttpClientHandler
            {
                MaxAutomaticRedirections = MaxAutomaticRedirections,
                AllowAutoRedirect = true,
            };
            return new HttpClient(handler);
        }

        private static string ValidateModelName(string? modelName)
        {
            if (modelName is null)
            {
                throw new ArgumentNullException(nameof(modelName));
            }

            if (modelName.Length == 0)
            {
                throw new ArgumentException("Model name must not be empty.", nameof(modelName));
            }

            if (modelName.IndexOf('/') >= 0 || modelName.IndexOf('\\') >= 0)
            {
                throw new ArgumentException(
                    $"Model name '{modelName}' must not contain directory separators.", nameof(modelName));
            }

            if (Path.IsPathRooted(modelName))
            {
                throw new ArgumentException(
                    $"Model name '{modelName}' must be a plain file name, not a rooted path.", nameof(modelName));
            }

            if (modelName == ".." || modelName.Contains("..", StringComparison.Ordinal))
            {
                throw new ArgumentException(
                    $"Model name '{modelName}' must not contain '..' segments.", nameof(modelName));
            }

            if (modelName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            {
                throw new ArgumentException(
                    $"Model name '{modelName}' contains invalid filename characters.", nameof(modelName));
            }

            return modelName;
        }

        private static string ValidateCacheDirectory(string cacheDirectory)
        {
            if (string.IsNullOrWhiteSpace(cacheDirectory))
            {
                throw new ArgumentException("Cache directory must not be empty or whitespace.", nameof(cacheDirectory));
            }

            if (cacheDirectory == ".." || cacheDirectory.Contains("..", StringComparison.Ordinal))
            {
                throw new ArgumentException(
                    $"Cache directory '{cacheDirectory}' must not contain '..' segments.", nameof(cacheDirectory));
            }

            if (cacheDirectory.IndexOfAny(Path.GetInvalidPathChars()) >= 0)
            {
                throw new ArgumentException(
                    $"Cache directory '{cacheDirectory}' contains invalid path characters.", nameof(cacheDirectory));
            }

            return cacheDirectory;
        }

        private static string? ValidateSha256(string? sha256)
        {
            if (sha256 is null)
            {
                return null;
            }

            if (sha256.Length != 64)
            {
                throw new ArgumentException("sha256 must be exactly 64 hex characters.", nameof(sha256));
            }

            foreach (var c in sha256)
            {
                if (!Uri.IsHexDigit(c))
                {
                    throw new ArgumentException("sha256 must contain only hex characters.", nameof(sha256));
                }
            }

            return sha256.ToLowerInvariant();
        }

        private void DeleteFileBestEffort(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                _logger.LogWarning(ex, "Failed to delete temporary download file '{Path}'.", path);
            }
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
