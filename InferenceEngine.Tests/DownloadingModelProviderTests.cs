using System;
using System.IO;
using System.Net.Http;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using InferenceEngine.Core.ModelProviders;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace InferenceEngine.Tests;

public class DownloadingModelProviderTests : IDisposable
{
    private const string Payload = "fake model payload for hash tests";

    private readonly string _testDir;

    public DownloadingModelProviderTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"download-tests-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try
            {
                Directory.Delete(_testDir, true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Suppress filesystem exceptions during cleanup to prevent escaping from Dispose (CA1065)
            }
        }
    }

    [Fact]
    public void Constructor_WithNullModelName_ShouldThrowArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => 
            new DownloadingModelProvider(null!, new[] { "http://example.com/model.onnx" }, _testDir));
    }

    [Fact]
    public void Constructor_WithNullUrls_ShouldThrowArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => 
            new DownloadingModelProvider("model.onnx", null!, _testDir));
    }

    [Fact]
    public void Constructor_WithEmptyUrls_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => 
            new DownloadingModelProvider("model.onnx", Array.Empty<string>(), _testDir));
    }

    [Fact]
    public async Task GetModelPathAsync_WithCachedModel_ShouldReturnCachedPath()
    {
        var modelName = "cached-model.onnx";
        var cacheDir = Path.Combine(_testDir, "cache1");
        Directory.CreateDirectory(cacheDir);
        var cachedPath = Path.Combine(cacheDir, modelName);
        File.WriteAllText(cachedPath, "cached model content");

        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { "http://example.com/model.onnx" }, 
            cacheDir,
            NullLogger.Instance);

        var result = await provider.GetModelPathAsync();

        Assert.Equal(cachedPath, result);
        Assert.Equal("cached model content", File.ReadAllText(result));
    }

    [Fact]
    public async Task GetModelPathAsync_WithAllUrlsFailing_ShouldThrowInvalidOperationException()
    {
        var modelName = "all-fail.onnx";
        var cacheDir = Path.Combine(_testDir, "cache4");

        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { 
                "http://localhost:65534/fail1.onnx", 
                "http://localhost:65533/fail2.onnx" 
            }, 
            cacheDir,
            NullLogger.Instance);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => provider.GetModelPathAsync());
        Assert.Contains("Failed to download model", ex.Message);
        Assert.Contains("all-fail.onnx", ex.Message);
    }

    [Fact]
    public async Task GetModelPathAsync_CreatesCacheDirectory_IfNotExists()
    {
        var modelName = "newdir-model.onnx";
        var cacheDir = Path.Combine(_testDir, "new", "cache", "dir");

        Assert.False(Directory.Exists(cacheDir));

        // Will fail to download but should create directory first
        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { "http://localhost:65532/test.onnx" }, 
            cacheDir,
            NullLogger.Instance);

        try
        {
            await provider.GetModelPathAsync();
        }
        catch (InvalidOperationException)
        {
            // Expected to fail
        }

        Assert.True(Directory.Exists(cacheDir));
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public async Task GetModelPathAsync_WithRealDownload_ShouldCacheAndReturnPath()
    {
        // Only run when explicitly enabled as an integration test
        var integrationFlag = Environment.GetEnvironmentVariable("INFERENCEENGINE_INTEGRATION_TESTS");
        Skip.IfNot(string.Equals(integrationFlag, "1", StringComparison.Ordinal),
            "Set INFERENCEENGINE_INTEGRATION_TESTS=1 to run this integration test.");
        
        // Skip if network is not available
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");

        var modelName = "yolo11n.onnx";
        var cacheDir = Path.Combine(_testDir, "real-download");
        
        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx" }, 
            cacheDir,
            NullLogger.Instance);

        var result = await provider.GetModelPathAsync();

        Assert.True(File.Exists(result));
        Assert.True(new FileInfo(result).Length > 0);
        Assert.Equal(Path.Combine(cacheDir, modelName), result);
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public async Task GetModelPathAsync_WithFirstUrlFailing_ShouldTrySecondUrl()
    {
        // Only run when explicitly enabled as an integration test
        var integrationFlag = Environment.GetEnvironmentVariable("INFERENCEENGINE_INTEGRATION_TESTS");
        Skip.IfNot(string.Equals(integrationFlag, "1", StringComparison.Ordinal),
            "Set INFERENCEENGINE_INTEGRATION_TESTS=1 to run this integration test.");
        
        // Skip if network is not available
        Skip.IfNot(NetworkIsAvailable(), "Network is not available");

        var modelName = "yolo11n.onnx";
        var cacheDir = Path.Combine(_testDir, "failover");
        
        // First URL is invalid, second is valid
        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { 
                "http://localhost:65531/nonexistent.onnx",
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
            }, 
            cacheDir,
            NullLogger.Instance);

        var result = await provider.GetModelPathAsync();

        Assert.True(File.Exists(result));
        Assert.True(new FileInfo(result).Length > 0);
    }

    [Fact]
    public async Task GetModelPathAsync_WithCancellation_ShouldCancelOperation()
    {
        var modelName = "cancel-model.onnx";
        var cacheDir = Path.Combine(_testDir, "cancel");
        var cts = new CancellationTokenSource();
        
        var provider = new DownloadingModelProvider(
            modelName, 
            new[] { "http://localhost:65530/cancel.onnx" }, 
            cacheDir,
            NullLogger.Instance);

        cts.Cancel();
        
        // Should throw either OperationCanceledException / TaskCanceledException
        // or InvalidOperationException wrapping one of those
        var ex = await Assert.ThrowsAnyAsync<Exception>(() => provider.GetModelPathAsync(cts.Token));

        Assert.True(
            ex is OperationCanceledException or TaskCanceledException ||
            (ex is InvalidOperationException ioe &&
             (ioe.InnerException is OperationCanceledException or TaskCanceledException)),
            "Expected a cancellation-related exception.");
    }

    // --- MEDIUM-1 hardening: path validation (fail closed) ---

    [Theory]
    [InlineData("../x")]
    [InlineData("..\\x")]
    [InlineData("/abs")]
    [InlineData("\\\\srv\\share\\m.onnx")]
    [InlineData("a/b")]
    [InlineData("a\\b")]
    [InlineData("..")]
    [InlineData("a..")]
    [InlineData("model\0.onnx")]
    public void Constructor_WithTraversalModelName_ShouldThrowArgumentException(string modelName)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new DownloadingModelProvider(modelName, new[] { "http://example.com/model.onnx" }, _testDir));
        Assert.Contains(modelName.TrimEnd('\0'), ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Constructor_WithTraversalCacheDirectory_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DownloadingModelProvider("model.onnx", new[] { "http://example.com/model.onnx" }, "../../escape"));
        Assert.Throws<ArgumentException>(() =>
            new DownloadingModelProvider("model.onnx", new[] { "http://example.com/model.onnx" }, " "));
    }

    // --- MEDIUM-1 hardening: SHA-256 integrity verification ---

    [Fact]
    public async Task GetModelPathAsync_WithMatchingSha256_ShouldAcceptDownload()
    {
        var payload = Encoding.UTF8.GetBytes(Payload);
        var sha256 = Convert.ToHexStringLower(SHA256.HashData(payload));
        var cacheDir = Path.Combine(_testDir, "sha-ok");
        var provider = CreateProvider("model.onnx", cacheDir, sha256, new StubHandler(_ => Ok(payload)));

        var result = await provider.GetModelPathAsync();

        Assert.Equal(Path.Combine(cacheDir, "model.onnx"), result);
        Assert.Equal(Payload, File.ReadAllText(result));
        AssertNoTempFiles(cacheDir);
    }

    [Fact]
    public async Task GetModelPathAsync_WithMismatchingSha256_ShouldThrowAndRemoveFile()
    {
        var payload = Encoding.UTF8.GetBytes(Payload);
        var wrongSha = new string('a', 64);
        var cacheDir = Path.Combine(_testDir, "sha-bad");
        var provider = CreateProvider("model.onnx", cacheDir, wrongSha, new StubHandler(_ => Ok(payload)));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => provider.GetModelPathAsync());

        Assert.Contains(wrongSha, ex.Message, StringComparison.Ordinal);
        Assert.Contains("SHA-256", ex.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(cacheDir, "model.onnx")), "final file must not survive a mismatch");
        AssertNoTempFiles(cacheDir);
    }

    [Fact]
    public void Constructor_WithMalformedSha256_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DownloadingModelProvider("model.onnx", new[] { "http://example.com/m" }, _testDir, NullLogger.Instance, "zz"));
        Assert.Throws<ArgumentException>(() =>
            new DownloadingModelProvider("model.onnx", new[] { "http://example.com/m" }, _testDir, NullLogger.Instance, new string('g', 64)));
    }

    [Fact]
    public async Task GetModelPathAsync_WithoutSha256_ShouldStillDownload()
    {
        // Backward compatibility: entries without sha256 keep working (logged as unverified).
        var payload = Encoding.UTF8.GetBytes(Payload);
        var cacheDir = Path.Combine(_testDir, "sha-none");
        var provider = CreateProvider("model.onnx", cacheDir, sha256: null, new StubHandler(_ => Ok(payload)));

        var result = await provider.GetModelPathAsync();

        Assert.Equal(Payload, File.ReadAllText(result));
    }

    // --- MEDIUM-1 hardening: atomic downloads ---

    [Fact]
    public async Task GetModelPathAsync_WithInterruptedDownload_ShouldLeaveNoFilesAndRetrySucceeds()
    {
        var payload = Encoding.UTF8.GetBytes(Payload);
        var cacheDir = Path.Combine(_testDir, "interrupted");
        var handler = new StubHandler(_ => OkWithBrokenStream(payload, bytesBeforeFailure: 5));
        var provider = CreateProvider("model.onnx", cacheDir, sha256: null, handler);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => provider.GetModelPathAsync());
        Assert.IsType<IOException>(ex.InnerException);

        Assert.False(File.Exists(Path.Combine(cacheDir, "model.onnx")), "partial download must never become the final file");
        AssertNoTempFiles(cacheDir);

        // Retry once the source works again: succeeds cleanly.
        handler.Respond = _ => Ok(payload);
        var result = provider.GetModelPathAsync();
        Assert.Equal(Path.Combine(cacheDir, "model.onnx"), (await result));
        Assert.Equal(Payload, File.ReadAllText(Path.Combine(cacheDir, "model.onnx")));
        AssertNoTempFiles(cacheDir);
    }

    // --- MEDIUM-1 hardening: size cap ---

    [Fact]
    public async Task GetModelPathAsync_WithDeclaredContentLengthOverCap_ShouldAbort()
    {
        var cacheDir = Path.Combine(_testDir, "cap-declared");
        var handler = new StubHandler(_ =>
        {
            var content = new ByteArrayContent(Array.Empty<byte>());
            content.Headers.ContentLength = 100; // declares more than the 10-byte cap
            return OkResponse(content);
        });
        var provider = CreateProvider("model.onnx", cacheDir, sha256: null, handler, maxDownloadBytes: 10);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => provider.GetModelPathAsync());
        Assert.Contains("exceeds the maximum", ex.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(cacheDir, "model.onnx")));
        AssertNoTempFiles(cacheDir);
    }

    [Fact]
    public async Task GetModelPathAsync_WithRunningByteCountOverCap_ShouldAbort()
    {
        var cacheDir = Path.Combine(_testDir, "cap-running");
        var handler = new StubHandler(_ => OkWithLengthHeaderDisabled(OneHundredBytes(), contentLength: null));
        var provider = CreateProvider("model.onnx", cacheDir, sha256: null, handler, maxDownloadBytes: 10);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => provider.GetModelPathAsync());
        Assert.Contains("exceeded the maximum", ex.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Combine(cacheDir, "model.onnx")));
        AssertNoTempFiles(cacheDir);
    }

    [Fact]
    public void Constructor_WithNonPositiveMaxDownloadBytes_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DownloadingModelProvider("model.onnx", new[] { "http://example.com/m" }, _testDir,
                NullLogger.Instance, null, maxDownloadBytes: 0));
        Assert.Throws<ArgumentException>(() =>
            new DownloadingModelProvider("model.onnx", new[] { "http://example.com/m" }, _testDir,
                NullLogger.Instance, null, maxDownloadBytes: -1));
    }

    private static byte[] OneHundredBytes() => new byte[100];

    private DownloadingModelProvider CreateProvider(
        string modelName,
        string cacheDir,
        string? sha256,
        StubHandler handler,
        long maxDownloadBytes = DownloadingModelProvider.DefaultMaxDownloadBytes)
    {
        return new DownloadingModelProvider(
            modelName,
            new[] { "http://example.com/" + modelName },
            cacheDir,
            NullLogger.Instance,
            sha256,
            maxDownloadBytes,
            new HttpClient(handler));
    }

    private static HttpResponseMessage OkResponse(HttpContent content) => new()
    {
        StatusCode = System.Net.HttpStatusCode.OK,
        Content = content,
    };

    private static HttpResponseMessage Ok(byte[] payload) => OkResponse(new ByteArrayContent(payload));

    private static HttpResponseMessage OkWithLengthHeaderDisabled(byte[] payload, long? contentLength)
    {
        var response = OkResponse(new StubStreamContent(new MemoryStream(payload)));
        response.Content.Headers.ContentLength = contentLength;
        return response;
    }

    private static HttpResponseMessage OkWithBrokenStream(byte[] payload, int bytesBeforeFailure)
        => OkResponse(new StubStreamContent(new MemoryStream(payload), bytesBeforeFailure));

    private void AssertNoTempFiles(string cacheDir)
    {
        if (Directory.Exists(cacheDir))
        {
            Assert.Empty(Directory.GetFiles(cacheDir, "*.downloading"));
        }
    }

    private sealed class StubHandler : HttpMessageHandler
    {
        public Func<HttpRequestMessage, HttpResponseMessage> Respond;

        public StubHandler(Func<HttpRequestMessage, HttpResponseMessage> respond) => Respond = respond;

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(Respond(request));
    }

    /// <summary>Content exposing a raw stream, optionally failing after N readable bytes.</summary>
    private sealed class StubStreamContent : HttpContent
    {
        private readonly MemoryStream _stream;
        private readonly int? _failAfterBytes;

        public StubStreamContent(MemoryStream stream, int? failAfterBytes = null)
        {
            _stream = stream;
            _failAfterBytes = failAfterBytes;
        }

        protected override async Task SerializeToStreamAsync(Stream stream, TransportContext? context)
        {
            await _stream.CopyToAsync(stream);
        }

        protected override bool TryComputeLength(out long length)
        {
            length = _stream.Length;
            return true;
        }

        protected override Task<Stream> CreateContentReadStreamAsync()
        {
            return Task.FromResult<Stream>(_failAfterBytes is int fail
                ? new ThrowingStream(_stream, fail)
                : _stream);
        }
    }

    /// <summary>A stream that yields <paramref name="bytesBeforeFailure"/> bytes then throws.</summary>
    private sealed class ThrowingStream : Stream
    {
        private readonly MemoryStream _inner;
        private readonly int _bytesBeforeFailure;
        private bool _thrown;

        public ThrowingStream(MemoryStream inner, int bytesBeforeFailure)
        {
            _inner = inner;
            _bytesBeforeFailure = bytesBeforeFailure;
        }

        public override bool CanRead => !_thrown;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (!_thrown && _inner.Position >= _bytesBeforeFailure)
            {
                _thrown = true;
                throw new IOException("simulated network failure mid-stream");
            }

            var toRead = (int)Math.Min(buffer.Length, Math.Max(1, _bytesBeforeFailure - _inner.Position));
            return await _inner.ReadAsync(buffer[..toRead], cancellationToken);
        }

        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override void Flush() => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private static bool NetworkIsAvailable()
    {
        try
        {
            using var client = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromSeconds(5) };
            var response = client.SendAsync(new System.Net.Http.HttpRequestMessage(System.Net.Http.HttpMethod.Head, "https://github.com")).GetAwaiter().GetResult();
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
}
