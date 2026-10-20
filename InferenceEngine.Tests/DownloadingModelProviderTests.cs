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

    // --- W3: constructor validation corners + default cache directory ---

    [Fact]
    public void Constructor_WithEmptyModelName_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new DownloadingModelProvider(string.Empty, new[] { "http://example.com/m" }, _testDir));
    }

    [Fact]
    public void Constructor_WithInvalidPathCharsInCacheDirectory_ShouldThrowArgumentException()
    {
        // '\0' appears in Path.GetInvalidPathChars() on every supported platform.
        Assert.Throws<ArgumentException>(() =>
            new DownloadingModelProvider("model.onnx", new[] { "http://example.com/m" }, "ca\0che"));
    }

    [Fact]
    public void Constructor_WithNullCacheDirectory_ShouldFallBackToDefaultCacheDirectory()
    {
        // Exercises the LocalApplicationData default; construction only (no download, no writes).
        var provider = new DownloadingModelProvider(
            "model.onnx", new[] { "http://example.com/model.onnx" }, cacheDirectory: null);
        Assert.NotNull(provider);
    }

    // --- W3: multi-URL fallback with a stubbed failing-then-succeeding source ---

    [Fact]
    public async Task GetModelPathAsync_WithFirstUrlFailingAndSecondSucceeding_ShouldUseSecondUrl()
    {
        var payload = Encoding.UTF8.GetBytes(Payload);
        var cacheDir = Path.Combine(_testDir, "multi-url");
        var handler = new StubHandler(req =>
            req.RequestUri!.AbsoluteUri.EndsWith("secondary")
                ? Ok(payload)
                : new HttpResponseMessage(HttpStatusCode.ServiceUnavailable));
        var provider = new DownloadingModelProvider(
            "multi-url.onnx",
            new[] { "http://example.com/primary", "http://example.com/secondary" },
            cacheDir,
            NullLogger.Instance,
            httpClient: new HttpClient(handler));

        var result = await provider.GetModelPathAsync();

        Assert.Equal(Path.Combine(cacheDir, "multi-url.onnx"), result);
        Assert.Equal(Payload, File.ReadAllText(result));
        AssertNoTempFiles(cacheDir);
    }

    // --- W3: best-effort temp-file cleanup when the cache directory becomes unwritable ---

    [SkippableFact]
    public async Task GetModelPathAsync_WhenTempFileCannotBeDeleted_DownloadStillFailsCleanly()
    {
        Skip.IfNot(PosixInterop.IsLinux, "Requires POSIX chmod to make the cache directory unwritable.");
        Skip.If(PosixInterop.RunningAsRoot, "chmod does not restrict root.");

        var payload = Encoding.UTF8.GetBytes(Payload);
        var cacheDir = Path.Combine(_testDir, "undeletable");
        Directory.CreateDirectory(cacheDir);

        // The response stream hands over the full payload, then locks the cache directory to
        // read-only before throwing: the download fails, and the subsequent temp-file delete
        // inside DeleteFileBestEffort must ALSO fail (and be swallowed) rather than propagate.
        var handler = new StubHandler(_ => OkResponse(new LockdownStreamContent(cacheDir, payload)));
        var provider = new DownloadingModelProvider(
            "undeletable.onnx",
            new[] { "http://example.com/undeletable.onnx" },
            cacheDir,
            NullLogger.Instance,
            httpClient: new HttpClient(handler));

        try
        {
            var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => provider.GetModelPathAsync());
            Assert.Contains("simulated network failure", ex.Message, StringComparison.Ordinal);

            // The best-effort delete failed (directory is read-only) and was swallowed: the
            // caller still sees the download failure, not an UnauthorizedAccessException.
            Assert.False(File.Exists(Path.Combine(cacheDir, "undeletable.onnx")), "final file must not exist");
        }
        finally
        {
            // Restore writability so the fixture's Directory.Delete cleanup can run.
            PosixInterop.TryChmod(cacheDir, PosixInterop.OwnerReadWriteExecute);
        }
    }

    /// <summary>
    /// Content whose read stream hands over the payload once, then flips the cache directory to
    /// read-only and throws: the temp file written next to it becomes undeletable.
    /// </summary>
    private sealed class LockdownStreamContent : HttpContent
    {
        private readonly string _directoryToLock;
        private readonly byte[] _payload;

        public LockdownStreamContent(string directoryToLock, byte[] payload)
        {
            _directoryToLock = directoryToLock;
            _payload = payload;
        }

        protected override Task<Stream> CreateContentReadStreamAsync(CancellationToken cancellationToken)
            => Task.FromResult<Stream>(new LockdownStream(_directoryToLock, _payload));

        protected override Task<Stream> CreateContentReadStreamAsync()
            => Task.FromResult<Stream>(new LockdownStream(_directoryToLock, _payload));

        protected override async Task SerializeToStreamAsync(Stream stream, TransportContext? context)
        {
            var s = new LockdownStream(_directoryToLock, _payload);
            var buffer = new byte[81920];
            int read;
            while ((read = await s.ReadAsync(buffer)) > 0)
                await stream.WriteAsync(buffer.AsMemory(0, read));
        }

        protected override bool TryComputeLength(out long length)
        {
            length = _payload.Length;
            return true;
        }

        private sealed class LockdownStream : Stream
        {
            private readonly string _directoryToLock;
            private readonly byte[] _payload;
            private bool _served;
            private bool _broken;

            public LockdownStream(string directoryToLock, byte[] payload)
            {
                _directoryToLock = directoryToLock;
                _payload = payload;
            }

            public override bool CanRead => !_broken;
            public override bool CanSeek => false;
            public override bool CanWrite => false;
            public override long Length => throw new NotSupportedException();
            public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

            public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
            {
                if (!_served)
                {
                    _served = true;
                    var toCopy = Math.Min(buffer.Length, _payload.Length);
                    _payload.AsMemory(0, toCopy).CopyTo(buffer);
                    return toCopy;
                }

                if (!_broken)
                {
                    _broken = true;
                    // The provider has written the partial temp file by now; lock the directory
                    // down so the best-effort delete after this failure cannot succeed.
                    PosixInterop.TryChmod(_directoryToLock, PosixInterop.OwnerReadOnlyDir);
                    await Task.Yield();
                    throw new IOException("simulated network failure mid-stream");
                }

                return 0;
            }

            public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
            public override void Flush() => throw new NotSupportedException();
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        }
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
