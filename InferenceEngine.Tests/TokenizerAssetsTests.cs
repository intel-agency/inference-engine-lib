using System;
using System.IO;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using InferenceEngine.Core;
using InferenceEngine.Core.Text;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// Hermetic tests for <see cref="TokenizerAssets"/> failure and corrupt-asset paths. Uses the
/// internal cache-directory override plus a local loopback HTTP server (or connection-refused
/// ports) for failure injection — no external network is touched.
/// </summary>
public sealed class TokenizerAssetsTests : IDisposable
{
    private readonly string _testDir;

    public TokenizerAssetsTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"tokenizer-assets-tests-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testDir);
        TokenizerAssets.TokenizerCacheDirectoryOverride = _testDir;
    }

    public void Dispose()
    {
        TokenizerAssets.TokenizerCacheDirectoryOverride = null;
        try { Directory.Delete(_testDir, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    [Fact]
    public async Task DownloadAssetAsync_AllSourcesFail_ThrowsWrappedInvalidOperationException()
    {
        // Local loopback server that answers 500 to every request: the download itself completes at
        // the HTTP layer, and EnsureSuccessStatusCode fails. Deterministic, no external network.
        await using var server = new LoopbackServer();

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            TokenizerAssets.DownloadAssetAsync(
                "some-asset.bin",
                new[] { server.Url("asset") },
                new InferenceEngineOptions(),
                NullLogger.Instance,
                CancellationToken.None));

        Assert.Contains("some-asset.bin", ex.Message);
        Assert.Contains("could not be loaded or downloaded", ex.Message);
        Assert.NotNull(ex.InnerException);
    }

    [Fact]
    public async Task DownloadAssetAsync_ConnectionRefused_ThrowsWrappedInvalidOperationException()
    {
        // No listener on this loopback port: the TCP connect fails outright.
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            TokenizerAssets.DownloadAssetAsync(
                "missing-asset.bin",
                new[] { "http://127.0.0.1:9/never-listened.bin" },
                new InferenceEngineOptions(),
                NullLogger.Instance,
                CancellationToken.None));

        Assert.Contains("missing-asset.bin", ex.Message);
    }

    [Fact]
    public async Task DownloadAssetAsync_NullOptions_ThrowsArgumentNullException()
    {
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            TokenizerAssets.DownloadAssetAsync(
                "f.txt", new[] { "http://127.0.0.1:9/f.txt" },
                null!, NullLogger.Instance, CancellationToken.None));
    }

    [Fact]
    public async Task DownloadAssetAsync_CachedFile_ReturnsWithoutDownloading()
    {
        string cached = Path.Combine(_testDir, "cached.txt");
        File.WriteAllText(cached, "payload");

        string result = await TokenizerAssets.DownloadAssetAsync(
            "cached.txt",
            new[] { "http://127.0.0.1:9/never-reached.txt" },
            new InferenceEngineOptions(),
            NullLogger.Instance,
            CancellationToken.None);

        Assert.Equal(cached, result);
        Assert.Equal("payload", File.ReadAllText(result));
    }

    [Fact]
    public async Task LoadMiniLmTokenizerAsync_CorruptCachedVocab_Throws()
    {
        // A vocab.txt that exists but is not a valid BERT vocab: the download layer succeeds (cache
        // hit) and the corruption surfaces when BertTokenizer.Create parses the stream.
        File.WriteAllText(Path.Combine(_testDir, "vocab.txt"), "\0\0not a vocab\0\0");

        await Assert.ThrowsAnyAsync<Exception>(() =>
            TokenizerAssets.LoadMiniLmTokenizerAsync(
                new InferenceEngineOptions(), NullLogger.Instance, CancellationToken.None));
    }

    // --- loopback server helper ---

    /// <summary>Tiny loopback HTTP server that answers 500 to every request (failure injection).</summary>
    private sealed class LoopbackServer : IAsyncDisposable
    {
        private readonly HttpListener _listener;
        private readonly Task _serve;

        public int Port { get; }

        public LoopbackServer()
        {
            // Grab a free loopback port, then bind HttpListener to it.
            var l = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
            l.Start();
            Port = ((System.Net.IPEndPoint)l.LocalEndpoint).Port;
            l.Stop();

            _listener = new HttpListener();
            _listener.Prefixes.Add($"http://localhost:{Port}/");
            _listener.Start();
            _serve = Task.Run(async () =>
            {
                try
                {
                    var ctx = await _listener.GetContextAsync();
                    byte[] body = System.Text.Encoding.UTF8.GetBytes("err");
                    ctx.Response.StatusCode = 500;
                    ctx.Response.ContentLength64 = body.Length;
                    await ctx.Response.OutputStream.WriteAsync(body);
                    ctx.Response.Close();
                }
                catch (Exception)
                {
                    // listener disposed / connection reset — fine for failure injection
                }
            });
        }

        public string Url(string path) => $"http://localhost:{Port}/{path}";

        public async ValueTask DisposeAsync()
        {
            try
            {
                _listener.Abort();
            }
            catch (Exception) { }
            await Task.WhenAny(_serve, Task.Delay(500));
        }
    }
}
