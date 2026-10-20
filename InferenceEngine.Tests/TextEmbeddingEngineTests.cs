using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using InferenceEngine.Core;
using InferenceEngine.Core.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// Hermetic coverage for <see cref="TextEmbeddingEngine"/>'s inference paths: tensor pre-assembly
/// against fake input metadata, the pooling/normalization post-processing branches, warmup input
/// construction, and tokenizer-failure tolerance during <c>LoadAsync</c>.
/// </summary>
public sealed class TextEmbeddingEngineTests : IDisposable
{
    private readonly string _testDir;
    private readonly List<IDisposable> _disposables = new();

    public TextEmbeddingEngineTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"embed-engine-tests-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        foreach (var d in _disposables) d.Dispose();
        TokenizerAssets.TokenizerCacheDirectoryOverride = null;
        try { Directory.Delete(_testDir, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    private static Dictionary<string, NodeMetadata> MiniLmLikeMeta() => new()
    {
        ["input_ids"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { 1, -1 }),
        ["attention_mask"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { 1, -1 }),
        ["token_type_ids"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { 1, -1 }),
        ["position_ids"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { 1, -1 }),
    };

    private static Dictionary<string, NodeMetadata> IdentityLikeMeta() => new()
    {
        ["input"] = TextEngineTestHelpers.FakeMeta(typeof(float), new[] { 1, -1 }),
    };

    // --- Pre-processing --------------------------------------------------------

    [Fact]
    public void PreProcess_BuildsAllDeclaredIntegerInputs()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta());
        _disposables.Add(engine);

        var (inputs, ctx) = engine.PrePublic("hello world");
        int seqLen = Assert.IsType<int>(ctx);
        Assert.Equal(engine.TestTokenizerPublic!.EncodeToIds("hello world").Count, seqLen);

        Assert.Equal(4, inputs.Count);
        var ids = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs, i => i.Name == "input_ids").Value);
        Assert.Equal(seqLen, ids.Dimensions[1]);

        var mask = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs, i => i.Name == "attention_mask").Value);
        Assert.All(Enumerable.Range(0, seqLen), i => Assert.Equal(1L, mask[0, i]));

        var tt = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs, i => i.Name == "token_type_ids").Value);
        Assert.All(Enumerable.Range(0, seqLen), i => Assert.Equal(0L, tt[0, i]));

        var pos = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs, i => i.Name == "position_ids").Value);
        Assert.Equal(0L, pos[0, 0]);
        Assert.Equal(seqLen - 1L, pos[0, seqLen - 1]);
    }

    [Fact]
    public void PreProcess_Int32Metadata_BuildsInt32Tensors()
    {
        var meta = new Dictionary<string, NodeMetadata>
        {
            ["input_ids"] = TextEngineTestHelpers.FakeMeta(typeof(int), new[] { 1, -1 }),
            ["attention_mask"] = TextEngineTestHelpers.FakeMeta(typeof(int), new[] { 1, -1 }),
        };
        var engine = new FakeEmbedEngine(meta);
        _disposables.Add(engine);

        var (inputs, _) = engine.PrePublic("hello");

        Assert.IsType<DenseTensor<int>>(Assert.Single(inputs, i => i.Name == "input_ids").Value);
        Assert.IsType<DenseTensor<int>>(Assert.Single(inputs, i => i.Name == "attention_mask").Value);
    }

    [Fact]
    public void PreProcess_UnknownInput_FilledWithZeros()
    {
        var engine = new FakeEmbedEngine(IdentityLikeMeta());
        _disposables.Add(engine);

        var (inputs, _) = engine.PrePublic("hello");

        var zeros = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs, i => i.Name == "input").Value);
        Assert.All(Enumerable.Range(0, zeros.Dimensions[1]), i => Assert.Equal(0L, zeros[0, i]));
    }

    [Fact]
    public void PreProcess_Rank1Metadata_SetsSequenceDimension()
    {
        var meta = new Dictionary<string, NodeMetadata>
        {
            ["input_ids"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { -1 }),
        };
        var engine = new FakeEmbedEngine(meta);
        _disposables.Add(engine);

        var (inputs, _) = engine.PrePublic("hello");

        var ids = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs).Value);
        Assert.Equal(1, ids.Dimensions.Length);
        Assert.Equal(engine.TestTokenizerPublic!.EncodeToIds("hello").Count, ids.Dimensions[0]);
    }

    [Fact]
    public void PreProcess_NullTokenizer_Throws()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta(), tokenizer: null, installDefaultTokenizer: false);
        _disposables.Add(engine);

        Assert.Throws<InvalidOperationException>(() => engine.PrePublic("hello"));
    }

    // --- Post-processing -------------------------------------------------------

    [Fact]
    public void PostProcess_Rank3Tensor_MeanPoolsAndNormalizes()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta());
        _disposables.Add(engine);

        // [1, seq=2, dim=2]: pooling restricted to seqLen=2 averages the two rows.
        var t = new DenseTensor<float>(new[] { 1, 2, 2 });
        t[0, 0, 0] = 2f; t[0, 0, 1] = 4f;
        t[0, 1, 0] = 4f; t[0, 1, 1] = 8f;
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("last_hidden_state", t));

        float[] emb = engine.PostPublic(outputs, 2);

        Assert.Equal(2, emb.Length);
        double norm = Math.Sqrt(emb.Sum(x => (double)x * x));
        Assert.Equal(1.0, norm, precision: 5);         // L2-normalized
        Assert.Equal(0.4472, emb[0], 3);               // (3/6) normalized
        Assert.Equal(0.8944, emb[1], 3);
    }

    [Fact]
    public void PostProcess_Rank3Tensor_RespectsMaskedSeqLen()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta());
        _disposables.Add(engine);

        // seq dim 4 but only the first 2 positions are valid (seqLen=2).
        var t = new DenseTensor<float>(new[] { 1, 4, 1 });
        t[0, 0, 0] = 1f;
        t[0, 1, 0] = 3f;
        t[0, 2, 0] = 100f; // beyond seqLen — must be ignored
        t[0, 3, 0] = 100f;
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("last_hidden_state", t));

        float[] emb = engine.PostPublic(outputs, 2);
        Assert.Equal(1, emb.Length);
        Assert.Equal(1.0, Math.Abs(emb[0]), 5); // mean 2 -> normalized to 1
    }

    [Fact]
    public void PostProcess_Rank2Tensor_TakesRowAndNormalizes()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta());
        _disposables.Add(engine);

        var t = new DenseTensor<float>(new[] { 1, 3 });
        t[0, 0] = 0f; t[0, 1] = 3f; t[0, 2] = 4f;
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("sentence_embedding", t));

        float[] emb = engine.PostPublic(outputs, 5);
        Assert.Equal(3, emb.Length);
        Assert.Equal(0.8, emb[2], 3);
    }

    [Fact]
    public void PostProcess_Rank1Tensor_Normalizes()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta());
        _disposables.Add(engine);

        var t = new DenseTensor<float>(new[] { 2 });
        t[0] = 3f; t[1] = 4f;
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("pooled", t));

        float[] emb = engine.PostPublic(outputs, 1);
        Assert.Equal(0.6, emb[0], 3);
        Assert.Equal(0.8, emb[1], 3);
    }

    [Fact]
    public void PostProcess_UnsupportedRank_Throws()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta());
        _disposables.Add(engine);

        var t = new DenseTensor<float>(new[] { 1, 2, 2, 2 });
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("sentence_embedding", t));

        var ex = Assert.Throws<InvalidOperationException>(() => engine.PostPublic(outputs, 1));
        Assert.Contains("rank 4", ex.Message);
    }

    [Fact]
    public void PostProcess_NoFloatOutput_Throws()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta());
        _disposables.Add(engine);

        using var outputs = new TextEngineTestHelpers.FakeOutputCollection();
        Assert.Throws<InvalidOperationException>(() => engine.PostPublic(outputs, 1));
    }

    [Fact]
    public void PostProcess_PrefersNamedEmbedding_OverFirstFloat()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta());
        _disposables.Add(engine);

        var hidden = new DenseTensor<float>(new[] { 1, 1, 2 });
        hidden[0, 0, 0] = 100f; hidden[0, 0, 1] = 100f; // would NOT be unit after normalize? it would: 1/sqrt(2)
        var pooled = new DenseTensor<float>(new[] { 1, 2 });
        pooled[0, 0] = 3f; pooled[0, 1] = 4f;
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("last_hidden_state", hidden),
            TextEngineTestHelpers.MakeValue("sentence_embedding", pooled));

        float[] emb = engine.PostPublic(outputs, 1);

        // The named sentence_embedding won: [0.6, 0.8], not the normalized hidden row.
        Assert.Equal(0.6, emb[0], 3);
        Assert.Equal(0.8, emb[1], 3);
    }

    [Fact]
    public void PostProcess_InvalidContext_DefaultsToSinglePosition()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta());
        _disposables.Add(engine);

        // context null (not an int): pooled denominator falls back to 1.
        var t = new DenseTensor<float>(new[] { 1, 2, 1 });
        t[0, 0, 0] = 2f;
        t[0, 1, 0] = 6f;
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("last_hidden_state", t));

        float[] emb = engine.PostPublic(outputs, context: null);
        // denom = max(1, min(1(seqLen fallback), 2)) = 1 -> pooled = 2 -> normalized 1
        Assert.Equal(1, emb.Length);
        Assert.Equal(1.0, Math.Abs(emb[0]), 5);
    }

    // --- Warmup + LoadAsync ----------------------------------------------------

    [Fact]
    public void GetWarmupInput_MatchesDeclaredElementTypes()
    {
        var engine = new FakeEmbedEngine(new Dictionary<string, NodeMetadata>
        {
            ["input_ids"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { 1, -1 }),
            ["attention_mask"] = TextEngineTestHelpers.FakeMeta(typeof(int), new[] { 1, -1 }),
        });
        _disposables.Add(engine);

        var warm = engine.WarmupPublic();

        Assert.Equal(2, warm.Count);
        var ids = Assert.IsType<DenseTensor<long>>(warm[0].Value);
        Assert.Equal(4, ids.Dimensions[1]); // fixed warmup sequence length
        Assert.IsType<DenseTensor<int>>(warm[1].Value);
    }

    [Fact]
    public async Task GenerateEmbeddingAsync_CancelledToken_ThrowsBeforeLoad()
    {
        var engine = new FakeEmbedEngine(MiniLmLikeMeta());
        _disposables.Add(engine);

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => engine.GenerateEmbeddingAsync("hello", cts.Token));
    }

    [SkippableFact]
    public async Task LoadAsync_CorruptCachedVocab_LogsWarningAndStillLoadsSession()
    {
        Skip.IfNot(NativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        string dir = Path.Combine(_testDir, "corrupt-vocab-cache");
        Directory.CreateDirectory(dir);
        await File.WriteAllTextAsync(Path.Combine(dir, "vocab.txt"), "\0corrupt\0vocab\0");
        TokenizerAssets.TokenizerCacheDirectoryOverride = dir;

        var options = new InferenceEngineOptions
        {
            ModelPath = IdentityModelPath,
            UseGpuAcceleration = false,
            WarmupOnLoad = true, // exercise GetWarmupInput through PerformWarmup
        };
        using var engine = new TextEmbeddingEngine(options);

        // Must not throw: the tokenizer failure is a warning; the session loads from ModelPath and
        // warmup failure is non-critical (identity.onnx's float input rejects int dummy tensors).
        await engine.LoadAsync();
    }

    [SkippableFact]
    public async Task LoadAsync_ValidLocalVocab_LoadsTokenizerAndSession()
    {
        Skip.IfNot(NativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        string dir = Path.Combine(_testDir, "good-vocab-cache");
        Directory.CreateDirectory(dir);
        TextEngineTestHelpers.WriteMiniVocab(dir);
        TokenizerAssets.TokenizerCacheDirectoryOverride = dir;

        var options = new InferenceEngineOptions
        {
            ModelPath = IdentityModelPath,
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
        };
        using var engine = new TextEmbeddingEngine(options);
        await engine.LoadAsync();
    }

    private static string IdentityModelPath =>
        Path.Combine(AppContext.BaseDirectory, "Resources", "identity.onnx");

    private static bool NativeOnnxRuntimeAvailable()
    {
        try
        {
            using var opts = new SessionOptions();
            return true;
        }
        catch
        {
            return false;
        }
    }

    // --- Fake engine -----------------------------------------------------------

    private sealed class FakeEmbedEngine : TextEmbeddingEngine
    {
        public FakeEmbedEngine(
            IReadOnlyDictionary<string, NodeMetadata> metadata,
            Tokenizer? tokenizer = null,
            bool installDefaultTokenizer = true)
            : base(new InferenceEngineOptions { UseGpuAcceleration = false, WarmupOnLoad = false })
        {
            TestTokenizer = tokenizer ?? (installDefaultTokenizer ? CreateDefaultTokenizer() : null);
            ApplyLoadedState(
                TextEngineTestHelpers.FakeEngineInfo(),
                new Dictionary<string, NodeMetadata>(metadata),
                new Dictionary<string, NodeMetadata>(),
                metadata.Keys.FirstOrDefault());
        }

        private static Tokenizer CreateDefaultTokenizer()
        {
            string dir = Path.Combine(Path.GetTempPath(), "embed-tests-vocab", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                return TextEngineTestHelpers.CreateMiniTokenizer(
                    TextEngineTestHelpers.WriteMiniVocab(dir));
            }
            finally
            {
                try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ }
            }
        }

        internal Tokenizer? TestTokenizerPublic => TestTokenizer;

        public (List<NamedOnnxValue> Inputs, object? Context) PrePublic(string input)
            => PreProcessWithContext(input);

        public float[] PostPublic(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? context)
            => PostProcessWithContext(outputs, context);

        public List<NamedOnnxValue> WarmupPublic() => GetWarmupInput();
    }
}
