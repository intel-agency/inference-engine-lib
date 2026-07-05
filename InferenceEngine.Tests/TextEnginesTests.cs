using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using InferenceEngine.Core;
using InferenceEngine.Core.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace InferenceEngine.Tests;

public sealed class TextEnginesTests : IDisposable
{
    private readonly List<IDisposable> _disposables = new();

    /// <summary>
    /// Returns false on hosts where the OnnxRuntime native library cannot be loaded (e.g. macOS CI
    /// without the dylib), so the live embedding/generation tests skip instead of failing. Mirrors
    /// the helper used by <c>EngineInfoTests</c>.
    /// </summary>
    private static bool IsNativeOnnxRuntimeAvailable()
    {
        try
        {
            using var opts = new SessionOptions();
            return true;
        }
        catch (Exception ex) when (
            ex is DllNotFoundException
            || ex is TypeInitializationException
            || (ex is TargetInvocationException { InnerException: Exception inner }
                && (inner is DllNotFoundException or TypeInitializationException)))
        {
            return false;
        }
    }

    [Fact]
    public void L2Normalize_ProducesUnitVector()
    {
        float[] v = { 0f, 3f, 4f };
        float[] n = TextEmbeddingEngine.L2Normalize(v);

        double norm = Math.Sqrt(n.Sum(x => (double)x * x));
        Assert.Equal(1.0, norm, precision: 5);
        Assert.Equal(0f, n[0]);
        Assert.Equal(0.6f, n[1], precision: 4);
        Assert.Equal(0.8f, n[2], precision: 4);
    }

    [Fact]
    public void L2Normalize_ZeroVectorStaysZero()
    {
        float[] v = { 0f, 0f, 0f };
        float[] n = TextEmbeddingEngine.L2Normalize(v);
        Assert.Equal(v, n);
        Assert.True(n.All(x => !float.IsNaN(x)));
    }

    [Fact]
    public void MeanPool_AggregatesSequenceDimension()
    {
        // [1, seq=2, dim=2]
        var tensor = new DenseTensor<float>(new[] { 1, 2, 2 });
        tensor[0, 0, 0] = 2f; tensor[0, 0, 1] = 4f;
        tensor[0, 1, 0] = 4f; tensor[0, 1, 1] = 8f;

        float[] pooled = TextEmbeddingEngine.MeanPool(tensor, seqLen: 2);

        Assert.Equal(3f, pooled[0]);
        Assert.Equal(6f, pooled[1]);
    }

    [Fact]
    public void MeanPool_HonorsValidSequenceLength()
    {
        var tensor = new DenseTensor<float>(new[] { 1, 4, 1 });
        tensor[0, 0, 0] = 1f;
        tensor[0, 1, 0] = 3f;
        tensor[0, 2, 0] = 100f; // beyond seqLen, must be ignored
        tensor[0, 3, 0] = 100f;

        float[] pooled = TextEmbeddingEngine.MeanPool(tensor, seqLen: 2);
        Assert.Equal(2f, pooled[0]);
    }

    [Fact]
    public void CosineSimilarity_SimilarExceedsDissimilar()
    {
        var similar = TextEmbeddingEngine.L2Normalize(new float[] { 1f, 1f, 1f }.AsSpan());
        var near = TextEmbeddingEngine.L2Normalize(new float[] { 1f, 1f, 1.1f }.AsSpan());
        var dissimilar = TextEmbeddingEngine.L2Normalize(new float[] { -1f, -1f, -1f }.AsSpan());

        double cosNear = TextEmbeddingEngine.CosineSimilarity(similar, near);
        double cosFar = TextEmbeddingEngine.CosineSimilarity(similar, dissimilar);

        Assert.True(cosNear > cosFar, $"expected cosNear ({cosNear}) > cosFar ({cosFar})");
        Assert.True(cosNear > 0.99);
        Assert.True(cosFar < -0.99);
    }

    [Fact]
    public void EmbeddingMath_PooledSimilar_Exceeds_Dissimilar()
    {
        // Two synthetic [1,3,3] hidden states; one aligned, one anti-aligned.
        var a = new DenseTensor<float>(new[] { 1, 3, 3 });
        var b = new DenseTensor<float>(new[] { 1, 3, 3 });
        var c = new DenseTensor<float>(new[] { 1, 3, 3 });
        for (int s = 0; s < 3; s++)
        {
            a[0, s, 0] = 1f; a[0, s, 1] = 0f; a[0, s, 2] = 0f;
            b[0, s, 0] = 0.9f; b[0, s, 1] = 0.1f; b[0, s, 2] = 0f;
            c[0, s, 0] = -1f; c[0, s, 1] = 0f; c[0, s, 2] = 0f;
        }

        float[] ea = TextEmbeddingEngine.L2Normalize(TextEmbeddingEngine.MeanPool(a, 3));
        float[] eb = TextEmbeddingEngine.L2Normalize(TextEmbeddingEngine.MeanPool(b, 3));
        float[] ec = TextEmbeddingEngine.L2Normalize(TextEmbeddingEngine.MeanPool(c, 3));

        double cosAB = TextEmbeddingEngine.CosineSimilarity(ea, eb);
        double cosAC = TextEmbeddingEngine.CosineSimilarity(ea, ec);

        Assert.True(cosAB > cosAC, $"expected similar > dissimilar");
        Assert.True(cosAB > 0.9);
        Assert.True(cosAC < -0.9);
    }

    [Fact]
    public void TensorAssembly_BuildsInt64TensorsForMiniLmInputs()
    {
        var fakeMetadata = new Dictionary<string, NodeMetadata>
        {
            ["input_ids"] = FakeMeta(typeof(long), new[] { 1, -1 }),
            ["attention_mask"] = FakeMeta(typeof(long), new[] { 1, -1 }),
            ["token_type_ids"] = FakeMeta(typeof(long), new[] { 1, -1 }),
        };

        var engine = new TensorAssemblyProbe(fakeMetadata);
        var (inputs, seqLen) = engine.AssemblePublic(new[] { 101, 202, 303 });

        Assert.Equal(3, seqLen);
        Assert.Equal(3, inputs.Count);

        var ids = Assert.Single(inputs, i => i.Name == "input_ids");
        var idsTensor = Assert.IsType<DenseTensor<long>>(ids.Value);
        Assert.Equal(new[] { 1, 3 }, idsTensor.Dimensions.ToArray());
        Assert.Equal(101L, idsTensor[0, 0]);
        Assert.Equal(303L, idsTensor[0, 2]);

        var mask = Assert.Single(inputs, i => i.Name == "attention_mask");
        var maskTensor = Assert.IsType<DenseTensor<long>>(mask.Value);
        Assert.Equal(1L, maskTensor[0, 0]);
        Assert.Equal(1L, maskTensor[0, 2]);

        var tt = Assert.Single(inputs, i => i.Name == "token_type_ids");
        var ttTensor = Assert.IsType<DenseTensor<long>>(tt.Value);
        Assert.Equal(0L, ttTensor[0, 0]);
    }

    [Fact]
    public void TensorAssembly_RespectsInt32ElementType()
    {
        var fakeMetadata = new Dictionary<string, NodeMetadata>
        {
            ["input_ids"] = FakeMeta(typeof(int), new[] { 1, -1 }),
            ["attention_mask"] = FakeMeta(typeof(int), new[] { 1, -1 }),
        };

        var engine = new TensorAssemblyProbe(fakeMetadata);
        var (inputs, _) = engine.AssemblePublic(new[] { 7, 8, 9 });

        var ids = Assert.Single(inputs, i => i.Name == "input_ids");
        Assert.IsType<DenseTensor<int>>(ids.Value);
    }

    // T4 (H3): the one-step-delayed delta decode reconstructs the full one-shot decode exactly,
    // even when the tokenizer is NOT prefix-monotonic (SentencePiece byte-fallback: appending an id
    // mutates a previously-emitted U+FFFD). The old substring-by-length logic dropped/duplicated
    // text here. Driven purely through the internal AdvanceDecodeState with a fake decode function.
    [Fact]
    public void DecodeDelta_ReconstructsFullEvenWhenPrefixMutates()
    {
        string Decode(IReadOnlyList<int> ids) => ids.Count switch
        {
            1 => "a",
            2 => "a\uFFFD", // id #2 is a partial multi-byte sequence -> U+FFFD
            3 => "abc",     // id #3 completes it -> the U+FFFD resolves to "bc"
            4 => "abcd",
            _ => throw new InvalidOperationException("unexpected id count"),
        };

        int[] ids = { 101, 102, 103, 104 };
        var gen = new TextGenerationEngine.GenerationState(Array.Empty<int>());
        var chunks = new List<string>();
        for (int i = 0; i < ids.Length; i++)
        {
            bool isEos = i == ids.Length - 1;
            chunks.Add(TextGenerationEngine.AdvanceDecodeState(gen, ids[i], isEos, Decode));
        }

        string streamed = string.Concat(chunks);
        Assert.Equal(Decode(ids), streamed);        // one-shot == concat(stream)
        Assert.DoesNotContain("\uFFFD", streamed);  // no stale replacement char leaked
    }

    // T4 (H3): regression guard — a normal prefix-monotonic decode also reconstructs exactly.
    [Fact]
    public void DecodeDelta_ReconstructsMonotonicPrefix()
    {
        string Decode(IReadOnlyList<int> ids) => string.Concat(ids.Select(i => (char)('a' + (i % 26))));
        int[] ids = { 0, 1, 2, 3, 4 };

        var gen = new TextGenerationEngine.GenerationState(Array.Empty<int>());
        var chunks = new List<string>();
        for (int i = 0; i < ids.Length; i++)
        {
            bool isEos = i == ids.Length - 1;
            chunks.Add(TextGenerationEngine.AdvanceDecodeState(gen, ids[i], isEos, Decode));
        }

        Assert.Equal(Decode(ids), string.Concat(chunks));
    }

    [SkippableFact]
    public async Task TextEmbeddingEngine_LoadsAndEmbeds_WhenNetworkAvailable()
    {
        Skip.If(Environment.GetEnvironmentVariable("CI_SKIP_NETWORK") is not null,
            "Network-gated; CI_SKIP_NETWORK is set.");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var options = new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
        };

        TextEmbeddingEngine engine;
        try
        {
            engine = new TextEmbeddingEngine(options);
            _disposables.Add(engine);
        }
        catch (Exception ex)
        {
            throw new SkipException($"Could not construct engine (tokenizer fetch failed): {ex.Message}");
        }

        await engine.LoadAsync();

        float[] embA = await engine.PredictAsync("docker container");
        float[] embB = await engine.PredictAsync("docker containerization");
        float[] embC = await engine.PredictAsync("the weather is sunny today");

        Assert.NotNull(embA);
        Assert.NotEmpty(embA);
        Assert.Equal(embA.Length, embB.Length);

        double cosSim = TextEmbeddingEngine.CosineSimilarity(embA, embB);
        double cosDis = TextEmbeddingEngine.CosineSimilarity(embA, embC);
        Assert.True(cosSim > cosDis, "related texts must be more similar than unrelated");
    }

    [SkippableFact]
    public async Task TextGenerationEngine_PredictAsyncMatchesStreaming_WhenBundleAvailable()
    {
        Skip.If(Environment.GetEnvironmentVariable("CI_SKIP_NETWORK") is not null,
            "Network-gated; CI_SKIP_NETWORK is set.");
        Skip.IfNot(IsNativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        var options = new InferenceEngineOptions
        {
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
        };

        TextGenerationEngine engine;
        try
        {
            engine = new TextGenerationEngine(options);
            _disposables.Add(engine);
        }
        catch (Exception ex)
        {
            throw new SkipException($"Could not construct engine (tokenizer fetch failed): {ex.Message}");
        }

        try
        {
            await engine.LoadAsync();
        }
        catch (Exception ex)
        {
            throw new SkipException($"Phi-3 model bundle could not be loaded ({ex.Message}). " +
                                   "The cpu-int4 export requires both .onnx and .onnx.data; " +
                                   "set InferenceEngineOptions.ModelPath to a complete bundle.");
        }

        string oneShot = await engine.PredictAsync("What is 2 + 2?");
        var streamed = new List<string>();
        await foreach (var chunk in engine.PredictStreamingAsync("What is 2 + 2?"))
        {
            streamed.Add(chunk);
        }

        Assert.False(string.IsNullOrEmpty(oneShot), "one-shot generation should be non-empty");
        Assert.Equal(string.Concat(streamed), oneShot);
    }

    public void Dispose()
    {
        foreach (var d in _disposables) d.Dispose();
    }

    // Construct a NodeMetadata reflectively. NodeMetadata and TensorTypeAndShape expose only private
    // constructors in OnnxRuntime 1.24.1, so build them via reflection to fake a session's metadata
    // (element type + dimensions) without loading a real ONNX model.
    private static NodeMetadata FakeMeta(Type elementType, int[] dims)
    {
        var ttsCtor = typeof(TensorTypeAndShape).GetConstructor(
            BindingFlags.Instance | BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(TensorElementType), typeof(int[]), typeof(string[]) },
            modifiers: null)!;
        TensorElementType tet = elementType == typeof(int)
            ? TensorElementType.Int32
            : TensorElementType.Int64;
        var ts = (TensorTypeAndShape)ttsCtor.Invoke(new object[] { tet, dims, new string[dims.Length] });

        var nmCtor = typeof(NodeMetadata).GetConstructor(
            BindingFlags.Instance | BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(OnnxValueType), typeof(TensorTypeAndShape) },
            modifiers: null)!;
        return (NodeMetadata)nmCtor.Invoke(new object[] { OnnxValueType.ONNX_TYPE_TENSOR, ts });
    }

    /// <summary>
    /// Test-only probe that exposes the embedding pre-processing tensor assembly against an
    /// arbitrary metadata dictionary, without requiring a real ONNX session.
    /// </summary>
    private sealed class TensorAssemblyProbe
    {
        private readonly IReadOnlyDictionary<string, NodeMetadata> _metadata;

        public TensorAssemblyProbe(IReadOnlyDictionary<string, NodeMetadata> metadata)
        {
            _metadata = metadata;
        }

        public (List<NamedOnnxValue> Inputs, int SeqLen) AssemblePublic(int[] ids)
        {
            int seqLen = ids.Length;
            var inputs = new List<NamedOnnxValue>();
            foreach (var (name, nodeMeta) in _metadata)
            {
                var shape = nodeMeta.Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
                if (shape.Length >= 2) shape[1] = seqLen;
                if (shape.Length == 1) shape[0] = seqLen;
                Type elementType = nodeMeta.ElementType ?? typeof(long);

                IEnumerable<int> data = name.Contains("input_ids", StringComparison.OrdinalIgnoreCase) ? ids
                    : name.Contains("attention_mask", StringComparison.OrdinalIgnoreCase) ? Enumerable.Repeat(1, seqLen)
                    : Enumerable.Repeat(0, seqLen);

                inputs.Add(MakeTensor(name, data, shape, elementType));
            }

            return (inputs, seqLen);
        }

        private static NamedOnnxValue MakeTensor(string name, IEnumerable<int> values, int[] shape, Type elementType)
        {
            var list = values as int[] ?? values.ToArray();
            if (elementType == typeof(int))
            {
                var t = new DenseTensor<int>(shape);
                var span = t.Buffer.Span;
                for (int i = 0; i < Math.Min(span.Length, list.Length); i++) span[i] = list[i];
                return NamedOnnxValue.CreateFromTensor(name, t);
            }

            var lt = new DenseTensor<long>(shape);
            var lspan = lt.Buffer.Span;
            for (int i = 0; i < Math.Min(lspan.Length, list.Length); i++) lspan[i] = list[i];
            return NamedOnnxValue.CreateFromTensor(name, lt);
        }
    }
}
