using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace InferenceEngine.Core.Text;

/// <summary>
/// One-shot text embedding engine backed by the all-MiniLM-L6-v2 ONNX model. Owns BERT
/// tokenization end-to-end: callers pass plain strings and receive an L2-normalized embedding
/// vector (<c>string in -> float[] out</c>).
/// </summary>
/// <remarks>
/// <para>
/// The embedding is produced by masked mean-pooling over the sequence dimension of the model's
/// primary output (typically <c>last_hidden_state</c> / <c>sentence_embedding</c>), using the
/// attention mask, followed by L2 normalization. The vector dimension is read from the model's
/// output metadata and is never hard-coded.
/// </para>
/// <para>
/// Tokenization uses an uncased BERT WordPiece tokenizer (<c>Microsoft.ML.Tokenizers</c>), the
/// vocab of which is downloaded and cached via <see cref="TokenizerAssets.LoadMiniLmTokenizerAsync"/>.
/// If the tokenizer asset is unavailable the engine still constructs (the embedding call will then
/// throw a clear error), so the instance is always safe to create.
/// </para>
/// </remarks>
public class TextEmbeddingEngine : BaseInferenceEngine<string, float[]>
{
    /// <summary>MiniLM embedding model ONNX download URL (single self-contained file).</summary>
    internal const string MiniLmModelUrl =
        "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";

    private Tokenizer? _tokenizer;

    /// <summary>
    /// Test seam (used by <c>InferenceEngine.Tests</c>): injects a pre-built tokenizer without
    /// network access, mirroring the seam on <see cref="TextGenerationEngine"/>. Null (default) =
    /// the tokenizer is resolved lazily during <see cref="LoadAsync"/>.
    /// </summary>
    internal Tokenizer? TestTokenizer
    {
        get => _tokenizer;
        set => _tokenizer = value;
    }

    /// <summary>
    /// Creates the engine. The ONNX model and tokenizer are resolved lazily during
    /// <see cref="LoadAsync"/>.
    /// </summary>
    /// <param name="options">Engine options. GPU acceleration defaults on; pass <c>UseGpuAcceleration = false</c> for CPU.</param>
    /// <param name="logger">Optional typed logger.</param>
    public TextEmbeddingEngine(InferenceEngineOptions options, ILogger<TextEmbeddingEngine>? logger = null)
        : base(options, logger)
    {
    }

    /// <summary>
    /// Loads the MiniLM tokenizer (best-effort) and then initializes the ONNX session. A tokenizer
    /// load failure is logged as a warning and does not abort the session load; the first embedding
    /// call will surface a clear error if a tokenizer is genuinely required.
    /// </summary>
    public override async Task LoadAsync()
    {
        if (_tokenizer == null)
        {
            try
            {
                _tokenizer = await TokenizerAssets.LoadMiniLmTokenizerAsync(_options, _logger, CancellationToken.None)
                    .ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not load the MiniLM tokenizer during LoadAsync.");
                _tokenizer = null;
            }
        }

        await base.LoadAsync().ConfigureAwait(false);
    }

    /// <summary>The model file name used for cache keys and registry lookups.</summary>
    protected override string ModelName => "all-MiniLM-L6-v2.onnx";

    /// <summary>Download URLs for the all-MiniLM-L6-v2 ONNX model.</summary>
    protected override string[]? ModelDownloadUrls => new[] { MiniLmModelUrl };

    /// <summary>
    /// Tokenizes the input, then assembles the model's declared integer inputs
    /// (<c>input_ids</c>, <c>attention_mask</c>, and <c>token_type_ids</c>/<c>position_ids</c>
    /// when the model declares them). Each tensor is typed to match its declared element type
    /// (<see cref="long"/> or <see cref="int"/>).
    /// </summary>
    /// <returns>The input tensors and the sequence length (carried as context for pooling).</returns>
    protected override (List<NamedOnnxValue> InputTensors, object? Context) PreProcessWithContext(string input)
    {
        var tokenizer = _tokenizer ?? throw new InvalidOperationException(
            "Tokenizer is not loaded. Ensure network access so the MiniLM vocab can be fetched.");

        var ids = tokenizer.EncodeToIds(input ?? string.Empty);
        if (ids.Count == 0)
        {
            ids = new[] { 0 };
        }

        int seqLen = ids.Count;
        var inputs = new List<NamedOnnxValue>();
        var metadata = InputMetadata ?? throw new InvalidOperationException("Engine is not loaded (no input metadata).");

        foreach (var (name, nodeMeta) in metadata)
        {
            var shape = nodeMeta.Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
            // Force sequence dim to the actual token length for [1, seq] inputs.
            if (shape.Length >= 2) shape[1] = seqLen;
            if (shape.Length == 1) shape[0] = seqLen;
            Type elementType = nodeMeta.ElementType ?? typeof(long);

            if (name.Contains("input_ids", StringComparison.OrdinalIgnoreCase))
            {
                inputs.Add(CreateIntTensor(name, ids, shape, elementType));
            }
            else if (name.Contains("attention_mask", StringComparison.OrdinalIgnoreCase))
            {
                inputs.Add(CreateIntTensor(name, Enumerable.Repeat(1, seqLen), shape, elementType));
            }
            else if (name.Contains("token_type_ids", StringComparison.OrdinalIgnoreCase))
            {
                inputs.Add(CreateIntTensor(name, Enumerable.Repeat(0, seqLen), shape, elementType));
            }
            else if (name.Contains("position_ids", StringComparison.OrdinalIgnoreCase))
            {
                inputs.Add(CreateIntTensor(name, Enumerable.Range(0, seqLen), shape, elementType));
            }
            else
            {
                // Unknown integer/other input: fill zeros with the matching declared type so the
                // forward pass has a valid binding. Fall back to int64 if not determinable.
                inputs.Add(CreateIntTensor(name, Enumerable.Repeat(0, seqLen), shape, elementType));
            }
        }

        return (inputs, seqLen);
    }

    /// <summary>
    /// Selects the embedding output (preferring <c>sentence_embedding</c>/<c>pooler</c>/<c>pooled</c>),
    /// masked-mean-pools a 3D [1,seq,dim] tensor (or takes the row of a 2D [1,dim] tensor), and
    /// L2-normalizes the result.
    /// </summary>
    /// <param name="output">Raw ONNX outputs.</param>
    /// <param name="context">Sequence length carried from pre-processing.</param>
    /// <returns>An L2-normalized embedding vector.</returns>
    protected override float[] PostProcessWithContext(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output,
        object? context)
    {
        int seqLen = context is int sl && sl > 0 ? sl : 1;
        DenseTensor<float>? tensor = null;

        foreach (var item in output)
        {
            if (item.Value is not DenseTensor<float> f) continue;
            string n = item.Name ?? string.Empty;
            if (n.Contains("sentence_embedding", StringComparison.OrdinalIgnoreCase) ||
                n.Contains("pooled", StringComparison.OrdinalIgnoreCase) ||
                n.Contains("pooler", StringComparison.OrdinalIgnoreCase))
            {
                tensor = f;
                break;
            }

            tensor ??= f;
        }

        if (tensor is null)
        {
            throw new InvalidOperationException("No float output tensor was produced by the embedding model.");
        }

        int dim;
        if (tensor.Dimensions.Length == 3)
        {
            dim = tensor.Dimensions[2];
            var pooled = new float[dim];
            for (int s = 0; s < seqLen && s < tensor.Dimensions[1]; s++)
            {
                for (int d = 0; d < dim; d++)
                {
                    pooled[d] += tensor[0, s, d];
                }
            }
            // masked mean-pool over the valid sequence positions
            float denom = Math.Max(1, Math.Min(seqLen, tensor.Dimensions[1]));
            for (int d = 0; d < dim; d++) pooled[d] /= denom;
            return L2Normalize(pooled);
        }

        if (tensor.Dimensions.Length == 2)
        {
            dim = tensor.Dimensions[1];
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = tensor[0, d];
            return L2Normalize(row);
        }

        if (tensor.Dimensions.Length == 1)
        {
            dim = tensor.Dimensions[0];
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = tensor[d];
            return L2Normalize(row);
        }

        throw new InvalidOperationException(
            $"Unsupported embedding output rank {tensor.Dimensions.Length} (expected 1, 2, or 3).");
    }

    /// <summary>
    /// Builds correctly-typed integer dummy tensors (int64/int32 per declared element type) so the
    /// base warmup pass does not fail on default float-zero tensors.
    /// </summary>
    protected override List<NamedOnnxValue> GetWarmupInput()
    {
        var inputs = new List<NamedOnnxValue>();
        if (InputMetadata is null) return inputs;

        const int seq = 4;
        foreach (var (name, nodeMeta) in InputMetadata)
        {
            var shape = nodeMeta.Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
            if (shape.Length >= 2) shape[1] = seq;
            if (shape.Length == 1) shape[0] = seq;
            Type elementType = nodeMeta.ElementType ?? typeof(long);

            if (elementType == typeof(int))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<int>(shape)));
            }
            else
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(shape)));
            }
        }

        return inputs;
    }

    /// <summary>
    /// Convenience helper that loads the engine and returns the embedding for a single string.
    /// </summary>
    public async Task<float[]> GenerateEmbeddingAsync(string text, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        await LoadAsync().ConfigureAwait(false);
        ct.ThrowIfCancellationRequested();
        return await PredictAsync(text).ConfigureAwait(false);
    }

    /// <summary>
    /// L2-normalizes a vector in place semantics: returns a new normalized vector. If the norm is
    /// zero the input is returned unchanged (no NaN).
    /// </summary>
    /// <param name="vector">The vector to normalize.</param>
    /// <returns>An L2-normalized copy of <paramref name="vector"/>.</returns>
    internal static float[] L2Normalize(ReadOnlySpan<float> vector)
    {
        float sum = 0f;
        for (int i = 0; i < vector.Length; i++) sum += vector[i] * vector[i];
        double norm = Math.Sqrt(sum);
        var result = new float[vector.Length];
        if (norm <= 0d)
        {
            vector.CopyTo(result);
            return result;
        }

        float scale = (float)(1d / norm);
        for (int i = 0; i < vector.Length; i++) result[i] = vector[i] * scale;
        return result;
    }

    /// <summary>
    /// Convenience overload that normalizes a writable vector's contents.
    /// </summary>
    internal static float[] L2Normalize(float[] vector) => L2Normalize(vector.AsSpan());

    /// <summary>
    /// Masked mean-pool of a 3D [1, seq, dim] tensor over the sequence axis, restricted to the first
    /// <paramref name="seqLen"/> positions.
    /// </summary>
    /// <param name="tensor">The [1, seq, dim] hidden-state tensor.</param>
    /// <param name="seqLen">Number of valid (masked) sequence positions.</param>
    /// <returns>The pooled [dim] vector.</returns>
    internal static float[] MeanPool(DenseTensor<float> tensor, int seqLen)
    {
        if (tensor.Dimensions.Length != 3)
        {
            throw new ArgumentException("MeanPool expects a rank-3 [1,seq,dim] tensor.", nameof(tensor));
        }

        int dim = tensor.Dimensions[2];
        int maxSeq = tensor.Dimensions[1];
        int valid = Math.Max(1, Math.Min(seqLen, maxSeq));
        var pooled = new float[dim];
        for (int s = 0; s < valid; s++)
        {
            for (int d = 0; d < dim; d++)
            {
                pooled[d] += tensor[0, s, d];
            }
        }

        for (int d = 0; d < dim; d++) pooled[d] /= valid;
        return pooled;
    }

    /// <summary>Cosine similarity between two equal-length vectors (assumed non-degenerate).</summary>
    internal static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length || a.Length == 0) return 0f;
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }

        double denom = Math.Sqrt(na) * Math.Sqrt(nb);
        return denom <= 0d ? 0f : (float)(dot / denom);
    }

    /// <summary>Builds an integer-valued tensor matching <paramref name="elementType"/> (long or int).</summary>
    private static NamedOnnxValue CreateIntTensor(string name, IEnumerable<int> values, int[] shape, Type elementType)
    {
        var list = values as int[] ?? values.ToArray();
        if (elementType == typeof(int))
        {
            var tensor = new DenseTensor<int>(shape);
            var span = tensor.Buffer.Span;
            int count = Math.Min(span.Length, list.Length);
            for (int i = 0; i < count; i++)
            {
                span[i] = list[i];
            }

            return NamedOnnxValue.CreateFromTensor(name, tensor);
        }

        var longTensor = new DenseTensor<long>(shape);
        var longSpan = longTensor.Buffer.Span;
        int longCount = Math.Min(longSpan.Length, list.Length);
        for (int i = 0; i < longCount; i++)
        {
            longSpan[i] = list[i];
        }

        return NamedOnnxValue.CreateFromTensor(name, longTensor);
    }
}
