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
/// Streaming text-generation engine for the Phi-3-mini-4k-instruct ONNX export. Owns SentencePiece
/// tokenization end-to-end: callers pass a prompt string and receive streamed text chunks
/// (<c>string in -> string out</c>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Decode strategy: full-recompute per step.</b> Each step re-feeds the entire growing token-id
/// sequence (plus <c>attention_mask</c> and <c>position_ids</c>) and runs a fresh forward pass. This
/// is model-agnostic and robust across ONNX export variants (which differ in their KV-cache I/O
/// names/shapes), at the cost of O(n²) compute. A KV-cache optimization is a follow-up.
/// </para>
/// <para>
/// <b>Phi-3 external-data note.</b> The cpu-int4 export is split across a graph file (<c>.onnx</c>)
/// and an external-data file (<c>.onnx.data</c>). <see cref="ModelDownloadUrls"/> references the
/// <c>.onnx</c> only; a session created from that single file will fail to initialize without its
/// sibling <c>.onnx.data</c>. For real runs, point <see cref="InferenceEngineOptions.ModelPath"/> at
/// a complete local bundle (both files co-located).
/// </para>
/// </remarks>
public class TextGenerationEngine : BaseStreamingInferenceEngine<string, string, string>
{
    /// <summary>Phi-3-mini-4k-instruct (cpu-int4) graph download URL.</summary>
    internal const string Phi3ModelUrl =
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32/phi3-mini-4k-instruct-cpu-int4-rtn-block-32.onnx";

    private Tokenizer? _tokenizer;

    /// <summary>
    /// Creates the engine. The ONNX model and tokenizer are resolved lazily during
    /// <see cref="LoadAsync"/>.
    /// </summary>
    /// <param name="options">Engine options. GPU acceleration defaults on; pass <c>UseGpuAcceleration = false</c> for CPU.</param>
    /// <param name="logger">Optional typed logger.</param>
    public TextGenerationEngine(InferenceEngineOptions options, ILogger<TextGenerationEngine>? logger = null)
        : base(options, logger)
    {
    }

    /// <summary>
    /// Loads the Phi-3 SentencePiece tokenizer (best-effort) and then initializes the ONNX session.
    /// A tokenizer load failure is logged as a warning and does not abort the session load; the
    /// first generation call will surface a clear error if a tokenizer is genuinely required.
    /// </summary>
    public override async Task LoadAsync()
    {
        if (_tokenizer == null)
        {
            try
            {
                _tokenizer = await TokenizerAssets.LoadPhi3TokenizerAsync(_options, _logger, CancellationToken.None)
                    .ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not load the Phi-3 tokenizer during LoadAsync.");
                _tokenizer = null;
            }
        }

        await base.LoadAsync().ConfigureAwait(false);
    }

    /// <summary>The model file name used for cache keys and registry lookups.</summary>
    protected override string ModelName => "phi-3-mini-4k-instruct.onnx";

    /// <summary>
    /// Download URL for the Phi-3 graph file. See the class remarks: this references the
    /// <c>.onnx</c> only; the co-located <c>.onnx.data</c> external-data file is required for a
    /// successful session but is not fetched here.
    /// </summary>
    protected override string[]? ModelDownloadUrls => new[] { Phi3ModelUrl };

    /// <summary>
    /// Default streaming options: greedy, capped at 256 steps, terminating on the Phi-3
    /// <c>&lt;|endoftext|&gt;</c> (32000) and <c>&lt;|im_end|&gt;</c> (32002) tokens.
    /// </summary>
    protected override StreamingOptions DefaultStreamingOptions => new()
    {
        MaxSteps = 256,
        EosTokenIds = new[] { 32000, 32002 },
    };

    /// <summary>
    /// Step 0 tokenizes the prompt and seeds a <see cref="GenerationState"/>. Steps &gt; 0 receive a
    /// state whose id list already includes the previously-sampled token; the full id list is rebuilt
    /// into <c>input_ids</c>, <c>attention_mask</c> (all-ones), and <c>position_ids</c>
    /// (0..n-1) — only for inputs the model actually declares.
    /// </summary>
    protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
        string input, object? state, int step)
    {
        var tokenizer = _tokenizer ?? throw new InvalidOperationException(
            "Tokenizer is not loaded. Ensure network access so the Phi-3 tokenizer.model can be fetched.");

        GenerationState gen;
        if (step == 0)
        {
            var ids = tokenizer.EncodeToIds(input ?? string.Empty);
            if (ids.Count == 0) ids = new[] { 0 };
            gen = new GenerationState(ids);
        }
        else
        {
            gen = (GenerationState)state!;
        }

        var inputs = BuildFullRecomputeInputs(gen.Ids);
        return (inputs, gen);
    }

    /// <summary>
    /// Reads the logits output, greedy-samples the last position, appends the sampled id to the
    /// state, detokenizes the full id list, and yields only the new text suffix since the last
    /// yield (correctly handling SentencePiece subword merges).
    /// </summary>
    protected override (string Step, object? NextState, bool IsEos) ProcessStepOutputs(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
    {
        var gen = (GenerationState)buildContext!;
        var tokenizer = _tokenizer ?? throw new InvalidOperationException("Tokenizer is not loaded.");

        DenseTensor<float>? logits = null;
        foreach (var item in outputs)
        {
            if (item.Value is not DenseTensor<float> f) continue;
            string n = item.Name ?? string.Empty;
            if (n.Contains("logits", StringComparison.OrdinalIgnoreCase))
            {
                logits = f;
                break;
            }

            logits ??= f;
        }

        if (logits is null)
        {
            throw new InvalidOperationException("No logits output was produced by the generation model.");
        }

        ReadOnlySpan<float> lastRow = GetLastPositionLogitRow(logits);
        int sampled = SampleNextToken(lastRow);

        bool isEos = CurrentOptions?.EosTokenIds is { Count: > 0 } eos && eos.Contains(sampled);

        // One-step-delayed delta decode: only emit text that is confirmed stable (the longest
        // common prefix of the previous full decode and the current full decode). This is robust
        // to SentencePiece byte-fallback, where appending one id can mutate a previously-emitted
        // replacement char (U+FFFD) — a plain substring-by-length slice would emit wrong/lost
        // chunks. On EOS the remaining (final) tail is flushed so string.Concat(stream) == Decode(all).
        string delta = AdvanceDecodeState(
            gen, sampled, isEos, ids => tokenizer.Decode(ids));

        return (delta, gen, isEos);
    }

    /// <summary>Reduces the streamed chunks into the final one-shot text.</summary>
    protected override string BuildFinalOutput(IReadOnlyList<string> steps) => string.Concat(steps);

    /// <summary>
    /// Builds the integer inputs for a full-recompute forward pass from the current id list, typed
    /// to match each input's declared element type (int64 or int32).
    /// </summary>
    private List<NamedOnnxValue> BuildFullRecomputeInputs(IReadOnlyList<int> ids)
    {
        var metadata = InputMetadata ?? throw new InvalidOperationException("Engine is not loaded (no input metadata).");
        int seqLen = ids.Count;
        var inputs = new List<NamedOnnxValue>();

        foreach (var (name, nodeMeta) in metadata)
        {
            var shape = nodeMeta.Dimensions.Select(d => d <= 0 ? 1 : d).ToArray();
            if (shape.Length >= 2) shape[1] = seqLen;
            if (shape.Length == 1) shape[0] = seqLen;
            Type elementType = nodeMeta.ElementType ?? typeof(long);

            IEnumerable<int> data;
            if (name.Contains("input_ids", StringComparison.OrdinalIgnoreCase))
            {
                data = ids;
            }
            else if (name.Contains("attention_mask", StringComparison.OrdinalIgnoreCase))
            {
                data = Enumerable.Repeat(1, seqLen);
            }
            else if (name.Contains("position_ids", StringComparison.OrdinalIgnoreCase))
            {
                data = Enumerable.Range(0, seqLen);
            }
            else
            {
                // KV-cache / unknown integer inputs are skipped under full-recompute; populating them
                // is unnecessary for forward-only passes and avoids mismatched past-key-value shapes.
                continue;
            }

            inputs.Add(CreateIntTensor(name, data, shape, elementType));
        }

        return inputs;
    }

    /// <summary>
    /// Extracts the vocab-length logit row for the last (most recent) token position from a logits
    /// tensor of arbitrary rank. For [*, vocab] the final dim is the vocabulary; the last row is the
    /// contiguous span ending at the tensor's end.
    /// </summary>
    private static ReadOnlySpan<float> GetLastPositionLogitRow(DenseTensor<float> logits)
    {
        var span = logits.Buffer.Span;
        if (logits.Dimensions.Length == 0 || span.IsEmpty)
        {
            throw new InvalidOperationException("Logits tensor is empty.");
        }

        int vocab = logits.Dimensions[^1];
        if (vocab <= 0)
        {
            throw new InvalidOperationException("Logits tensor has a non-positive vocabulary dimension.");
        }

        int total = (int)span.Length;
        if (total < vocab)
        {
            throw new InvalidOperationException(
                $"Logits tensor length ({total}) is less than vocabulary size ({vocab}).");
        }

        int start = total - vocab;
        return span.Slice(start, vocab);
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

    /// <summary>
    /// Advances the decode state by one sampled token and returns the text chunk to yield this
    /// step. Pure except for mutating <paramref name="gen"/>; the <paramref name="decode"/>
    /// callback lets tests drive it with a fake tokenizer that mutates the prefix.
    /// </summary>
    internal static string AdvanceDecodeState(
        GenerationState gen, int sampled, bool isEos, Func<IReadOnlyList<int>, string> decode)
    {
        gen.Ids.Add(sampled);
        string currFull = decode(gen.Ids);
        int confirmed = CommonPrefixLength(gen.PrevFull, currFull);

        string delta;
        if (isEos)
        {
            // Flush the final tail: everything not yet emitted.
            delta = currFull.Substring(gen.EmittedLen);
            gen.EmittedLen = currFull.Length;
        }
        else if (confirmed > gen.EmittedLen)
        {
            // Emit the newly-confirmed stable region (common prefix with the previous full decode).
            delta = currFull.Substring(gen.EmittedLen, confirmed - gen.EmittedLen);
            gen.EmittedLen = confirmed;
        }
        else
        {
            delta = string.Empty;
        }

        gen.PrevFull = currFull;
        return delta;
    }

    /// <summary>Length of the longest common leading character prefix of two strings.</summary>
    internal static int CommonPrefixLength(string a, string b)
    {
        int max = Math.Min(a.Length, b.Length);
        int i = 0;
        while (i < max && a[i] == b[i]) i++;
        return i;
    }

    /// <summary>Carry-forward decode state for full-recompute generation.</summary>
    internal sealed class GenerationState
    {
        public List<int> Ids { get; }
        /// <summary>Full decoded text after the previous step (Decode of ids[0..n-1]).</summary>
        public string PrevFull { get; set; } = string.Empty;
        /// <summary>Number of leading chars of the current full decode that have been emitted.</summary>
        public int EmittedLen { get; set; }

        public GenerationState(IReadOnlyList<int> ids)
        {
            Ids = new List<int>(ids);
        }
    }
}
