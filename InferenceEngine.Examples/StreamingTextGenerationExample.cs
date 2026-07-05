using System;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using InferenceEngine.Core;

namespace InferenceEngine.Examples
{
    /// <summary>
    /// Reference skeleton for a streaming text-generation engine (e.g. an HF-optimum decoder
    /// export such as GPT-2 / OPT / Llama) built on
    /// <see cref="BaseStreamingInferenceEngine{TInput, TOutput, TStep}"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a <b>reference skeleton</b>: it demonstrates the prefill/decode tensor assembly,
    /// KV-cache carry-back, sampling, detokenization, and EOS-detection shape that a real
    /// streaming LLM engine requires, but it ships <b>no ONNX model and no tokenizer</b>.
    /// Tokenization is a consumer concern and is intentionally out of scope (see streaming spec §1).
    /// Provide a real HF-optimum ONNX export plus a tokenizer before calling
    /// <see cref="IInferenceEngine{TInput, TOutput}.LoadAsync"/>.
    /// </para>
    /// <para>
    /// Design reference: <c>plan_docs/inference-engine-streaming-upstream-spec.md</c>.
    /// </para>
    /// </remarks>
    public class StreamingTextGenerationExample
        : BaseStreamingInferenceEngine<string, string, string>
    {
        /// <summary>
        /// Creates the skeleton engine, forwarding options + logger to the base.
        /// </summary>
        /// <param name="options">Engine options (model path/provider, GPU, warmup).</param>
        /// <param name="logger">Optional logger.</param>
        public StreamingTextGenerationExample(
            InferenceEngineOptions options,
            ILogger<StreamingTextGenerationExample>? logger = null)
            : base(options, logger)
        {
        }

        /// <summary>Model file this engine expects. Not present in the default registry.</summary>
        protected override string ModelName => "streaming-text-generation.onnx";

        // Note: ModelDownloadUrls / ModelRegistry are not overridden — supply the model through
        // InferenceEngineOptions.ModelPath or InferenceEngineOptions.ModelProvider.

        /// <summary>
        /// Build the ONNX inputs for one forward pass and return the carry-forward context.
        /// <para>
        /// <paramref name="step"/> == 0 means prefill (encode the full prompt);
        /// step &gt; 0 means decode (feed the previous step's sampled token plus the carried
        /// KV cache back as <c>past_key_values</c>).
        /// </para>
        /// </summary>
        protected override (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepInputs(
            string input, object? state, int step)
        {
            // Discover the model's I/O names from session metadata (populated by LoadAsync).
            // A typical HF-optimum decoder export exposes: input_ids, attention_mask, position_ids,
            // and one present_key_values.N.{key,value} pair per layer:
            //
            //   string inputIdsName = PrimaryInputName ?? "input_ids";
            //   _ = InputMetadata;   // map: input_ids, attention_mask, position_ids, past_key_values.*
            //   _ = OutputMetadata;  // map: logits, present_key_values.*

            var inputs = new List<NamedOnnxValue>();

            if (step == 0)
            {
                // ---- Prefill: encode the full prompt. ----
                // TODO: tokenize `input` into int64 ids with your tokenizer
                // (SentencePiece / HuggingFace tokenizers / Microsoft.ML.Tokenizers).
                //
                //   long[] promptIds = Tokenizer.Encode(input);
                //   var idsTensor = new DenseTensor<long>(promptIds, new[] { 1, promptIds.Length });
                //   inputs.Add(NamedOnnxValue.CreateFromTensor(inputIdsName, idsTensor));
                //   inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask", Ones(1, promptIds.Length)));
                //   inputs.Add(NamedOnnxValue.CreateFromTensor("position_ids", Range(1, promptIds.Length)));
                //
                // Prefill feeds no past_key_values on step 0.
            }
            else
            {
                // ---- Decode: feed the single previously-sampled token + carried KV cache. ----
                //   var carry = (KvCacheState)state!;
                //
                //   var idsTensor = new DenseTensor<long>(new[] { carry.LastToken }, new[] { 1, 1 });
                //   inputs.Add(NamedOnnxValue.CreateFromTensor(inputIdsName, idsTensor));
                //
                // KV-cache carry-back: replay the previous step's present_* outputs as past_* inputs.
                // The carried tensors were cloned in ProcessStepOutputs via CloneAsPastKeyValue:
                //
                //   foreach (var kv in carry.PastKeyValues)
                //       inputs.Add(kv);
            }

            // No bundled model/tokenizer ships with this skeleton — guard so it never runs
            // half-configured. Supply a real model before calling LoadAsync.
            _ = inputs;
            throw new NotSupportedException(
                "Reference skeleton — provide a real HF-optimum ONNX export and a tokenizer. " +
                "See plan_docs/inference-engine-streaming-upstream-spec.md.");
        }

        /// <summary>
        /// Decode one forward pass into the streamed text chunk, the next KV-cache state, and an
        /// end-of-sequence flag. The engine owns token-level EOS detection against
        /// <see cref="StreamingOptions.EosTokenIds"/>; the base enforces the returned
        /// <c>IsEos</c> flag and <see cref="StreamingOptions.MaxSteps"/>.
        /// </summary>
        protected override (string Step, object? NextState, bool IsEos) ProcessStepOutputs(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
        {
            // TODO: locate the logits output (shape [batch, seq, vocab]) by name from OutputMetadata,
            // then read the next-token distribution from the last row:
            //
            //   var logitsOut = outputs.First(o => o.Name == "logits");
            //   var logitsTensor = (DenseTensor<float>)logitsOut.Value;          // [1, seqLen, vocab]
            //   int vocabSize = logitsTensor.Dimensions[2];
            //   var vocabRow = logitsTensor.Buffer.Span.Slice(vocabSize);          // last row
            //
            // Sample the next token id (greedy argmax via the base helper; override
            // SampleNextToken for top-k/top-p/temperature honoring CurrentOptions.Sampling):
            //
            //   int nextToken = SampleNextToken(vocabRow);
            //
            // Detokenize the single id into the text chunk to yield (a consumer concern; spec §1):
            //
            //   string chunk = Tokenizer.Decode(new[] { nextToken });

            // KV-cache carry-back: clone each present_*.N output into the matching past_*.N input
            // so it survives disposal of this output collection, then stash it on the next state:
            //
            //   var carried = new List<NamedOnnxValue>();
            //   foreach (var present in outputs.Where(o => o.Name.StartsWith("present", StringComparison.Ordinal)))
            //   {
            //       string pastName = present.Name.Replace("present", "past", StringComparison.Ordinal);
            //       var past = CloneAsPastKeyValue(present, pastName);
            //       if (past is not null) carried.Add(past);
            //   }
            //   object? nextState = new KvCacheState { LastToken = nextToken };
            //
            // EOS detection (the engine holds the sampled token):
            //
            //   bool isEos = CurrentOptions?.EosTokenIds.Contains(nextToken) ?? false;
            //   return (chunk, nextState, isEos);

            throw new NotSupportedException(
                "Reference skeleton — provide a real model and tokenizer to decode steps. " +
                "See plan_docs/inference-engine-streaming-upstream-spec.md.");
        }

        /// <summary>
        /// Reduce the accumulated streamed chunks into the final one-shot text.
        /// </summary>
        protected override string BuildFinalOutput(IReadOnlyList<string> steps)
            => string.Concat(steps);

        /// <summary>
        /// Carry-forward state for the autoregressive decode loop: the last sampled token id and
        /// the cloned KV-cache tensors replayed as the next step's <c>past_key_values</c>.
        /// </summary>
        private sealed class KvCacheState
        {
            /// <summary>The token id sampled in the previous step.</summary>
            public int LastToken { get; set; }

            /// <summary>Cloned present_* outputs renamed to past_* for the next forward pass.</summary>
            public List<NamedOnnxValue> PastKeyValues { get; } = new();
        }
    }
}
