# Text inference engines

`InferenceEngine.Core.Text` adds two string-in/string-out engines that own tokenization and model
asset fetching end-to-end, so consuming apps need no tokenizer code of their own. The dependency
`Microsoft.ML.Tokenizers` (1.0.3) provides the tokenizers.

## TextEmbeddingEngine

`TextEmbeddingEngine : BaseInferenceEngine<string, float[]>` produces L2-normalized embeddings from
the **all-MiniLM-L6-v2** ONNX model.

- Tokenizer: uncased BERT WordPiece (`BertTokenizer`, `LowerCaseBeforeTokenization = true`), vocab
  fetched from `Xenova/all-MiniLM-L6-v2/vocab.txt`.
- Inputs are assembled from the model's declared input metadata (`input_ids`, `attention_mask`, and
  `token_type_ids`/`position_ids` when present), each typed to match its declared element type
  (`int64` or `int32`).
- Output: the embedding tensor (`sentence_embedding`/`pooler`/`pooled`, else the first output) is
  masked-mean-pooled over the sequence axis (for 3D `[1,seq,dim]`) and L2-normalized. The vector
  dimension is read from metadata, never hard-coded.

```csharp
using InferenceEngine.Core;
using InferenceEngine.Core.Text;

var options = new InferenceEngineOptions { UseGpuAcceleration = false };
using var engine = new TextEmbeddingEngine(options);
await engine.LoadAsync();
float[] emb = await engine.PredictAsync("docker container");
```

The deterministic math (`L2Normalize`, `MeanPool`, `CosineSimilarity`) is exposed as `internal static`
helpers so it can be unit-tested without a model.

## TextGenerationEngine

`TextGenerationEngine : BaseStreamingInferenceEngine<string, string, string>` streams text from the
**Phi-3-mini-4k-instruct** ONNX export (cpu-int4). `TStep` is a decoded text chunk.

- Tokenizer: SentencePiece model read via `LlamaTokenizer.Create` (the public SentencePiece-backed
  factory in Tokenizers 1.0.3; see the note below), `addBeginOfSentence = addEndOfSentence = false`,
  special tokens `<|endoftext|>`=32000, `<|im_start|>`=32001, `<|im_end|>`=32002.
- Default `StreamingOptions`: greedy, `MaxSteps = 256`, EOS `{ 32000, 32002 }`.

### Decode strategy: full-recompute per step

Each decode step re-feeds the **entire growing token-id list** (`input_ids`, `attention_mask = ones`,
`position_ids = 0..n-1`) and runs a fresh forward pass. The last position's logit row is greedy-sampled,
the sampled id is appended to the state, and the full id list is re-detokenized; only the **new text
suffix** since the previous yield is emitted (correctly handling SentencePiece subword merges).

This is model-agnostic and robust across ONNX export variants (which differ in KV-cache I/O
names/shapes), at the cost of **O(n²)** compute. A KV-cache optimization that reuses `present_*` /
`past_*` tensors is a planned follow-up.

```csharp
var options = new InferenceEngineOptions { UseGpuAcceleration = false };
using var engine = new TextGenerationEngine(options);
await foreach (string chunk in engine.PredictStreamingAsync("What is 2 + 2?"))
{
    Console.Write(chunk);
}
```

### Phi-3 `.onnx.data` limitation

The cpu-int4 export is split across a graph file (`...int4-rtn-block-32.onnx`) and an external-data
file (`...onnx.data`). `ModelDownloadUrls` references the `.onnx` **only**; an `InferenceSession`
created from that single file will fail to initialize without its sibling `.onnx.data`. For real runs,
point `InferenceEngineOptions.ModelPath` at a complete local bundle with both files co-located.

### Tokenizer factory note

`Microsoft.ML.Tokenizers` 1.0.3 exposes no public factory on `SentencePieceTokenizer` itself. The
public factory that reads a SentencePiece model and returns a `SentencePieceTokenizer`-derived
instance is `LlamaTokenizer.Create(Stream, bool, bool, IReadOnlyDictionary<string,int>)`, which is
what `TokenizerAssets.LoadPhi3TokenizerAsync` uses. Phi-3's `tokenizer.model` is a SentencePiece
model.

## Tokenizer assets

`TokenizerAssets` (internal) downloads/caches tokenizer vocab and model files via
`DownloadingModelProvider` under the tokenizers cache directory
(`{LocalApplicationData}/InferenceEngine/Tokenizers`), then constructs the configured `Tokenizer` over
a read-only file stream. Both engines load their tokenizer best-effort in the constructor; if the
asset is unavailable at construction (e.g. offline), the engine still constructs and surfaces a clear
error on the first inference call.
