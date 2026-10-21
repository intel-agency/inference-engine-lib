using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;
using InferenceEngine.Core.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// Hermetic coverage for <see cref="TextGenerationEngine"/>'s streaming/generate paths. A test
/// subclass fakes the loaded session via <c>ApplyLoadedState</c> (reflective metadata) and the
/// forward pass via a <c>RunStep</c> override returning crafted logits tensors, so the full
/// tokenize → build inputs → sample → delta-decode loop runs without a real Phi-3 model.
/// </summary>
public sealed class TextGenerationEngineTests : IDisposable
{
    private const int Vocab = 16;
    private const int HelloId = 5;
    private const int WorldId = 6;
    private const int SepId = 3; // [SEP] doubles as the EOS token in these tests

    private readonly string _testDir;
    private readonly List<IDisposable> _disposables = new();

    public TextGenerationEngineTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"gen-engine-tests-{Guid.NewGuid()}");
        Directory.CreateDirectory(_testDir);
    }

    public void Dispose()
    {
        foreach (var d in _disposables) d.Dispose();
        try { Directory.Delete(_testDir, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    private Tokenizer NewTokenizer() => TextEngineTestHelpers.CreateMiniTokenizer(
        TextEngineTestHelpers.WriteMiniVocab(_testDir));

    private static Dictionary<string, NodeMetadata> StandardMeta() => new()
    {
        ["input_ids"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { 1, -1 }),
        ["attention_mask"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { 1, -1 }),
        ["position_ids"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { 1, -1 }),
        // KV-cache-style input must be skipped by the full-recompute input builder.
        ["past_key_values.0.key"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { 1, -1 }),
    };

    // --- Full streaming loop -------------------------------------------------------

    [Fact]
    public async Task GenerateLoop_StreamsOneShotParity_AndStopsOnEos()
    {
        // Argmax schedule: after the "hello world" prompt, sample hello, world, then EOS. A
        // generator (not a queue) so the one-shot re-run after the stream can consume more steps.
        var sampled = new[] { HelloId, WorldId, SepId };
        int i = 0;
        Func<int> schedule = () => sampled[Math.Min(i++, sampled.Length - 1)];
        var tokenizer = NewTokenizer();
        var engine = new FakeGenEngine(StandardMeta(), tokenizer, schedule);
        _disposables.Add(engine);

        var options = new StreamingOptions { MaxSteps = 10, EosTokenIds = new[] { SepId } };

        var streamed = new List<string>();
        await foreach (var chunk in engine.PredictStreamingAsync("hello world", options))
            streamed.Add(chunk);

        // The one-shot path uses DefaultStreamingOptions (Phi-3 EOS ids 32000/32002), so its
        // schedule terminates with the Phi-3 end-of-text id instead of [SEP].
        int j = 0;
        var oneShotSchedule = new[] { HelloId, WorldId, 32000 };
        Func<int> s2 = () => oneShotSchedule[Math.Min(j++, oneShotSchedule.Length - 1)];
        var oneShotEngine = new FakeGenEngine(StandardMeta(), NewTokenizer(), s2, vocab: 32003);
        _disposables.Add(oneShotEngine);
        string oneShot = await oneShotEngine.PredictAsync("hello world");

        int[] promptIds = tokenizer.EncodeToIds("hello world").ToArray();
        int[] allIds = promptIds.Concat(new[] { HelloId, WorldId, SepId }).ToArray();
        string expected = tokenizer.Decode(allIds);
        string expectedOneShot = tokenizer.Decode(promptIds.Concat(new[] { HelloId, WorldId, 32000 }).ToArray());

        Assert.Equal(expected, string.Concat(streamed));
        Assert.Equal(expectedOneShot, oneShot);
        Assert.Equal(3, streamed.Count);
    }

    [Fact]
    public async Task GenerateLoop_MaxSteps_CapsGeneration()
    {
        // EOS is never sampled; the loop must stop at MaxSteps with the tokenizer still applied.
        var engine = new FakeGenEngine(StandardMeta(), NewTokenizer(), () => HelloId);
        _disposables.Add(engine);

        var options = new StreamingOptions { MaxSteps = 4, EosTokenIds = new[] { SepId } };

        var streamed = new List<string>();
        await foreach (var chunk in engine.PredictStreamingAsync("hello world", options))
            streamed.Add(chunk);

        Assert.Equal(4, streamed.Count);
    }

    // --- Input assembly (BuildStepInputs / BuildFullRecomputeInputs / CreateIntTensor) ---

    [Fact]
    public void BuildStepInputs_Int64Metadata_BuildsExpectedTensors()
    {
        var engine = new FakeGenEngine(StandardMeta(), NewTokenizer(), () => HelloId);
        _disposables.Add(engine);

        var (inputs, ctx) = engine.BuildStepPublic("hello world", null, step: 0);

        var gen = Assert.IsType<TextGenerationEngine.GenerationState>(ctx);
        int[] expectedIds = engine.TestTokenizer!.EncodeToIds("hello world").ToArray();
        Assert.Equal(expectedIds, gen.Ids);

        // input_ids / attention_mask / position_ids present; past_key_values skipped.
        Assert.Equal(3, inputs.Count);
        var ids = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs, i => i.Name == "input_ids").Value);
        Assert.Equal(expectedIds.Length, ids.Dimensions[1]);
        Assert.Equal(expectedIds[0], ids[0, 0]);
        Assert.Equal(expectedIds[^1], ids[0, expectedIds.Length - 1]);

        var mask = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs, i => i.Name == "attention_mask").Value);
        Assert.All(Enumerable.Range(0, expectedIds.Length), i => Assert.Equal(1L, mask[0, i]));

        var pos = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs, i => i.Name == "position_ids").Value);
        Assert.Equal(0L, pos[0, 0]);
        Assert.Equal(expectedIds.Length - 1L, pos[0, expectedIds.Length - 1]);

        Assert.DoesNotContain(inputs, i => i.Name.Contains("past_key_values"));
    }

    [Fact]
    public void BuildStepInputs_Int32Metadata_BuildsInt32Tensors()
    {
        var meta = new Dictionary<string, NodeMetadata>
        {
            ["input_ids"] = TextEngineTestHelpers.FakeMeta(typeof(int), new[] { 1, -1 }),
            ["attention_mask"] = TextEngineTestHelpers.FakeMeta(typeof(int), new[] { 1, -1 }),
        };
        var engine = new FakeGenEngine(meta, NewTokenizer(), () => HelloId);
        _disposables.Add(engine);

        var (inputs, _) = engine.BuildStepPublic("hello", null, step: 0);

        Assert.IsType<DenseTensor<int>>(Assert.Single(inputs, i => i.Name == "input_ids").Value);
        Assert.IsType<DenseTensor<int>>(Assert.Single(inputs, i => i.Name == "attention_mask").Value);
    }

    [Fact]
    public void BuildStepInputs_Rank1Metadata_SetsSequenceDimension()
    {
        var meta = new Dictionary<string, NodeMetadata>
        {
            ["input_ids"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { -1 }),
        };
        var engine = new FakeGenEngine(meta, NewTokenizer(), () => HelloId);
        _disposables.Add(engine);

        var (inputs, _) = engine.BuildStepPublic("hello", null, step: 0);

        var ids = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs).Value);
        Assert.Equal(1, ids.Dimensions.Length);
        Assert.Equal(engine.TestTokenizer!.EncodeToIds("hello").Count, ids.Dimensions[0]);
    }

    [Fact]
    public void BuildStepInputs_PositiveDims_AreKeptForNonSequenceAxes()
    {
        // A fixed batch dim of 2 (not dynamic) must be preserved; only the seq dim is overwritten.
        var meta = new Dictionary<string, NodeMetadata>
        {
            ["input_ids"] = TextEngineTestHelpers.FakeMeta(typeof(long), new[] { 2, 64 }),
        };
        var engine = new FakeGenEngine(meta, NewTokenizer(), () => HelloId);
        _disposables.Add(engine);

        var (inputs, _) = engine.BuildStepPublic("hello world", null, step: 0);

        var ids = Assert.IsType<DenseTensor<long>>(Assert.Single(inputs).Value);
        // seq replaced 64 -> token count ("hello world" -> [hello, world]); batch dim of 2 kept.
        Assert.Equal(new[] { 2, 2 }, ids.Dimensions.ToArray());
    }

    [Fact]
    public void BuildStepInputs_StepGreaterThanZero_ReusesCarriedState()
    {
        var engine = new FakeGenEngine(StandardMeta(), NewTokenizer(), () => HelloId);
        _disposables.Add(engine);

        var gen = new TextGenerationEngine.GenerationState(new[] { 1, 2, 3 });
        var (_, ctx) = engine.BuildStepPublic("ignored", gen, step: 2);

        Assert.Same(gen, ctx);
    }

    [Fact]
    public void BuildStepInputs_NullTokenizer_Throws()
    {
        var engine = new FakeGenEngine(StandardMeta(), tokenizer: null, () => HelloId);
        _disposables.Add(engine);

        Assert.Throws<InvalidOperationException>(() => engine.BuildStepPublic("hi", null, 0));
    }

    // --- Output processing (ProcessStepOutputs / GetLastPositionLogitRow) ---

    [Fact]
    public void ProcessStepOutputs_PrefersNamedLogits_OverEarlierFloat()
    {
        var engine = new FakeGenEngine(StandardMeta(), NewTokenizer(), () => HelloId);
        _disposables.Add(engine);
        var gen = new TextGenerationEngine.GenerationState(new[] { 2 });

        var hidden = TextEngineTestHelpers.LogitsTensor(seq: 1, vocab: Vocab, sampledId: HelloId);
        var logits = TextEngineTestHelpers.LogitsTensor(seq: 1, vocab: Vocab, sampledId: WorldId);
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("last_hidden_state", hidden),
            TextEngineTestHelpers.MakeValue("logits", logits));

        var (delta, next, isEos) = engine.ProcessStepPublic(outputs, gen);

        Assert.False(isEos);
        Assert.Same(gen, next);
        // The named "logits" tensor (argmax = world) won over the earlier "last_hidden_state"
        // (argmax = hello): world was the sampled id. The first post-prefill step confirms no
        // stable text yet, so the delta is empty.
        Assert.Equal(WorldId, gen.Ids[^1]);
        Assert.Equal(string.Empty, delta);
    }

    [Fact]
    public void ProcessStepOutputs_FallsBackToFirstFloat_WhenNoNamedLogits()
    {
        var engine = new FakeGenEngine(StandardMeta(), NewTokenizer(), () => HelloId);
        _disposables.Add(engine);
        var gen = new TextGenerationEngine.GenerationState(new[] { 2 });

        var hidden = TextEngineTestHelpers.LogitsTensor(seq: 1, vocab: Vocab, sampledId: HelloId);
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("last_hidden_state", hidden));

        var (delta, _, isEos) = engine.ProcessStepPublic(outputs, gen);
        Assert.False(isEos);
        // Fallback tensor (argmax = hello) was used, and no text is confirmed on the first step.
        Assert.Equal(HelloId, gen.Ids[^1]);
        Assert.Equal(string.Empty, delta);
    }

    [Fact]
    public void ProcessStepOutputs_SkipsNonFloatTensors()
    {
        var engine = new FakeGenEngine(StandardMeta(), NewTokenizer(), () => HelloId);
        _disposables.Add(engine);

        // Only an int tensor named "logits": skipped by the float type check -> "no logits" throw.
        var intTensor = new DenseTensor<int>(new[] { 1, 1, Vocab });
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("logits", intTensor));

        Assert.Throws<InvalidOperationException>(() => engine.ProcessStepPublic(outputs, new TextGenerationEngine.GenerationState(new[] { 2 })));
    }

    [Fact]
    public void ProcessStepOutputs_NoFloatOutput_Throws()
    {
        var engine = new FakeGenEngine(StandardMeta(), NewTokenizer(), () => HelloId);
        _disposables.Add(engine);

        using var outputs = new TextEngineTestHelpers.FakeOutputCollection();

        Assert.Throws<InvalidOperationException>(() =>
            engine.ProcessStepPublic(outputs, new TextGenerationEngine.GenerationState(new[] { 2 })));
    }

    [Fact]
    public void ProcessStepOutputs_EmptyLogitsTensor_Throws()
    {
        var engine = new FakeGenEngine(StandardMeta(), NewTokenizer(), () => HelloId);
        _disposables.Add(engine);

        var empty = new DenseTensor<float>(new[] { 0 });
        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("logits", empty));

        Assert.Throws<InvalidOperationException>(() =>
            engine.ProcessStepPublic(outputs, new TextGenerationEngine.GenerationState(new[] { 2 })));
    }

    [Fact]
    public void ProcessStepOutputs_NullTokenizer_Throws()
    {
        var engine = new FakeGenEngine(StandardMeta(), tokenizer: null, () => HelloId);
        _disposables.Add(engine);

        using var outputs = new TextEngineTestHelpers.FakeOutputCollection(
            TextEngineTestHelpers.MakeValue("logits", TextEngineTestHelpers.LogitsTensor(1, Vocab, HelloId)));

        Assert.Throws<InvalidOperationException>(() =>
            engine.ProcessStepPublic(outputs, new TextGenerationEngine.GenerationState(new[] { 2 })));
    }

    [Fact]
    public void AdvanceDecodeState_EosStep_FlushesRemainingTail()
    {
        var gen = new TextGenerationEngine.GenerationState(Array.Empty<int>());
        gen.PrevFull = "hello";
        gen.EmittedLen = 2;

        // On EOS the final tail (everything past the already-emitted prefix) is flushed so that
        // string.Concat(stream) == Decode(all ids).
        string delta = TextGenerationEngine.AdvanceDecodeState(gen, 6, isEos: true, _ => "hello world");

        Assert.Equal("llo world", delta);
        Assert.Equal(11, gen.EmittedLen);
    }

    // --- Delta decode helpers ------------------------------------------------------

    [Fact]
    public void CommonPrefixLength_MixedCases()
    {
        Assert.Equal(3, TextGenerationEngine.CommonPrefixLength("abcX", "abcY"));
        Assert.Equal(0, TextGenerationEngine.CommonPrefixLength("", "abc"));
        Assert.Equal(0, TextGenerationEngine.CommonPrefixLength("abc", ""));
        Assert.Equal(3, TextGenerationEngine.CommonPrefixLength("abc", "abc"));
    }

    [Fact]
    public void AdvanceDecodeState_ShrinkingDecode_ThrowsDescriptiveError()
    {
        // A decode that shrinks below the already-emitted length is a corrupt tokenizer state;
        // the guard must fail with the descriptive error instead of an ArgumentOutOfRange.
        var gen = new TextGenerationEngine.GenerationState(Array.Empty<int>());
        gen.PrevFull = "abcdef";
        gen.EmittedLen = 6;

        var ex = Assert.Throws<InvalidOperationException>(() =>
            TextGenerationEngine.AdvanceDecodeState(gen, 1, isEos: false, _ => "abc"));

        Assert.Contains("shrank", ex.Message);
    }

    [Fact]
    public void AdvanceDecodeState_NoNewConfirmedChars_YieldsEmpty()
    {
        var gen = new TextGenerationEngine.GenerationState(Array.Empty<int>());
        gen.PrevFull = "ab";
        gen.EmittedLen = 2;

        // Growing decode ("abcd") but the common prefix (2) does not exceed EmittedLen (2).
        string delta = TextGenerationEngine.AdvanceDecodeState(gen, 3, isEos: false, _ => "abcd");
        Assert.Equal(string.Empty, delta);
        Assert.Equal(2, gen.EmittedLen);
        Assert.Equal("abcd", gen.PrevFull);
    }

    // --- LoadAsync behavior --------------------------------------------------------

    [SkippableFact]
    public async Task LoadAsync_CorruptCachedTokenizer_LogsWarningAndStillLoadsSession()
    {
        Skip.IfNot(NativeOnnxRuntimeAvailable(), "ONNX Runtime native library is not available.");

        // Point the tokenizer cache at a directory whose tokenizer.model is corrupt: the Phi-3
        // tokenizer load fails, is logged as a warning, and the ONNX session still initializes from
        // the local ModelPath.
        string dir = Path.Combine(_testDir, "corrupt-cache");
        Directory.CreateDirectory(dir);
        await File.WriteAllTextAsync(Path.Combine(dir, "tokenizer.model"), "not-a-sentencepiece-proto");
        var options = new InferenceEngineOptions
        {
            ModelPath = IdentityModelPath,
            UseGpuAcceleration = false,
            WarmupOnLoad = false,
        };
        using var engine = new TextGenerationEngine(options);

        // Must not throw despite the tokenizer failure; the load runs under the shared
        // cache-override gate (the seam is process-global and test classes run in parallel).
        await TextEngineTestHelpers.WithCacheOverrideAsync(dir, () => engine.LoadAsync());
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

    // --- Fake engine ---------------------------------------------------------------

    /// <summary>
    /// A TextGenerationEngine with a faked loaded state (reflective input metadata, no real session)
    /// and a faked forward pass: RunStep returns a logits tensor whose last-row argmax follows a
    /// caller-supplied schedule.
    /// </summary>
    private sealed class FakeGenEngine : TextGenerationEngine
    {
        private readonly Func<int> _argmaxSchedule;
        private readonly int _vocab;

        public FakeGenEngine(
            IReadOnlyDictionary<string, NodeMetadata> metadata,
            Tokenizer? tokenizer,
            Func<int> argmaxSchedule,
            int vocab = Vocab)
            : base(new InferenceEngineOptions { UseGpuAcceleration = false, WarmupOnLoad = false })
        {
            TestTokenizer = tokenizer;
            _argmaxSchedule = argmaxSchedule;
            _vocab = vocab;
            ApplyLoadedState(
                TextEngineTestHelpers.FakeEngineInfo(),
                new Dictionary<string, NodeMetadata>(metadata),
                new Dictionary<string, NodeMetadata>(),
                metadata.Keys.FirstOrDefault());
        }

        protected override IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunStep(
            IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            int seq = inputs.Count > 0
                ? (inputs.First(i => i.Name.Contains("input_ids")).Value as DenseTensor<long>)!.Dimensions[1]
                : 1;
            int sampled = _argmaxSchedule();
            return new TextEngineTestHelpers.FakeOutputCollection(
                TextEngineTestHelpers.MakeValue(
                    "logits",
                    TextEngineTestHelpers.LogitsTensor(seq, _vocab, sampled)));
        }

        public (List<NamedOnnxValue> Inputs, object? BuildContext) BuildStepPublic(
            string input, object? state, int step)
            => BuildStepInputs(input, state, step);

        public (string Step, object? NextState, bool IsEos) ProcessStepPublic(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, object? buildContext)
            => ProcessStepOutputs(outputs, buildContext);
    }
}
