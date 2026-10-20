using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using InferenceEngine.Core;
using InferenceEngine.Core.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace InferenceEngine.Tests;

/// <summary>
 /// Shared helpers for the text-engine coverage tests: reflective <see cref="NodeMetadata"/>
 /// construction, fake <see cref="DisposableNamedOnnxValue"/> outputs, and a locally-built mini
 /// BERT vocab so the tokenizer-driven paths run hermetically (no network, no real Phi-3 asset).
 /// </summary>
internal static class TextEngineTestHelpers
{
    /// <summary>
    /// A minimal valid BERT vocab: the five standard special tokens followed by a few words.
    /// Line index == token id: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4, hello=5, world=6, c=7, d=8, o=9.
    /// </summary>
    public const string MiniVocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nc\nd\no\n";

    /// <summary>Builds a BertTokenizer over the mini vocab written to a temp file.</summary>
    public static Tokenizer CreateMiniTokenizer(string vocabFilePath)
    {
        using var stream = File.OpenRead(vocabFilePath);
        return BertTokenizer.Create(stream, new BertOptions { LowerCaseBeforeTokenization = true });
    }

    /// <summary>Writes the mini vocab and returns its path.</summary>
    public static string WriteMiniVocab(string directory)
    {
        string path = Path.Combine(directory, "vocab.txt");
        File.WriteAllText(path, MiniVocab);
        return path;
    }

    // NodeMetadata and TensorTypeAndShape expose only private constructors in OnnxRuntime 1.24.1,
    // so build them via reflection to fake a session's metadata (element type + dimensions) without
    // loading a real ONNX model. Mirrors the helper in TextEnginesTests.
    public static NodeMetadata FakeMeta(Type elementType, int[] dims)
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

    // The internal constructor's signature references IOrtValueOwner, which is itself internal to
    // the OnnxRuntime assembly, so the ctor cannot be referenced by type at compile time. Locate it
    // by its parameter types (string, object, TensorElementType, internal-owner).
    private static readonly ConstructorInfo? DisposableValueCtor = typeof(DisposableNamedOnnxValue)
        .GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic)
        .SingleOrDefault(c =>
        {
            var p = c.GetParameters();
            return p.Length == 4
                && p[0].ParameterType == typeof(string)
                && p[1].ParameterType == typeof(object)
                && p[2].ParameterType == typeof(TensorElementType);
        });

    /// <summary>
    /// Creates a DisposableNamedOnnxValue carrying a managed tensor. The internal constructor takes
    /// an OrtValue owner that is only used for native disposal; passing null leaves the managed
    /// tensor intact, which is all the text engines' post-processing inspects.
    /// </summary>
    public static DisposableNamedOnnxValue MakeValue(string name, object tensorValue)
    {
        if (DisposableValueCtor is null)
            throw new InvalidOperationException("DisposableNamedOnnxValue ctor not found.");
        return (DisposableNamedOnnxValue)DisposableValueCtor.Invoke(
            new object?[] { name, tensorValue, TensorElementType.Float, null });
    }

    /// <summary>A disposable output collection over caller-supplied values.</summary>
    public sealed class FakeOutputCollection : IDisposableReadOnlyCollection<DisposableNamedOnnxValue>
    {
        private readonly DisposableNamedOnnxValue[] _values;
        public bool Disposed { get; private set; }

        public FakeOutputCollection(params DisposableNamedOnnxValue[] values) => _values = values;

        public int Count => _values.Length;
        public DisposableNamedOnnxValue this[int index] => _values[index];
        public IEnumerator<DisposableNamedOnnxValue> GetEnumerator() => ((IEnumerable<DisposableNamedOnnxValue>)_values).GetEnumerator();
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => _values.GetEnumerator();
        public void Dispose() => Disposed = true;
    }

    /// <summary>
    /// A [1, seq, vocab] logits tensor whose LAST position row has its argmax at
    /// <paramref name="sampledId"/>, with every earlier position filled with non-argmax junk so the
    /// "last position" selection is observable.
    /// </summary>
    public static DenseTensor<float> LogitsTensor(int seq, int vocab, int sampledId)
    {
        var t = new DenseTensor<float>(new[] { 1, seq, vocab });
        for (int s = 0; s < seq; s++)
        for (int v = 0; v < vocab; v++)
            t[0, s, v] = -5f;
        t[0, seq - 1, sampledId] = 5f;
        return t;
    }

    /// <summary>The EngineInfo stub used with ApplyLoadedState to fake a loaded session.</summary>
    public static InferenceEngineInfo FakeEngineInfo() => new()
    {
        Provider = InferenceProvider.Cpu,
        ModelName = "fake.onnx",
        ModelPath = "fake.onnx",
        ModelSource = ModelSource.ExplicitPath,
    };
}
