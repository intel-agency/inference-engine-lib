using System.Collections.Generic;

namespace InferenceEngine.Core;

/// <summary>Per-call streaming controls. Defaults are safe/greedy.</summary>
public sealed class StreamingOptions
{
    /// <summary>Maximum decode steps (prefill counts as step 0) before the loop stops.</summary>
    public int MaxSteps { get; set; } = 256;

    /// <summary>Token ids that terminate generation when sampled.</summary>
    public IReadOnlyList<int> EosTokenIds { get; set; } = System.Array.Empty<int>();

    /// <summary>Sampling strategy. Honored by <c>BaseStreamingInferenceEngine.SampleNextToken</c>;
    /// temperature 0 (default) is greedy argmax; &gt;0 enables softmax sampling with top-k/top-p.</summary>
    public SamplingParams Sampling { get; set; } = new();

    /// <summary>If true, also yield prompt tokens from the prefill pass. Default false.</summary>
    public bool IncludePrefillInStream { get; set; } = false;

    /// <summary>Batch >=1 decoded steps per yield (null/1 = one step per yield). Default null.</summary>
    public int? ChunkSize { get; set; } = null;
}

/// <summary>Sampling parameters. Disabled/identity values mean "off".</summary>
public sealed class SamplingParams
{
    /// <summary>0 => greedy (argmax).</summary>
    public float Temperature { get; set; } = 0f;

    /// <summary>0 => disabled.</summary>
    public int TopK { get; set; } = 0;

    /// <summary>1 => disabled.</summary>
    public float TopP { get; set; } = 1f;

    /// <summary>1 => disabled.</summary>
    public float RepetitionPenalty { get; set; } = 1f;
}
