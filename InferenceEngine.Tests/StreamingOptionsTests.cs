using InferenceEngine.Core;
using Xunit;

namespace InferenceEngine.Tests;

/// <summary>
/// Covers the full property surface of <see cref="StreamingOptions"/> and
/// <see cref="SamplingParams"/>, including defaults and null/identity guard semantics.
/// </summary>
public sealed class StreamingOptionsTests
{
    [Fact]
    public void StreamingOptions_Defaults_AreSafeGreedy()
    {
        var o = new StreamingOptions();

        Assert.Equal(256, o.MaxSteps);
        Assert.NotNull(o.EosTokenIds);
        Assert.Empty(o.EosTokenIds);
        Assert.NotNull(o.Sampling);
        Assert.Equal(0f, o.Sampling.Temperature);
        Assert.False(o.IncludePrefillInStream);
        Assert.Null(o.ChunkSize);
    }

    [Fact]
    public void StreamingOptions_AllProperties_RoundTrip()
    {
        var o = new StreamingOptions
        {
            MaxSteps = 7,
            EosTokenIds = new[] { 32000, 32002 },
            Sampling = new SamplingParams { Temperature = 0.7f },
            IncludePrefillInStream = true,
            ChunkSize = 4,
        };

        Assert.Equal(7, o.MaxSteps);
        Assert.Equal(new[] { 32000, 32002 }, o.EosTokenIds);
        Assert.Same(o.Sampling, o.Sampling);
        Assert.Equal(0.7f, o.Sampling.Temperature);
        Assert.True(o.IncludePrefillInStream);
        Assert.Equal(4, o.ChunkSize);
    }

    [Fact]
    public void StreamingOptions_EosTokenIds_CanBeSetToEmpty()
    {
        // Engines treat an empty EOS list as "no early termination"; the guard in
        // TextGenerationEngine.ProcessStepOutputs checks Count > 0 before Contains.
        var o = new StreamingOptions { EosTokenIds = new int[0] };

        Assert.NotNull(o.EosTokenIds);
        Assert.Empty(o.EosTokenIds);

        // Null assignments are representable (property is not guarded); engines must tolerate
        // them via null-conditional access. Exercise the null-tolerant read path.
        var nullEos = new StreamingOptions { EosTokenIds = null! };
        Assert.Null(nullEos.EosTokenIds);
        Assert.False(nullEos.EosTokenIds is { Count: > 0 });
    }

    [Fact]
    public void SamplingParams_Defaults_AreIdentity()
    {
        var p = new SamplingParams();

        Assert.Equal(0f, p.Temperature);       // 0 => greedy argmax
        Assert.Equal(0, p.TopK);               // 0 => disabled
        Assert.Equal(1f, p.TopP);              // 1 => disabled
        Assert.Equal(1f, p.RepetitionPenalty); // 1 => disabled
    }

    [Fact]
    public void SamplingParams_AllProperties_RoundTrip()
    {
        var p = new SamplingParams
        {
            Temperature = 1.3f,
            TopK = 40,
            TopP = 0.9f,
            RepetitionPenalty = 1.1f,
        };

        Assert.Equal(1.3f, p.Temperature);
        Assert.Equal(40, p.TopK);
        Assert.Equal(0.9f, p.TopP);
        Assert.Equal(1.1f, p.RepetitionPenalty);
    }
}
