using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace InferenceEngine.Core;

/// <summary>
/// An inference engine that can also yield intermediate model-level steps
/// (token ids / logits / chunks) as an asynchronous stream.
/// </summary>
/// <remarks>
/// Implementations continue to satisfy the one-shot <see cref="IInferenceEngine{TInput, TOutput}"/>
/// contract; streaming is an additive capability. A consumer may program to either contract on the
/// same instance.
/// </remarks>
/// <typeparam name="TInput">The raw input to the engine (e.g. a prompt string or token-id array).</typeparam>
/// <typeparam name="TOutput">The fully-reduced one-shot result (e.g. the complete generated text).</typeparam>
/// <typeparam name="TStep">One streamed model-level step (e.g. a token id, a logit row, or a text chunk).</typeparam>
public interface IStreamingInferenceEngine<TInput, TOutput, TStep> : IInferenceEngine<TInput, TOutput>
{
    /// <summary>
    /// Yield one model-level step (token id / logit row / chunk) at a time.
    /// The consumer decides how to render or accumulate the steps. Cancellation is honored at each
    /// step boundary.
    /// </summary>
    /// <param name="input">The raw input.</param>
    /// <param name="options">Optional per-call streaming controls; <see langword="null"/> uses the engine default.</param>
    /// <param name="cancellationToken">Observed between steps; aborts promptly and leaves the instance clean.</param>
    IAsyncEnumerable<TStep> PredictStreamingAsync(
        TInput input,
        StreamingOptions? options = null,
        CancellationToken cancellationToken = default);
}
