using System;
using System.Text.Json;

namespace InferenceEngine.Core.Worker;

/// <summary>
/// Serializes <typeparamref name="TInput"/>/<typeparamref name="TOutput"/> across the
/// out-of-process IPC boundary. The default <see cref="SystemTextJsonWorkerSerializer{TInput,TOutput}"/>
/// handles <see cref="byte"/>[] as base64 (System.Text.Json's built-in) and POCOs as JSON.
/// </summary>
public interface IWorkerSerializer<TInput, TOutput>
{
    string SerializeInput(TInput input);
    TInput DeserializeInput(string serialized);
    string SerializeOutput(TOutput output);
    TOutput DeserializeOutput(string serialized);
}

/// <summary>
/// Default <see cref="IWorkerSerializer{TInput,TOutput}"/> using <see cref="JsonSerializer"/>.
/// <see cref="byte"/>[] is serialized as a base64 string (System.Text.Json's built-in behavior),
/// so <c>byte[]</c>→<c>List&lt;T&gt;</c> engines (e.g. ONNX detection) work out of the box.
/// </summary>
public sealed class SystemTextJsonWorkerSerializer<TInput, TOutput> : IWorkerSerializer<TInput, TOutput>
{
    private static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web)
    {
        IncludeFields = true,
    };

    public string SerializeInput(TInput input) => JsonSerializer.Serialize(input, Options);

    public TInput DeserializeInput(string serialized) =>
        JsonSerializer.Deserialize<TInput>(serialized, Options)
            ?? throw new InvalidOperationException("Deserialized worker input was null.");

    public string SerializeOutput(TOutput output) => JsonSerializer.Serialize(output, Options);

    public TOutput DeserializeOutput(string serialized) =>
        JsonSerializer.Deserialize<TOutput>(serialized, Options)
            ?? throw new InvalidOperationException("Deserialized worker output was null.");
}
