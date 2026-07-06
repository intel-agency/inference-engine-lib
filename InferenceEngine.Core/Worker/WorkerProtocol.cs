using System;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using InferenceEngine.Core.Diagnostics;

namespace InferenceEngine.Core.Worker;

/// <summary>Serializable subset of <see cref="InferenceEngineOptions"/> sent over IPC.
/// Delegates (<see cref="InferenceEngineOptions.OnSessionInitialized"/>) cannot cross the boundary
/// and are intentionally omitted.</summary>
public sealed class WorkerEngineOptions
{
    public string? ModelPath { get; set; }
    public bool UseGpuAcceleration { get; set; } = true;
    public int DeviceId { get; set; } = 0;
    public bool WarmupOnLoad { get; set; } = false;
}

/// <summary>
/// Single envelope for all worker messages. The <see cref="Kind"/> discriminator selects the role
/// of the other fields. Framed on the wire as <c>&lt;4-byte little-endian length&gt;&lt;UTF-8 JSON&gt;</c>.
/// </summary>
public sealed class WorkerEnvelope
{
    /// <summary>One of: <c>load</c>, <c>loaded</c>, <c>predict</c>, <c>result</c>, <c>shutdown</c>.</summary>
    public string Kind { get; set; } = "";

    /// <summary>Consumer-defined engine id (load only).</summary>
    public string EngineId { get; set; } = "";

    /// <summary>Serialized input (predict) / output (result) payload, crossed via
    /// <see cref="IWorkerSerializer{TInput,TOutput}"/>.</summary>
    public string? Payload { get; set; }

    /// <summary>Engine options (load only).</summary>
    public WorkerEngineOptions? Options { get; set; }

    /// <summary>Loaded engine diagnostics (loaded only).</summary>
    public InferenceEngineInfo? EngineInfo { get; set; }

    /// <summary>Non-null when the worker hit an error handling the request.</summary>
    public string? Error { get; set; }
}

/// <summary>Length-prefixed JSON framing over a duplex stream (stdin/stdout pipe).</summary>
internal static class WorkerFrame
{
    // 16 MiB cap: a corrupt/buggy length prefix is rejected before allocating, rather than forcing a
    // single huge contiguous (LOH) allocation. Any plausible IPC JSON envelope is far below this.
    private const int MaxFrameBytes = 16 * 1024 * 1024; // 16 MiB guard against corrupt length prefixes

    public static async Task<WorkerEnvelope?> ReadAsync(Stream stream, CancellationToken ct)
    {
        var lengthBytes = await ReadExactAsync(stream, 4, ct).ConfigureAwait(false);
        if (lengthBytes is null) return null; // clean EOF

        var length = BitConverter.ToInt32(lengthBytes, 0);
        if (length <= 0) throw new InvalidDataException($"Invalid frame length: {length}.");
        if (length > MaxFrameBytes) throw new InvalidDataException($"Frame too large: {length} bytes.");

        var payload = await ReadExactAsync(stream, length, ct).ConfigureAwait(false)
            ?? throw new EndOfStreamException("Stream closed mid-frame.");

        return JsonSerializer.Deserialize<WorkerEnvelope>(payload, WorkerJson.Options);
    }

    public static async Task WriteAsync(Stream stream, WorkerEnvelope envelope, CancellationToken ct)
    {
        var json = JsonSerializer.SerializeToUtf8Bytes(envelope, WorkerJson.Options);
        var prefix = BitConverter.GetBytes(json.Length); // 4-byte little-endian
        await stream.WriteAsync(prefix, ct).ConfigureAwait(false);
        await stream.WriteAsync(json, ct).ConfigureAwait(false);
        await stream.FlushAsync(ct).ConfigureAwait(false);
    }

    public static void Write(Stream stream, WorkerEnvelope envelope)
    {
        var json = JsonSerializer.SerializeToUtf8Bytes(envelope, WorkerJson.Options);
        var prefix = BitConverter.GetBytes(json.Length); // 4-byte little-endian
        stream.Write(prefix, 0, prefix.Length);
        stream.Write(json, 0, json.Length);
        stream.Flush();
    }

    private static async Task<byte[]?> ReadExactAsync(Stream stream, int count, CancellationToken ct)
    {
        var buffer = new byte[count];
        var read = 0;
        while (read < count)
        {
            var n = await stream.ReadAsync(buffer.AsMemory(read, count - read), ct).ConfigureAwait(false);
            if (n == 0)
                return read == 0 ? null : throw new EndOfStreamException("Stream closed mid-frame.");
            read += n;
        }
        return buffer;
    }
}

internal static class WorkerJson
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web) { IncludeFields = true };
}
