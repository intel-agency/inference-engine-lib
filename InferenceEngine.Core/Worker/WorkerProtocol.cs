using System;
using System.Buffers;
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

    /// <summary>MIGraphX compiled-model cache directory (null = worker-side default,
    /// empty string = disabled). Mirrors <see cref="InferenceEngineOptions.MIGraphXModelCacheDirectory"/>.</summary>
    public string? MIGraphXModelCacheDirectory { get; set; }
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
        // Rent the 4-byte length prefix instead of allocating it; a clean EOF here (the worker is
        // gone before any frame bytes) is surfaced as a null return so the caller can shut down
        // gracefully instead of throwing.
        byte[]? prefix = null;
        int length;
        try
        {
            prefix = ArrayPool<byte>.Shared.Rent(4);
            if (!await ReadExactIntoAsync(stream, prefix.AsMemory(0, 4), ct).ConfigureAwait(false))
                return null; // clean EOF
            length = BitConverter.ToInt32(prefix, 0);
        }
        finally
        {
            if (prefix is not null)
                ArrayPool<byte>.Shared.Return(prefix);
        }

        if (length <= 0) throw new InvalidDataException($"Invalid frame length: {length}.");
        if (length > MaxFrameBytes) throw new InvalidDataException($"Frame too large: {length} bytes.");

        // Rent the payload buffer and deserialize straight from the rented span, then return it. This
        // avoids allocating a fresh byte[length] on every IPC read; JsonSerializer.Deserialize copies
        // the strings out, so the buffer is safe to return before this method exits.
        byte[]? payload = null;
        try
        {
            payload = ArrayPool<byte>.Shared.Rent(length);
            if (!await ReadExactIntoAsync(stream, payload.AsMemory(0, length), ct).ConfigureAwait(false))
                throw new EndOfStreamException("Stream closed mid-frame.");
            return JsonSerializer.Deserialize<WorkerEnvelope>(payload.AsSpan(0, length), WorkerJson.Options);
        }
        finally
        {
            if (payload is not null)
                ArrayPool<byte>.Shared.Return(payload);
        }
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

    /// <summary>
    /// Reads exactly <c>destination.Length</c> bytes into <paramref name="destination"/>. Returns
    /// <see langword="false"/> on a clean EOF at the very start (zero bytes read) and throws
    /// <see cref="EndOfStreamException"/> if the stream ends partway through. Used by
    /// <see cref="ReadAsync"/> to distinguish graceful worker shutdown (null return) from a truncated
    /// frame (throw). Reads into a caller-supplied buffer so no per-frame allocation occurs.
    /// </summary>
    private static async Task<bool> ReadExactIntoAsync(Stream stream, Memory<byte> destination, CancellationToken ct)
    {
        int read = 0;
        int count = destination.Length;
        while (read < count)
        {
            var n = await stream.ReadAsync(destination.Slice(read, count - read), ct).ConfigureAwait(false);
            if (n == 0)
                return read == 0 ? false : throw new EndOfStreamException("Stream closed mid-frame.");
            read += n;
        }
        return true;
    }
}

internal static class WorkerJson
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web) { IncludeFields = true };
}
