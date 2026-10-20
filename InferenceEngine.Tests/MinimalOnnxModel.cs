using System.Collections.Generic;
using System.Text;

namespace InferenceEngine.Tests;

/// <summary>
/// Builds a minimal valid ONNX model (a single Identity node mapping one input to one
/// output of a 1-element tensor) by hand-encoding the protobuf wire format. Lets lifecycle
/// tests construct real CPU <c>InferenceSession</c>s without shipping model binaries.
/// </summary>
internal static class MinimalOnnxModel
{
    /// <summary>ONNX tensor element types used by <see cref="Build"/>.</summary>
    public enum ElementType
    {
        Float = 1,
        Int32 = 6,
    }

    /// <summary>Writes an identity model (<c>input</c> -> Identity -> <c>output</c>) to <paramref name="path"/>.</summary>
    public static void Build(string path, ElementType elementType = ElementType.Float)
    {
        File.WriteAllBytes(path, Build(elementType));
    }

    public static byte[] Build(ElementType elementType = ElementType.Float)
    {
        var model = new List<byte>();
        WriteVarintField(model, 1, 9); // ir_version
        WriteStringField(model, 2, "InferenceEngine.Tests");
        WriteLengthDelimitedField(model, 7, BuildGraph(elementType));
        WriteLengthDelimitedField(model, 8, BuildOpsetImport());
        return [.. model];
    }

    private static byte[] BuildGraph(ElementType elementType)
    {
        var graph = new List<byte>();
        WriteLengthDelimitedField(graph, 1, BuildIdentityNode());
        WriteStringField(graph, 2, "minimal-identity");
        WriteLengthDelimitedField(graph, 11, BuildValueInfo("input", elementType));
        WriteLengthDelimitedField(graph, 12, BuildValueInfo("output", elementType));
        return [.. graph];
    }

    private static byte[] BuildIdentityNode()
    {
        var node = new List<byte>();
        WriteStringField(node, 1, "input");
        WriteStringField(node, 2, "output");
        WriteStringField(node, 4, "Identity");
        return [.. node];
    }

    private static byte[] BuildValueInfo(string name, ElementType elementType)
    {
        var valueInfo = new List<byte>();
        WriteStringField(valueInfo, 1, name);
        // type_proto -> tensor_type
        var typeProto = new List<byte>();
        WriteLengthDelimitedField(typeProto, 1, BuildTensorTypeProto(elementType));
        WriteLengthDelimitedField(valueInfo, 2, [.. typeProto]);
        return [.. valueInfo];
    }

    private static byte[] BuildTensorTypeProto(ElementType elementType)
    {
        var tensorType = new List<byte>();
        WriteVarintField(tensorType, 1, (ulong)elementType);
        WriteLengthDelimitedField(tensorType, 2, BuildShapeProto());
        return [.. tensorType];
    }

    private static byte[] BuildShapeProto()
    {
        var shape = new List<byte>();
        WriteLengthDelimitedField(shape, 1, BuildDimensionProto(1));
        return [.. shape];
    }

    private static byte[] BuildDimensionProto(ulong value)
    {
        var dimension = new List<byte>();
        WriteVarintField(dimension, 1, value);
        return [.. dimension];
    }

    private static byte[] BuildOpsetImport()
    {
        var opset = new List<byte>();
        WriteVarintField(opset, 2, 13); // ai.onnx default domain, opset 13
        return [.. opset];
    }

    private static void WriteStringField(List<byte> target, int fieldNumber, string value)
        => WriteLengthDelimitedField(target, fieldNumber, Encoding.UTF8.GetBytes(value));

    private static void WriteLengthDelimitedField(List<byte> target, int fieldNumber, byte[] payload)
    {
        WriteTag(target, fieldNumber, wireType: 2);
        WriteVarint(target, (ulong)payload.Length);
        target.AddRange(payload);
    }

    private static void WriteVarintField(List<byte> target, int fieldNumber, ulong value)
    {
        WriteTag(target, fieldNumber, wireType: 0);
        WriteVarint(target, value);
    }

    private static void WriteTag(List<byte> target, int fieldNumber, int wireType)
        => WriteVarint(target, (ulong)((fieldNumber << 3) | wireType));

    private static void WriteVarint(List<byte> target, ulong value)
    {
        while (value >= 0x80)
        {
            target.Add((byte)((value & 0x7F) | 0x80));
            value >>= 7;
        }
        target.Add((byte)value);
    }
}
