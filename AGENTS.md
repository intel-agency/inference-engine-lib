# AI Agents Integration Guide

## Overview
This library provides a foundation for building AI agents that require inference capabilities. The generic interface design allows agents to integrate various AI models (computer vision, NLP, etc.) through a unified API.

## Agent Architecture

### Core Components
- **IInferenceEngine<TInput, TOutput>**: Generic contract for AI inference
- **BaseInferenceEngine**: ONNX Runtime implementation with hardware acceleration
- **InferenceEngineOptions**: Configuration for model loading and execution

### Agent Integration Patterns

#### 1. Vision Agent
```csharp
public class VisionAgent
{
    private readonly IInferenceEngine<byte[], List<Detection>> _detector;
    
    public async Task<AgentResponse> ProcessImageAsync(byte[] image)
    {
        var detections = await _detector.PredictAsync(image);
        return new AgentResponse { Objects = detections };
    }
}
```

#### 2. Multi-Modal Agent
```csharp
public class MultiModalAgent
{
    private readonly IInferenceEngine<byte[], List<Detection>> _vision;
    private readonly IInferenceEngine<string, string> _nlp;
    
    public async Task<string> AnalyzeSceneAsync(byte[] image, string query)
    {
        var objects = await _vision.PredictAsync(image);
        var context = $"Scene contains: {string.Join(", ", objects.Select(o => o.Label))}";
        return await _nlp.PredictAsync($"{context}\nQuery: {query}");
    }
}
```

## Implementation Guidelines

### Custom Agent Engines
1. Inherit from `BaseInferenceEngine<TInput, TOutput>`
2. Implement `PreProcessWithContext()` for input preparation
3. Implement `PostProcessWithContext()` for output processing
4. Handle agent-specific data types and transformations

### Agent Lifecycle
1. **Initialization**: Load models during agent startup
2. **Processing**: Use `PredictAsync()` for inference requests
3. **Cleanup**: Dispose engines when agent shuts down

### Performance Considerations
- Enable `WarmupOnLoad` for production agents
- Use hardware acceleration when available
- Implement proper resource disposal
- Consider batching for high-throughput scenarios

## Example Agent Implementation

See `Examples/FaceDetectionEngine.cs` for a complete computer vision agent that demonstrates:
- Image preprocessing and scaling
- Object detection with confidence filtering
- Non-maximum suppression
- Coordinate space handling