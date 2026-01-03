# Inference Engine Library - Development Guidelines

## Code Quality Standards

### Formatting and Style
- **Namespace Organization**: Single namespace per file (`InferenceEngine.Core`, `InferenceEngine.Core.Examples`)
- **Using Statements**: Organized alphabetically, system namespaces first
- **Indentation**: 4 spaces, no tabs
- **Line Endings**: Windows CRLF (`\r\n`)
- **Braces**: Opening braces on same line for methods, classes, and control structures
- **Access Modifiers**: Always explicit (`public`, `private`, `protected`, `internal`)

### Naming Conventions
- **Classes**: PascalCase (`BaseInferenceEngine`, `FaceDetectionEngine`)
- **Interfaces**: PascalCase with `I` prefix (`IInferenceEngine`)
- **Methods**: PascalCase (`LoadAsync`, `PredictAsync`, `PreProcessWithContext`)
- **Properties**: PascalCase (`ModelPath`, `UseGpuAcceleration`, `DeviceId`)
- **Fields**: camelCase with underscore prefix (`_options`, `_logger`, `_session`)
- **Constants**: PascalCase (`ConfidenceThreshold`, `IouThreshold`)
- **Local Variables**: camelCase (`maxScore`, `maxClassId`, `sourceImage`)

### Documentation Standards
- **XML Documentation**: Required for all public APIs
- **Summary Tags**: Comprehensive descriptions for classes and methods
- **Parameter Documentation**: `<param>` tags for all method parameters
- **Return Documentation**: `<returns>` tags for non-void methods
- **Type Parameter Documentation**: `<typeparam>` tags for generic types
- **Example Usage**: Include code examples in complex implementations

## Structural Conventions

### Generic Interface Design
```csharp
public interface IInferenceEngine<TInput, TOutput> : IDisposable
{
    Task LoadAsync();
    Task<TOutput> PredictAsync(TInput input);
}
```

### Abstract Base Class Pattern
```csharp
public abstract class BaseInferenceEngine<TInput, TOutput> : IInferenceEngine<TInput, TOutput>
{
    protected readonly InferenceEngineOptions _options;
    protected readonly ILogger _logger;
    
    protected abstract (List<NamedOnnxValue>, object?) PreProcessWithContext(TInput input);
    protected abstract TOutput PostProcessWithContext(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context);
}
```

### Configuration Class Pattern
```csharp
public class InferenceEngineOptions
{
    public string ModelPath { get; set; } = string.Empty;
    public bool UseGpuAcceleration { get; set; } = true;
    public int DeviceId { get; set; } = 0;
    public bool WarmupOnLoad { get; set; } = true;
}
```

## Implementation Patterns

### Context Passing Pattern
- **Purpose**: Share state between preprocessing and postprocessing
- **Implementation**: Return tuple from `PreProcessWithContext` with tensors and context object
- **Usage**: Pass context to `PostProcessWithContext` for coordinate transformations, scaling factors
```csharp
protected override (List<NamedOnnxValue>, object?) PreProcessWithContext(TInput input)
{
    var context = new ImageScaleContext { ScaleX = scaleX, ScaleY = scaleY };
    return (tensors, context);
}
```

### Resource Management Pattern
- **IDisposable Implementation**: All engines implement IDisposable
- **Using Statements**: Wrap disposable resources (`using var outputTensors`)
- **Finally Blocks**: Ensure cleanup in exception scenarios
- **Null-Forgiving Operator**: Use `!` when LoadAsync guarantees non-null state

### Error Handling Pattern
- **Validation**: Check file existence before loading models
- **Exception Wrapping**: Wrap native exceptions with meaningful messages
- **Logging**: Log errors, warnings, and informational messages
- **Graceful Degradation**: Fallback to CPU when hardware acceleration fails

### Async/Await Pattern
- **Async Methods**: All I/O operations use async/await
- **Task.Run**: Wrap CPU-intensive operations
- **ConfigureAwait**: Not used (UI context not relevant)
- **Async Suffix**: All async methods end with "Async"

## Internal API Usage Patterns

### ONNX Runtime Integration
```csharp
// Session creation with options
_session = new InferenceSession(_options.ModelPath, sessionOptions);

// Tensor creation
var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

// Named value creation
NamedOnnxValue.CreateFromTensor(inputName, tensor)

// Inference execution
using var outputTensors = _session.Run(inputTensors, _session.OutputNames, _runOptions);
```

### Logging Integration
```csharp
// Constructor injection
protected BaseInferenceEngine(InferenceEngineOptions options, ILogger? logger = null)
{
    _logger = logger ?? NullLogger.Instance;
}

// Structured logging
_logger.LogInformation("Model loaded. Primary Input: {InputName}", PrimaryInputName);
_logger.LogWarning("DirectML failed to load. Falling back to CPU. Error: {Error}", ex.Message);
```

### Platform Detection Pattern
```csharp
if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
{
    options.AppendExecutionProvider_DML(_options.DeviceId);
}
else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
{
    options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
}
```

## Frequently Used Code Idioms

### Null Coalescing with Validation
```csharp
_options = options ?? throw new ArgumentNullException(nameof(options));
_logger = logger ?? NullLogger.Instance;
```

### Metadata Access Pattern
```csharp
int modelH = InputMetadata![PrimaryInputName!].Dimensions[2];
int modelW = InputMetadata![PrimaryInputName!].Dimensions[3];
```

### Parallel Processing
```csharp
Parallel.For(0, modelH, y =>
{
    for (int x = 0; x < modelW; x++)
    {
        // Pixel processing
    }
});
```

### Switch Expressions
```csharp
return space switch
{
    BoxCoordSpace.Normalized01 => (x * srcW, y * srcH, w * srcW, h * srcH),
    BoxCoordSpace.ModelInputPixels => (x * sX, y * sY, w * sX, h * sY),
    BoxCoordSpace.SourcePixels => (x, y, w, h),
    _ => throw new InvalidOperationException($"Unknown coordinate space detected.")
};
```

## Common Annotations and Attributes

### Nullable Reference Types
- **Enabled**: Project-wide nullable reference types
- **Null-Forgiving Operator**: `!` for guaranteed non-null scenarios
- **Nullable Parameters**: `ILogger? logger = null`
- **Nullable Returns**: `object?` for optional context objects

### Environment Variables
```csharp
private static readonly bool BboxDiagEnabled =
    string.Equals(Environment.GetEnvironmentVariable("INFERENCEAPP_BBOX_DIAG"), "1", StringComparison.OrdinalIgnoreCase);
```

### Collection Initialization
```csharp
private readonly string[] _labels = 
[
    "person", "bicycle", "car", // C# 12 collection expressions
];
```

## Performance Optimization Patterns

### Memory Efficiency
- **Span<T> Usage**: For high-performance array operations
- **ArrayPool**: For temporary buffer allocation
- **Streaming Disposal**: Immediate cleanup of large objects

### Computational Efficiency
- **Parallel.For**: CPU-intensive loops
- **SIMD Operations**: Vector operations where applicable
- **Graph Optimization**: `GraphOptimizationLevel.ORT_ENABLE_ALL`

### Caching Strategies
- **Session Reuse**: Single session per engine instance
- **Metadata Caching**: Store input/output metadata after loading
- **Warmup Inference**: Eliminate first-run compilation overhead