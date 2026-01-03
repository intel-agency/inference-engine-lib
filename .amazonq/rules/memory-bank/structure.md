# Inference Engine Library - Project Structure

## Directory Organization

### Root Structure
```
inference-engine-lib/
├── BaseInferenceEngine.cs          # Core abstract base class
├── IInferenceEngine.cs             # Generic interface contract
├── InferenceEngineOptions.cs       # Configuration class
├── Examples/                       # Implementation examples
│   └── FaceDetectionEngine.cs     # Complete computer vision example
├── Tests/                          # Test projects
├── AGENTS.md                       # AI agent integration guide
├── README.md                       # Project documentation
└── InferenceEngine.Core.csproj     # Project configuration
```

### Build Artifacts
```
bin/Debug/net8.0/                   # Compiled assemblies
├── runtimes/                       # Platform-specific native libraries
│   ├── win-x64/native/            # Windows DirectML libraries
│   ├── osx-arm64/native/          # macOS CoreML libraries
│   └── linux-x64/native/          # Linux CPU libraries
└── InferenceEngine.Core.dll       # Main library assembly
```

## Core Components

### 1. Interface Layer
- **IInferenceEngine<TInput, TOutput>**: Generic contract defining the inference API
  - `LoadAsync()`: Asynchronous model loading
  - `PredictAsync(TInput)`: Main inference method
  - Metadata properties for model introspection

### 2. Base Implementation
- **BaseInferenceEngine<TInput, TOutput>**: Abstract base class providing:
  - ONNX Runtime session management
  - Cross-platform hardware acceleration
  - Resource lifecycle management
  - Context passing between pre/post processing
  - Warmup functionality

### 3. Configuration
- **InferenceEngineOptions**: Configuration class containing:
  - Model path specification
  - Hardware acceleration settings
  - Device selection for multi-GPU
  - Warmup configuration

### 4. Examples
- **FaceDetectionEngine**: Complete implementation demonstrating:
  - Image preprocessing with SkiaSharp
  - YOLOv8-style object detection
  - Non-maximum suppression
  - Coordinate space transformations

## Architectural Patterns

### Generic Interface Design
```csharp
public interface IInferenceEngine<TInput, TOutput>
{
    Task LoadAsync();
    Task<TOutput> PredictAsync(TInput input);
    // Metadata properties...
}
```

### Context Passing Pattern
```csharp
protected abstract (List<NamedOnnxValue>, object?) PreProcessWithContext(TInput input);
protected abstract TOutput PostProcessWithContext(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, object? context);
```

### Hardware Acceleration Strategy
1. **Platform Detection**: Runtime OS detection
2. **Provider Selection**: DirectML (Windows) → CoreML (macOS) → CPU (fallback)
3. **Graceful Degradation**: Automatic fallback on provider failure

### Resource Management
- **IDisposable Implementation**: Proper cleanup of native resources
- **Using Statements**: Automatic disposal of ONNX tensors
- **Session Lifecycle**: Controlled initialization and cleanup

## Component Relationships

### Inheritance Hierarchy
```
IInferenceEngine<TInput, TOutput>
└── BaseInferenceEngine<TInput, TOutput>
    └── [Custom Implementations]
        └── FaceDetectionEngine (example)
```

### Dependency Flow
```
Custom Engine → BaseInferenceEngine → ONNX Runtime → Native Libraries
                     ↓
              InferenceEngineOptions
                     ↓
              ILogger (optional)
```

### Data Flow
```
TInput → PreProcessWithContext() → ONNX Tensors → Inference → Output Tensors → PostProcessWithContext() → TOutput
           ↓                                                                            ↑
        Context Object --------------------------------------------------------→ Context Object
```

## Cross-Platform Architecture

### Platform-Specific Providers
- **Windows**: DirectML execution provider for GPU acceleration
- **macOS**: CoreML execution provider for Neural Engine/GPU
- **Linux**: CPU execution provider with potential CUDA support

### Native Library Management
- Runtime-specific native libraries in `runtimes/` folder
- Automatic platform detection and library loading
- Fallback mechanisms for unsupported hardware

### Build Configuration
- .NET 8.0 target framework
- Nullable reference types enabled
- Local assembly copying for native dependencies
- Conditional package references for platform-specific providers