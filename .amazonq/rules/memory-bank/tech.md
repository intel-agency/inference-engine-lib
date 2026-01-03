# Inference Engine Library - Technology Stack

## Programming Languages & Frameworks

### Primary Language
- **C# 12.0** with .NET 8.0 target framework
- **Nullable Reference Types**: Enabled for enhanced type safety
- **Implicit Usings**: Enabled for cleaner code

### Runtime Requirements
- **.NET 8.0 Runtime**: Cross-platform runtime support
- **Native Libraries**: Platform-specific ONNX Runtime binaries

## Core Dependencies

### Machine Learning & Inference
```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.23.0" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.Extensions" Version="0.8.0" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.DirectML" Version="1.17.0" />
```

### Logging & Dependency Injection
```xml
<PackageReference Include="Microsoft.Extensions.Logging.Abstractions" Version="8.0.0" />
<PackageReference Include="Microsoft.Extensions.DependencyInjection.Abstractions" Version="8.0.0" />
```

### Image Processing
```xml
<PackageReference Include="SkiaSharp" Version="3.119.1" />
```

### Tensor Operations
- **System.Numerics.Tensors**: Built-in .NET tensor support

## Build System

### Project Configuration
- **SDK**: Microsoft.NET.Sdk
- **Target Framework**: net8.0
- **Assembly Loading**: `CopyLocalLockFileAssemblies=true` for native DLL support

### Build Commands
```bash
# Restore dependencies
dotnet restore

# Build project
dotnet build

# Run tests
dotnet test

# Create NuGet package
dotnet pack
```

## Platform-Specific Technologies

### Windows
- **DirectML**: Hardware acceleration via DirectX Machine Learning
- **Native Libraries**: `onnxruntime.dll`, `onnxruntime_providers_shared.dll`

### macOS
- **CoreML**: Hardware acceleration via Apple's ML framework
- **Native Libraries**: `libonnxruntime.dylib`, `libortextensions.dylib`

### Linux
- **CPU Provider**: Standard CPU execution
- **Native Libraries**: `libonnxruntime.so`, `libonnxruntime_providers_shared.so`

## Development Tools

### Code Quality
- **Trunk**: Code formatting and linting
- **EditorConfig**: Consistent code style
- **Nullable Reference Types**: Compile-time null safety

### IDE Support
- **Visual Studio Code**: Primary development environment
- **Visual Studio**: Full IDE support
- **JetBrains Rider**: Cross-platform IDE support

## Runtime Architecture

### Execution Providers
1. **DirectML Provider** (Windows)
   - GPU acceleration via DirectX
   - Automatic device selection
   - Fallback to CPU on failure

2. **CoreML Provider** (macOS)
   - Neural Engine acceleration
   - GPU acceleration
   - Automatic optimization

3. **CPU Provider** (All platforms)
   - Universal fallback
   - Optimized CPU execution
   - Multi-threading support

### Memory Management
- **Native Resource Cleanup**: IDisposable pattern
- **Tensor Lifecycle**: Automatic disposal
- **Session Management**: Controlled initialization/cleanup

## Performance Optimizations

### Graph Optimization
- **ORT_ENABLE_ALL**: Maximum ONNX Runtime optimizations
- **Session Reuse**: Single session per engine instance
- **Warmup Inference**: Eliminates first-run latency

### Memory Efficiency
- **Zero-Copy Operations**: Where possible with native tensors
- **Streaming Disposal**: Immediate cleanup of temporary resources
- **Context Passing**: Efficient state sharing between processing steps

## Testing Framework

### Test Infrastructure
- **MSTest**: Primary testing framework
- **xUnit**: Alternative testing support
- **Integration Tests**: End-to-end model testing

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Full pipeline testing
- **Performance Tests**: Latency and throughput validation

## Deployment Considerations

### Package Distribution
- **NuGet Package**: Primary distribution method
- **Native Dependencies**: Automatically included via runtime packages
- **Platform Detection**: Runtime provider selection

### Production Requirements
- **.NET 8.0 Runtime**: Required on target systems
- **Hardware Drivers**: DirectML/CoreML drivers for acceleration
- **Model Files**: ONNX format models (.onnx files)