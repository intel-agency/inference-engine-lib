# **Inference Engine Library \- Architecture & Build Guide**

## **1\. Executive Summary**

This repository hosts **InferenceEngine.Core**, a cross-platform .NET 10.0 library designed to provide a unified interface for running AI models (specifically ONNX) with hardware acceleration on Windows, Linux, and macOS.

**Core Philosophy:** "Write Once, Run Accelerated Anywhere."

The library abstracts away the complexity of loading different Execution Providers (DirectML, CoreML, ROCm) so the consumer code looks identical on all platforms.

## **2\. The "Native Gap" Challenge**

While .NET is cross-platform, high-performance AI inference relies on **Native C++ Libraries** (.dll, .so, .dylib) to talk to GPU drivers.

* **Windows:** Microsoft provides Microsoft.ML.OnnxRuntime.DirectML via NuGet. This works out-of-the-box.  
* **macOS:** Microsoft provides CoreML support in the base package. This works out-of-the-box.  
* **Linux (AMD GPU):** **CRITICAL GAP.** Microsoft does *not* provide a NuGet package for ROCm (AMD GPU) support on Linux. The standard Linux package is CPU-only.

### **The Solution: Dedicated ROCm NuGet Package**

ROCm acceleration for Linux is provided by a **standalone NuGet package** built and published independently:

* **Package:** `InferenceEngine.ROCm.Runtime.linux-x64`
* **Source:** [`intel-agency/inference-engine-rocm`](https://github.com/intel-agency/inference-engine-rocm)
* **Usage:** Add a `PackageReference` to `InferenceEngine.ROCm.Runtime.linux-x64` alongside `InferenceEngine.Core`. The package ships the pre-compiled `libonnxruntime.so` and `libonnxruntime_providers_rocm.so` under `runtimes/linux-x64/native/`.

This keeps `InferenceEngine.Core` a pure managed package and allows the ~60-minute Docker compilation to live in its own dedicated CI pipeline.

## **3\. System Architecture**

```mermaid
graph TD
    subgraph Consumer_App
        App[Agent / App] --> IE_Core[InferenceEngine.Core]
    end

    subgraph InferenceEngine_Core
        IE_Core --> BaseEngine[BaseInferenceEngine.cs]
        BaseEngine --> Options[InferenceEngineOptions]
        BaseEngine --> |Loads| ModelProvider[Model Providers]
    end

    subgraph Hardware_Acceleration
        BaseEngine -- Windows --> DML[DirectML (NuGet)]
        BaseEngine -- macOS --> CoreML[CoreML (Built-in)]
        BaseEngine -- Linux --> ROCm[InferenceEngine.ROCm.Runtime.linux-x64 (NuGet)]
    end
```

## **4\. Build System Details**

### **4.1. Linux ROCm Native Libraries**

ROCm native libraries are **not built in this repository**. They are compiled and packaged in
[`intel-agency/inference-engine-rocm`](https://github.com/intel-agency/inference-engine-rocm) and published as
`InferenceEngine.ROCm.Runtime.linux-x64` to GitHub Packages / NuGet.org.

To consume ROCm acceleration, add a reference to that package. See the `inference-engine-rocm` repo for build details.

### **4.2. CI/CD Workflows**

* `.github/workflows/build-test.yml` — Cross-platform build and test (Windows, macOS, Linux x64/arm64).
* `.github/workflows/publish.yml` — Packs `InferenceEngine.Core` and publishes to GitHub Packages and NuGet.org.

## **5\. Runtime Logic (BaseInferenceEngine.cs)**

The engine detects the OS at runtime and loads the appropriate provider. For Linux ROCm, it uses the **typed ROCm API** exposed by ONNX Runtime, configuring `OrtROCMProviderOptions` and then calling `AppendExecutionProvider_ROCm(...)`.

if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))  
{  
    // Configure the ROCm execution provider using the typed ONNX Runtime API  
    using var rocmOptions = new OrtROCMProviderOptions();  
    rocmOptions.UpdateOptions(new Dictionary<string, string> { ["device_id"] = deviceId, ... });  
    options.AppendExecutionProvider_ROCm(rocmOptions);  
}  
