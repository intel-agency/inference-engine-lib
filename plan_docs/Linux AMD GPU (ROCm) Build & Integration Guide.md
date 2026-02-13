# **Linux AMD GPU (ROCm) Build & Integration Guide**

## **1\. The Challenge: The "Missing Link"**

In the .NET ecosystem, Microsoft provides excellent, ready-made NuGet packages for the most common inference scenarios. These packages handle the complex task of bridging managed C\# code with native C++ binaries.

* **Windows:** Microsoft.ML.OnnxRuntime.DirectML. This single package abstracts away the underlying hardware differences between AMD, NVIDIA, and Intel GPUs using the OS-level DirectML API.  
* **macOS:** Microsoft.ML.OnnxRuntime. Since version 1.13+, this standard package natively includes the CoreML execution provider, allowing direct access to the Apple Neural Engine (NPU) on Silicon chips.  
* **Linux CPU:** Microsoft.ML.OnnxRuntime. This provides a highly optimized CPU runner using AVX/AVX512 instructions, which is sufficient for light workloads but insufficient for real-time computer vision.

### **The Gap**

There is **no official NuGet package** for AMD ROCm on Linux (e.g., Microsoft.ML.OnnxRuntime.ROCm does not exist on NuGet.org).

**The Consequence:** If you deploy your standard Linux-compatible application to a high-performance machine equipped with an AMD GPU (like your RX 6700 XT), the application will default to CPU execution. This leaves the powerful GPU completely idle, resulting in inference times that are 10x-50x slower than potential performance. To unlock hardware acceleration, you must manually compile the native C++ shared library (libonnxruntime\_providers\_rocm.so) and manually bundle it with your application.

## **2\. The Solution: Dockerized Compilation**

Compiling ROCm software on a general-purpose Linux distribution (like Ubuntu 24.04 or Debian) is notoriously difficult due to "Dependency Hell." You often face glibc version mismatches, missing legacy libraries (like libcholmod), or conflicts between system packages and the ROCm SDK.

**The Fix:** We will use an official **AMD ROCm Docker Image** (rocm/dev-ubuntu-22.04) as a disposable, reproducible build environment.

**Why Docker?**

1. **Pristine Toolchain:** The image comes with the correct version of the HIP compiler (hipcc) pre-installed and added to the system PATH.  
2. **Resolved Dependencies:** All the math libraries (MIOpen, rocBLAS, libcholmod) are already present and version-matched to the compiler.  
3. **Isolation:** The build process happens inside the container, keeping your host system clean. We map your source code into the container, build it there, and write the resulting artifacts back to your host machine.

### **Prerequisites**

* **Windows with WSL2** installed (Any standard distro, e.g., Ubuntu 22.04/24.04).  
* **Docker Desktop** installed on Windows and configured to use the WSL2 backend (Settings \-\> Resources \-\> WSL Integration).  
* **Git** installed in your WSL environment.

## **3\. The Build Procedure**

Follow these steps exactly to generate the required .so files.

### **Step 1: Prepare the Source Code (On Host)**

Open your WSL2 terminal. We need to create a dedicated workspace and clone the Microsoft ONNX Runtime repository.

*Technical Note: We use \--recursive because ONNX Runtime depends on dozens of third-party submodules (Protobuf, Flatbuffers, etc.) that must be present for the build to succeed.*

\# Create a clean workspace directory  
mkdir \-p \~/onnx-build-work  
cd \~/onnx-build-work

\# Clone recursively to get all dependencies  
\# This pulls \~1GB of source code, so it may take a few minutes.  
git clone \--recursive \[https://github.com/microsoft/onnxruntime\](https://github.com/microsoft/onnxruntime)

