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
git clone \--recursive <https://github.com/microsoft/onnxruntime>

### **Step 2: Create the Build Script**

We will create a script named compile\_in\_docker.sh inside the onnxruntime folder. This script is designed to run **inside** the Docker container, not on your host machine.

**Action:** Create this file at \~/onnx-build-work/onnxruntime/compile\_in\_docker.sh:

\#\!/bin/bash  
set \-e \# Exit immediately if any command fails

\# \=== CONFIGURATION \===  
\# We pin to v1.19.2 because the 'main' branch is unstable and often contains  
\# breaking changes for the build flags. Using a tagged release guarantees stability.  
ORT\_TAG="v1.19.2"  
ROCM\_HOME="/opt/rocm"

echo "\>\>\> \[1/6\] Setting up Git Safe Directory..."  
\# Docker mounts the volume as root, but the files might be owned by your user.  
\# Git prevents this by default; we must override it.  
git config \--global \--add safe.directory /code

echo "\>\>\> \[2/6\] Checking out stable release $ORT\_TAG..."  
git checkout $ORT\_TAG  
\# Ensure submodules match the specific commit of the tag  
git submodule update \--init \--recursive

echo "\>\>\> \[3/6\] Installing Build Dependencies..."  
\# The AMD base image is barebones to save space. We must install the standard  
\# C++ build chain (CMake, Python) required by the ONNX build system.  
apt-get update  
apt-get install \-y build-essential cmake git python3 python3-dev python3-pip libprotobuf-dev protobuf-compiler

echo "\>\>\> \[4/6\] Configuring Environment..."  
\# These variables force CMake's FindHIP module to locate the compiler  
\# in the correct directory, bypassing auto-detection issues.  
export ROCM\_PATH="$ROCM\_HOME"  
export HIP\_PATH="$ROCM\_HOME"  
export PATH="$ROCM\_HOME/bin:$PATH"

echo "\>\>\> \[5/6\] Cleaning Previous Builds..."  
\# A dirty build cache is the \#1 cause of compilation errors.  
rm \-rf build/Linux

echo "\>\>\> \[6/6\] Starting Compilation (This takes 30-60 mins)..."  
\# \--skip\_tests: CRITICAL. The Docker container (compilation phase) usually DOES NOT  
\# have access to the physical GPU hardware. Running tests would cause the build to fail.  
\# \--cmake\_extra\_defines: "gfx1031" is the architecture code for Navi 22 (RX 6700 XT).  
\# Specifying this speeds up the build by skipping code generation for other cards.  
\# Note: If you switch GPUs, find your architecture code by running 'rocminfo | grep gfx' on the target machine.  
\# Common codes: gfx1030 (RX 6800/6900), gfx1031 (RX 6700 XT), gfx1032 (RX 6600).  
./build.sh \\  
    \--config Release \\  
    \--build\_wheel \\  
    \--use\_rocm \\  
    \--rocm\_home "$ROCM\_HOME" \\  
    \--skip\_tests \\  
    \--skip\_submodule\_sync \\  
    \--parallel \\  
    \--cmake\_extra\_defines CMAKE\_HIP\_ARCHITECTURES="gfx1031"

echo "✅ SUCCESS\! Artifacts available in build/Linux/Release/"

**Make it executable:**

chmod \+x \~/onnx-build-work/onnxruntime/compile\_in\_docker.sh

### **Step 3: Launch the Builder**

Run this Docker command from your WSL2 terminal. It performs the "Volume Mount" magic that links your file system to the container.

\# Ensure you are in the folder containing 'compile\_in\_docker.sh'  
cd \~/onnx-build-work/onnxruntime

\# \-it: Interactive terminal  
\# \--rm: Automatically delete the container after it finishes (saves disk space)  
\# \-v $(pwd):/code:  Maps your current folder (WSL) to /code (Container)  
\# \-w /code: Sets the working directory inside the container  
docker run \-it \--rm \\  
  \-v $(pwd):/code \\  
  \-w /code \\  
  rocm/dev-ubuntu-22.04:latest \\  
  /code/compile\_in\_docker.sh

**What happens next:**

1. Docker pulls the \~4GB AMD image (if not already cached).  
2. It executes your script inside that pristine environment.  
3. It compiles the code using the source files on your hard drive.  
4. **Result:** The compiled binary files (.so) appear in your local build/Linux/Release folder on Windows/WSL, ready to use.

## **4\. Integration Guide**

Once the build finishes, navigate to \~/onnx-build-work/onnxruntime/build/Linux/Release/. You will find two critical files that constitute the "Holy Grail" of Linux GPU support:

1. libonnxruntime.so (The core engine logic).  
2. libonnxruntime\_providers\_rocm.so (The hardware adapter for AMD GPUs).

### **Step 1: Organize Files in C\# Project**

We need to place these files in a location where the .NET build system can find them. The standard convention for native assets is the runtimes folder structure.

In your **InferenceApp.Core** project directory, create this folder structure:

InferenceApp.Core/  
├── runtimes/  
│   └── linux-x64/  
│       └── native/  
│           ├── libonnxruntime.so  
│           └── libonnxruntime\_providers\_rocm.so

### **Step 2: Configure .csproj**

We must instruct MSBuild to bundle these files when (and only when) we are publishing for Linux. Open InferenceApp.Core.csproj and add this ItemGroup.

*Why PreserveNewest?* This ensures that if you re-compile the libraries later, the new versions will overwrite the old ones in your build output automatically.

\<ItemGroup Condition="'$(RuntimeIdentifier)' \== 'linux-x64'"\>  
  \<Content Include="runtimes\\linux-x64\\native\\\*.so"\>  
    \<CopyToOutputDirectory\>PreserveNewest\</CopyToOutputDirectory\>  
    \<Link\>%(RecursiveDir)%(Filename)%(Extension)\</Link\>  
  \</Content\>  
\</ItemGroup\>

### **Step 3: Loading Logic (Recap)**

The standard C\# helper methods (like AppendExecutionProvider\_DML) are hardcoded to look for Windows DLLs. For ROCm, we must use the generic AppendExecutionProvider string method.

Ensure your OnnxInferenceEngine.cs contains this specific logic block:

if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))  
{  
    try
    {  
        // Tuning Options for AMD MIOpen  
        var rocmOptions \= new Dictionary\<string, string\>  
        {  
            { "device\_id", "0" }, // Use the first GPU found  
            { "miopen\_conv\_exhaustive\_search", "1" }, // Performance: Benchmark convolution algos upfront  
            { "arena\_extend\_strategy", "kSameAsRequested" } // Memory: Optimize VRAM allocation  
        };  

        // This call searches for 'libonnxruntime\_providers\_rocm.so' adjacent to the executable.  
        // If the file is missing (deployment error), it will throw immediately.  
        options.AppendExecutionProvider("ROCmExecutionProvider", rocmOptions);  
        \_logger.LogInformation("ROCm GPU Acceleration loaded successfully.");  
    }  
    catch (Exception ex)  
    {  
        // Graceful Fallback: If the GPU init fails (driver mismatch, etc.),  
        // fall back to CPU so the app doesn't crash entirely.  
        \_logger.LogError($"Failed to load ROCm provider: {ex.Message}. Using CPU fallback.");  
        options.AppendExecutionProvider\_CPU();  
    }  
}

### **Step 4: Deployment**

When you are ready to ship your application to the Linux machine, use the self-contained flag. This bundles the .NET Runtime, your Managed Code, and your Custom Native Assets into a single folder.

dotnet publish \-r linux-x64 \--self-contained

When you copy this folder to your Linux host (e.g., a Proxmox VM with GPU passthrough), you simply run ./InferenceApp.UI. The application will detect the Linux OS and load your custom .so files.

**Crucial Note:** The target machine **must** have the ROCm runtime drivers installed. Your custom .so files are just adapters; they depend on system libraries like libMIOpen.so and librocblas.so being present on the target OS. Ensure you run sudo apt install rocm-hip-runtime (or equivalent) on your deployment machine.
