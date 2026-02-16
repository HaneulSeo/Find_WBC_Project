# WBC Detector (CapStone Project)

White Blood Cell (WBC) Detection project utilizing **RT-DETR** with **LibTorch (C++)** for high-performance inference and Python for training/benchmarking.

## 📁 Repository Structure

- **C++ Inference**: High-performance deployment using LibTorch & OpenCV (`main.cpp`).
- **Python Setup**: Environment setup for training, exporting, and PyTorch benchmarks.

---

## 🛠️ Step 0: Model Export (RT-DETR to TorchScript)

Before running the C++ application, you need to convert the trained RT-DETR model to TorchScript format.

### 1. Install Ultralytics
```bash
pip install ultralytics
```

### 2. Export Command
Run the following command to export the model with **FP16 (Half-Precision)** optimization.

```bash
# For generic export (uses GPU if available)
yolo export model=rt_detr.pt format=torchscript device=0 half=True
```
* `model=rt_detr.pt`: Path to your trained weights.
* `half=True`: **Crucial** for the C++ code (FP16 inference).
* `device=0`: Use GPU for export. (Use `device=cpu` if you don't have a GPU).

> **Action:** Move the generated `rt_detr.torchscript` file to the root of this repository.

---

## 🚀 Step 1: C++ Environment Setup

### 1. Install Dependencies (OpenCV)
You need OpenCV installed on your system.

* **macOS (Homebrew):**
    ```bash
    brew install opencv
    ```
* **Ubuntu/Linux:**
    ```bash
    sudo apt-get update && sudo apt-get install libopencv-dev
    ```

### 2. Download & Install LibTorch (Core Step)

You must download the **C++ library** manually. Choose your OS below.

#### 🍎 Option A: macOS (Apple Silicon - M1/M2/M3)
Use the specific version utilized in this project (v2.5.1).

```bash
# 1. Download LibTorch (CPU/MPS support)
curl -O [https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip](https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip)

# 2. Unzip
unzip libtorch-macos-arm64-2.5.1.zip

# 3. Clean up zip file
rm libtorch-macos-arm64-2.5.1.zip
```

#### 🟢 Option B: Linux / Windows (NVIDIA CUDA)
If you want to run this on an NVIDIA GPU (e.g., RTX 3060, A100), you must download the **CUDA version** of LibTorch.

1.  Go to [PyTorch Get Started](https://pytorch.org/get-started/locally/).
2.  Select: **Stable** -> **Linux** -> **LibTorch** -> **C++** -> **CUDA 11.8 (or 12.x)**.
3.  **Download "Pre-cxx11 ABI" version** (Recommended for compatibility).
4.  Unzip it so the `libtorch` folder is in the project root.

---

## 🏗️ Step 2: Build Application

```bash
# 1. Create build directory
mkdir build
cd build

# 2. Configure with CMake
cmake ..

# 3. Build executable
make
```

---

## 🏃 Step 3: Run Inference

Ensure the model and test image are in the same folder as the executable (or copy them).

```bash
# Copy assets to build folder
cp ../rt_detr.torchscript .
cp ../img_294.png .

# Run
./LibTorchApp
```

---

## 💡 How to use CUDA (NVIDIA GPU)?

By default, the code is optimized for macOS (MPS). To use NVIDIA CUDA, follow these changes:

**1. Download the correct LibTorch:**
Follow "Option B" in Step 1 above.

**2. Modify `main.cpp`:**
Change the device selection logic in `main.cpp`:

```cpp
// [Original] macOS MPS focus
torch::Device device(torch::kCPU);
if (torch::hasMPS()) {
    device = torch::Device(torch::kMPS);
}

// [Change to] CUDA focus
torch::Device device(torch::kCPU);
if (torch::cuda::is_available()) {
    std::cout << ">>> CUDA (NVIDIA GPU) Detected!" << std::endl;
    device = torch::Device(torch::kCUDA);
} else if (torch::hasMPS()) {
    device = torch::Device(torch::kMPS);
}
```

**3. Re-build:**
```bash
cd build
make clean
cmake ..
make
```

---

## 🐍 Python Environment (Training & Dev)

If you want to train the model or run Python-based benchmarks:

```bash
# Setup Conda
conda create -n wbc python=3.11 pip
conda activate wbc

# Install PyTorch (Choose based on your hardware)
# For DGX/CUDA:
pip install -U torch torchvision --index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)
```

---

## 📊 Benchmark Results

- [MPS (Apple Silicon)](./Benchmark_inference.ipynb)
- [NVIDIA A100](./Benchmark_inference_A100.ipynb)
- [NVIDIA RTX 5090](./Benchmark_inference_RTX-5090.ipynb)
- [NVIDIA RTX Pro 6000 Workstation](./Benchmark_inference_RTX-PRO-6000-WKS.ipynb)
- [NVIDIA H100 PCIe](./Benchmark_inference_H100-PCIe-try1.ipynb)
- [DGX Spark Cluster](./Benchmark_inference_DGX-SPARK.ipynb)
