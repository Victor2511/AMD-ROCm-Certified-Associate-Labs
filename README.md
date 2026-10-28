# 🚀 AMD ROCm™ AI & GPU Programming Certification Practice Labs

[![Certification - AMD ROCm Certified Associate](https://img.shields.io/badge/Certification-AMD_ROCm™_Certified_Associate-E21B23?style=for-the-badge&logo=amd&logoColor=white)](https://credly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![ROCm Version](https://img.shields.io/badge/ROCm-7.2+-0052CC?style=for-the-badge&logo=linux&logoColor=white)](https://rocm.docs.amd.com/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.10%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository contains the complete hands-on practical notebooks, C++/HIP source code, and microbenchmarks developed during the **AMD AI Academy Certification Program**. 

The code demonstrates advanced performance tuning, memory coalescing, shared memory (LDS) tiling, mixed-precision matrix multiplication (GEMM), CUDA-to-HIP porting, and PyTorch execution on **AMD Instinct™ GPUs (MI300X)**.

---

## 📜 Certification Credential

- **Certification:** AMD ROCm™ Certified Associate
- **Issued To:** Victor Rojas
- **Issuer:** AMD AI Academy / Lablab.ai
- **Valid Through:** October 28, 2028
- **Credential ID:** `12283899-7cdb-4600-94f2-3491baee3add`

<p align="center">
  <img src="https://raw.githubusercontent.com/victor-rojas/AMD-ROCm-Certified-Associate-Labs/main/assets/AMD_ROCm_Certified_Associate.png" alt="AMD ROCm Certified Associate Badge" width="220"/>
</p>

---

## 🗂️ Repository Structure

```text
.
├── notebooks/
│   ├── Lab_3_Libraries_PyTorch.ipynb
│   ├── Lab_4_HIP_Programming_Porting.ipynb
│   ├── Lab_5-1_Performance_Optimization.ipynb
│   ├── Lab_5-2_Architecture_Microbenchmarks.ipynb
│   └── Lab_5-3_Kernel_Tuning.ipynb
├── src/
│   ├── compute_bench.cpp
│   ├── histogram_cuda.cu
│   ├── histogram_hip.cpp
│   ├── matmul.cpp
│   ├── matmul_v1.cpp
│   ├── matmul_v2.cpp
│   ├── mem_bandwidth.cpp
│   ├── reduction.cpp
│   ├── rocblas_gemm.cpp
│   ├── sweep.cpp
│   ├── transpose.cpp
│   └── vec_add_hip.cpp
├── assets/
│   ├── lab5_2_results.png
│   └── AMD_ROCm_Certified_Associate.png
├── README.md
└── LICENSE
```

---

## 📚 Overview of Practical Labs

### 🧪 [Lab 3 — ROCm Libraries, PyTorch & AI Frameworks](./notebooks/Lab_3_Libraries_PyTorch.ipynb)
- **Direct Library Invocations:** Benchmarking high-performance BLAS operations with `rocblas_sgemm()`, reaching **>66 TFLOPS** on MI300X.
- **Deep Learning Primitives:** Evaluating MIOpen 2D convolutions and observing automatic kernel search/autotuning caching.
- **Signal Processing:** Executing round-trip Fast Fourier Transforms (FFT -> IFFT) using `rocFFT` with verification.
- **End-to-End PyTorch Training:** Training ResNet-18 on CIFAR-10 with multi-worker GPU data loading and mixed-precision optimization.

### 🔄 [Lab 4 — HIP Programming & Porting CUDA](./notebooks/Lab_4_HIP_Programming_Porting.ipynb)
- **HIP Kernels from Scratch:** Implementing vector-add with global thread indexing (`blockDim.x * blockIdx.x + threadIdx.x`).
- **CUDA to HIP Porting:** Automated translation of CUDA code using `hipify-perl` and `hipify-clang`.
- **Wavefront-Aware Reductions:** Implementing portable parallel reduction algorithms avoiding hardcoded warp size assumptions (accounting for 64-thread wavefronts).

### ⚡ [Lab 5-1 — Performance Optimization](./notebooks/Lab_5-1_Performance_Optimization.ipynb)
- **Uncoalesced vs. Coalesced Memory:** Resolving strided memory bottlenecks in matrix transposition using Local Data Share (LDS).
- **Workgroup Size Sweeps:** Sweeping thread block sizes (64 to 1024) on memory-bound kernels (SAXPY) to analyze GPU occupancy vs. actual throughput.
- **Mixed-Precision Computing:** Benchmarking FP32, FP16, and BF16 GEMM execution in PyTorch, demonstrating **~3.8x speedups** using reduced precision.

### 🔬 [Lab 5-2 — GPU Architecture Fundamentals: Microbenchmarks](./notebooks/Lab_5-2_Architecture_Microbenchmarks.ipynb)
- **Memory Hierarchy Profiling:** Stream Triad memory bandwidth benchmarking across L1/L2/Infinity Cache and HBM3.
- **Compute vs. Memory Limits:** Measuring peak FMA GFLOPS and classifying workloads with the **Roofline Model**.
- **Profiling Tools:** Hardware counter collection using `rocprofv3` and visual throughput plotting.

### 🛠️ [Lab 5-3 — Kernel Tuning (Matrix Multiplication)](./notebooks/Lab_5-3_Kernel_Tuning.ipynb)
An iterative optimization study converting a naive, suboptimal matrix multiplication kernel into a highly efficient GPU routine:

| Optimization Step | Technique | Execution Time | Performance | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Naive global memory accesses | 2.285 ms | 939.8 GFLOPS | 1.0x |
| **Iteration 1** | Memory Coalescing (Re-indexing) | 0.528 ms | 4,065.6 GFLOPS | 4.3x |
| **Iteration 2** | + LDS (Shared Memory) Tiling | **0.161 ms** | **13,339.5 GFLOPS** | **14.2x** |

---

## 📊 Key Benchmark Results

### Matrix Transpose: Naive vs. LDS Tiled (16384x16384)
- **Naive (Uncoalesced Writes):** 1.518 ms | 1,415.0 GB/s
- **Tiled (LDS Coalesced):** 0.949 ms | **2,263.5 GB/s**

### Mixed-Precision GEMM (2048x2048x2048)
- **FP32:** 0.178 ms | 96,667.1 GFLOPS
- **FP16:** 0.050 ms | 343,549.8 GFLOPS (**3.56x speedup**)
- **BF16:** 0.047 ms | **362,033.1 GFLOPS** (**3.78x speedup**)

---

## 🛠️ Environment & Prerequisites

To execute these notebooks, use the official AMD ROCm Docker container or an environment equipped with ROCm 6.x/7.x:

```bash
# Example ROCm Docker launch command
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  --shm-size 8g \
  rocm/pytorch:latest-py3.12
```

### Verification Command
```python
import torch
print(f"HIP Available: {torch.cuda.is_available()}")
print(f"Device Name:   {torch.cuda.get_device_name(0)}")
print(f"HIP Version:   {torch.version.hip}")
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
