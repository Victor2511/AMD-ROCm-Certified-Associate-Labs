#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

// ============================================================
// STREAM TRIAD MEMORY BANDWIDTH TEST
//
// Computes:
//
//     C[i] = scalar * A[i] + B[i]
//
// Each element performs:
//   - 2 global memory reads  (A and B)
//   - 1 global memory write  (C)
//
// This benchmark demonstrates:
//
//   1. Correct GPU indexing
//   2. Memory bandwidth
//   3. Coalesced vs non-coalesced access
//   4. Common GPU programming bugs
// ============================================================


// ============================================================
// MODE SELECTION
// ============================================================
//
// MODE_BASELINE
//   Ideal case.
//   Neighboring threads access neighboring memory.
//
// MODE_STRIDE
//   Poor memory coalescing.
//   Neighboring threads access distant locations.
//
// MODE_RANDOM
//   Pseudo-random access pattern.
//   Poor cache locality.
//
// MODE_INDEX_BUG
//   Incorrect indexing.
//   Ignores blockIdx.x so all blocks overwrite
//   the same memory region.
// ============================================================

#define MODE_BASELINE   0
#define MODE_STRIDE     1
#define MODE_RANDOM     2
#define MODE_INDEX_BUG  3

#ifndef MODE
#define MODE MODE_BASELINE
//#define MODE MODE_STRIDE
//#define MODE MODE_RANDOM
//#define MODE MODE_INDEX_BUG
#endif

static const char* spModes[] = 
{
    "MODE_BASELINE",
    "MODE_STRIDE",
    "MODE_RANDOM",
    "MODE_INDEX_BUG"
};

// CPU (host) memory
static float* h_A;
static float* h_B;
static float* h_C_ref;
static float* h_C_gpu;

// GPU (device) memory
static float* d_A;
static float* d_B;
static float* d_C;


// ============================================================
// GPU Kernel
// ============================================================

__global__ void stream_triad(
    const float* A,
    const float* B,
    float* C,
    float scalar,
    int N)
{
    int tid = threadIdx.x;

    // Global thread index
    int i = blockIdx.x * blockDim.x + tid;

#if MODE == MODE_BASELINE

    // Sequential access:
    // neighboring threads access neighboring elements
    int idx = i;

#elif MODE == MODE_STRIDE

    // Poor memory access pattern:
    // neighboring threads access distant elements
    int idx = ((i * 65537) + 17) & (N - 1);

#elif MODE == MODE_RANDOM

    // Pseudo-random access pattern
    int idx = ((i << 5) + i) & (N - 1);

#elif MODE == MODE_INDEX_BUG

    // WRONG:
    // ignores blockIdx.x
    // all blocks overwrite the same region
    int idx = tid;

#endif    

    if (idx < N)
    {
        C[idx] = scalar * A[idx] + B[idx];
    }
}


// ============================================================
// Initialize host/device memory
// ============================================================

void init(int N, float scalar)
{
    hipError_t hipErr;

    size_t bytes = N * sizeof(float);

    // Allocate CPU memory
    h_A     = (float*)malloc(bytes);
    h_B     = (float*)malloc(bytes);
    h_C_ref = (float*)malloc(bytes);
    h_C_gpu = (float*)malloc(bytes);

    // Allocate GPU memory
    hipErr = hipMalloc(&d_A, bytes);
    hipErr = hipMalloc(&d_B, bytes);
    hipErr = hipMalloc(&d_C, bytes);

    // Initialize input vectors
    for (int i = 0; i < N; i++)
    {
        // (i % 1000) prevents floating-point errors when N is very lagre here
        h_A[i] = (i % 1000) * 0.5f;
        h_B[i] = (i % 1000) * 2.0f;
    }

    // Copy input vectors to GPU
    hipErr = hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
    hipErr = hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);

    // CPU reference computation
    for (int i = 0; i < N; i++)
    {
        h_C_ref[i] = scalar * h_A[i] + h_B[i];
    }
}


// ============================================================
// Cleanup
// ============================================================

void cleanup()
{
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_gpu);

    hipError_t hipErr = hipFree(d_A);
    hipErr = hipFree(d_B);
    hipErr = hipFree(d_C);
}


// ============================================================
// Validation
// ============================================================

int check_kernel(int N, int block_size)
{
    size_t bytes = N * sizeof(float);

    int grid_size = (N + block_size - 1) / block_size;

    // Launch kernel
    stream_triad<<<grid_size, block_size>>>(
        d_A,
        d_B,
        d_C,
        3.0f,
        N);

    // Check launch errors
    hipError_t hipErr = hipGetLastError();

    if (hipErr != hipSuccess)
    {
        printf("Kernel launch failed: %s\n",
               hipGetErrorString(hipErr));
        return -1;
    }

    hipErr = hipDeviceSynchronize();

    // Copy result back to CPU
    hipErr = hipMemcpy(h_C_gpu, d_C, bytes, hipMemcpyDeviceToHost);

    const float epsilon = 1e-5f;

    float max_error = 0.0f;
    int max_error_index = 0;

    int mismatch_found = 0;

    for (int i = 0; i < N; i++)
    {
        float error = fabs(h_C_gpu[i] - h_C_ref[i]);

        if (error > max_error)
        {
            max_error = error;
            max_error_index = i;
        }

        if (error > epsilon)
        {
            printf("Validation FAILED\n");

            printf("First mismatch at index: %d\n", i);

            printf("CPU value: %f\n",
                   h_C_ref[i]);

            printf("GPU value: %f\n",
                   h_C_gpu[i]);

            printf("Error: %f\n", error);

            mismatch_found = 1;

            break;
        }
    }

    if (!mismatch_found)
    {
        printf("Validation PASSED\n");

        printf("Maximum error: %f at index %d\n",
               max_error,
               max_error_index);

        return 0;
    }

    return -1;
}


// ============================================================
// Benchmark
// ============================================================

double benchmark_kernel(
    int N,
    int block_size,
    int n_iters)
{
    int grid_size =
        (N + block_size - 1) / block_size;

    // Warmup
    for (int i = 0; i < 3; i++)
    {
        stream_triad<<<grid_size, block_size>>>(
            d_A,
            d_B,
            d_C,
            3.0f,
            N);
    }

    hipError_t hipErr = hipDeviceSynchronize();

    // Timed runs
    // NOTE:
    // This is a simplified benchmark for educational purposes.
    // Accurate GPU timing is more complicated and will be
    // discussed later.

    auto start =
        std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_iters; i++)
    {
        stream_triad<<<grid_size, block_size>>>(
            d_A,
            d_B,
            d_C,
            3.0f,
            N);
    }

    hipErr = hipDeviceSynchronize();

    auto end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli>
        diff = end - start;

    double elapsed_ms = diff.count();

    // 2 reads + 1 write per element
    double total_bytes =
        (double)N *
        sizeof(float) *
        3 *
        n_iters;

    // Convert to GB/s
    double bw =
        ((total_bytes / 1e9) / elapsed_ms) * 1e3;

    printf(
        "  %-12s N=%10d  BW=%8.1f GB/s  time=%.4f ms\n",
        "stream_triad",
        N,
        bw,
        elapsed_ms / n_iters);

    return bw;
}


// ============================================================
// Main
// ============================================================

int main()
{
    printf("\n=== Memory Bandwidth Benchmark: MODE : %s ===\n\n", spModes[MODE]);

    int sizes[] =
    {
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 24,
        1 << 26
    };

    int n_sizes = 5;
    int n_iters = 200;
    int block_size = 256;

    // Allocate maximum size once
    init(1 << 26, 3.0f);

    for (int s = 0; s < n_sizes; s++)
    {
        int N = sizes[s];

        printf(
            "Size: %d elements (%.1f MB)\n",
            N,
            (double)N * sizeof(float) / 1e6);

        check_kernel(N, block_size);

        benchmark_kernel(
            N,
            block_size,
            n_iters);

        printf("\n");
    }

    cleanup();

    return 0;
}
