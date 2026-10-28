#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>


// ============================================================
// BASELINE: Naive Matrix Multiply
//
// Deliberately non-optimized implementation.
//
// Performance issues:
//   - Poor thread-to-data mapping
//   - No tiling
//   - No shared memory (LDS) usage
//
// Matrix sizes:
//
//   A : M x K
//   B : K x N
//   C : M x N
//
// ============================================================

__global__ void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // Deliberately poor thread mapping:
    // neighboring threads do not access neighboring
    // output elements efficiently.

    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; k++)
        {
            sum +=
                A[row * K + k] *
                B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


// ============================================================
// CPU reference implementation
// Used for correctness validation
// ============================================================

void matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; k++)
            {
                sum +=
                    A[row * K + k] *
                    B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}


int main(int argc, char** argv)
{
    int M = 1024;
    int N = 1024;
    int K = 1024;

    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    // Host memory

    float* h_A   = (float*)malloc(sA);
    float* h_B   = (float*)malloc(sB);
    float* h_C   = (float*)malloc(sC);
    float* h_ref = (float*)malloc(sC);

    // Initialize matrices with random values

    srand(42);

    for (int i = 0; i < M * K; i++)
        h_A[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < K * N; i++)
        h_B[i] = (float)rand() / RAND_MAX;

    // Device memory

    float* d_A;
    float* d_B;
    float* d_C;

    hipMalloc(&d_A, sA);
    hipMalloc(&d_B, sB);
    hipMalloc(&d_C, sC);

    hipMemcpy(d_A, h_A, sA, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, sB, hipMemcpyHostToDevice);

    dim3 block(16, 16); // Use 256 threads

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    // Warmup launch

    matmul_naive<<<grid, block>>>(
        d_A, d_B, d_C,
        M, N, K);

    hipDeviceSynchronize();

    // GPU timing using HIP events

    hipEvent_t start, stop;

    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);

    for (int i = 0; i < 10; i++)
    {
        matmul_naive<<<grid, block>>>(
            d_A, d_B, d_C,
            M, N, K);
    }

    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;

    hipEventElapsedTime(&ms, start, stop);

    ms /= 10.0f;

    // Copy result back to CPU

    hipMemcpy(
        h_C,
        d_C,
        sC,
        hipMemcpyDeviceToHost);

    // Correctness validation

    matmul_cpu(
        h_A,
        h_B,
        h_ref,
        M, N, K);

    float maxErr = 0.0f;

    for (int i = 0; i < M * N; i++)
    {
        maxErr = fmax(
            maxErr,
            fabs(h_C[i] - h_ref[i]));
    }

    // GEMM performs:
    //
    //   2 * M * N * K floating-point operations

    double gflops =
        2.0 * M * N * K /
        (ms * 1e-3) /
        1e9;

    // Results

    printf("Kernel:    BASELINE (naive)\n");

    printf(
        "Block:     %dx%d = %d threads\n",
        block.x,
        block.y,
        block.x * block.y);

    printf("Time:      %.3f ms\n", ms);

    printf("GFLOPS:    %.1f\n", gflops);

    printf("Max error: %e\n", maxErr);

    printf(
        "Status:    %s\n",
        maxErr < 1e-2 ? "PASS" : "FAIL");

    // Cleanup

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return 0;
}
