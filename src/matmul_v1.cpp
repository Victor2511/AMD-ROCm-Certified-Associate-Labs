#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>


// ============================================================
// Iteration 1:
// Improve coalescing and workgroup configuration
//
// Changes from baseline:
//   - Better thread-to-data mapping
//   - Better memory coalescing
//   - Larger workgroup: 32 -> 256 threads
//
// Still missing:
//   - LDS / shared memory tiling
// ============================================================

__global__ void matmul_coalesced(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // FIXED mapping:
    // neighboring threads now process neighboring columns

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];

        C[row * N + col] = sum;
    }
}


// CPU reference implementation

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
                sum += A[row * K + k] * B[k * N + col];

            C[row * N + col] = sum;
        }
    }
}


int main()
{
    int M = 1024;
    int N = 1024;
    int K = 1024;

    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    float* h_A   = (float*)malloc(sA);
    float* h_B   = (float*)malloc(sB);
    float* h_C   = (float*)malloc(sC);
    float* h_ref = (float*)malloc(sC);

    srand(42);

    for (int i = 0; i < M * K; i++)
        h_A[i] = (float)rand() / RAND_MAX;

    for (int i = 0; i < K * N; i++)
        h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;

    hipMalloc(&d_A, sA);
    hipMalloc(&d_B, sB);
    hipMalloc(&d_C, sC);

    hipMemcpy(d_A, h_A, sA, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, sB, hipMemcpyHostToDevice);

    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    // Warmup

    matmul_coalesced<<<grid, block>>>(
        d_A, d_B, d_C,
        M, N, K);

    hipDeviceSynchronize();

    // GPU timing

    hipEvent_t start, stop;

    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);

    for (int i = 0; i < 10; i++)
    {
        matmul_coalesced<<<grid, block>>>(
            d_A, d_B, d_C,
            M, N, K);
    }

    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;

    hipEventElapsedTime(&ms, start, stop);

    ms /= 10.0f;

    hipMemcpy(h_C, d_C, sC, hipMemcpyDeviceToHost);

    // Validation

    matmul_cpu(h_A, h_B, h_ref, M, N, K);

    float maxErr = 0.0f;

    for (int i = 0; i < M * N; i++)
        maxErr = fmax(maxErr, fabs(h_C[i] - h_ref[i]));

    double gflops =
        2.0 * M * N * K /
        (ms * 1e-3) /
        1e9;

    printf("Kernel:    COALESCED\n");

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

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return 0;
}
