#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

// Compute-heavy kernel: repeated FMA operations
__global__ void compute_kernel(float* out, int N, int n_ops) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float val = 1.0f;
        for (int j = 0; j < n_ops; j++) {
            val = val * 1.00001f + 0.00001f;  // FMA
        }
        out[i] = val;
    }
}

int main() {
    printf("\n=== Compute Throughput Benchmark ===\n\n");

    int N = 1 << 22;  // 4M threads
    size_t bytes = N * sizeof(float);
    float *d_out;
    hipError_t hipErr;

    hipErr = hipMalloc(&d_out, bytes);

    int block_sizes[] = {64, 128, 256, 512, 1024};
    int n_blocks = 5;
    int n_ops = 1000;  // FMA ops per thread
    int n_iters = 20;

    for (int b = 0; b < n_blocks; b++) {
        int block = block_sizes[b];
        int grid = (N + block - 1) / block;

        // Warmup
        compute_kernel<<<grid, block>>>(d_out, N, n_ops);
        hipErr = hipDeviceSynchronize();

        // Timed
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_iters; i++)
            compute_kernel<<<grid, block>>>(d_out, N, n_ops);
        hipErr = hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        // 2 FLOP per FMA (multiply + add)
        double total_flops = 2.0 * n_ops * N * n_iters;
        double gflops = total_flops / elapsed / 1e9;

        printf("  Block=%4d  Grid=%6d  GFLOPS=%8.1f  time/iter=%.4f ms\n",
               block, grid, gflops, elapsed / n_iters * 1000);
    }

    hipErr = hipFree(d_out);
    return 0;
}
