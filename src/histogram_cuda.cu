// histogram_cuda.cu  — ascending bin counts
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_BINS 256

#define CUDA_CHECK(cmd)                                                  \
  do {                                                                   \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s:%d — %s\n",                        \
              __FILE__, __LINE__, cudaGetErrorString(e));                \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void histogram(const unsigned char* data, int* bins, int N) {
    __shared__ int local_bins[NUM_BINS];

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
        local_bins[i] = 0;
    __syncthreads();

    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
        atomicAdd(&local_bins[data[i]], 1);
    __syncthreads();

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
        atomicAdd(&bins[i], local_bins[i]);
}

int main() {
    // N = 0+1+2+...+255 = 255*256/2 = 32640
    const int N = (NUM_BINS - 1) * NUM_BINS / 2;

    // Fill: value v appears exactly v times
    unsigned char* h_data = (unsigned char*)malloc(N * sizeof(unsigned char));
    int pos = 0;
    for (int v = 0; v < NUM_BINS; v++)
        for (int c = 0; c < v; c++)
            h_data[pos++] = (unsigned char)v;

    unsigned char* d_data;
    int*           d_bins;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_bins, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    histogram<<<blocks, threads>>>(d_data, d_bins, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_bins[NUM_BINS] = {0};
    CUDA_CHECK(cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Verify: bin[v] must equal v
    int errors = 0;
    for (int v = 0; v < NUM_BINS; v++) {
        if (h_bins[v] != v) {
            fprintf(stderr, "MISMATCH bin[%d]: got %d, expected %d\n",
                    v, h_bins[v], v);
            errors++;
        }
    }

    if (errors == 0) {
        printf("bins[0]=%d, bins[1]=%d, bins[2]=%d, ... bins[255]=%d\n",
               h_bins[0], h_bins[1], h_bins[2], h_bins[255]);
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_bins));
    free(h_data);
    return errors > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
