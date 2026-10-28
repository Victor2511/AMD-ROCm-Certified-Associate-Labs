#include <hip/hip_runtime.h>
#include <stdio.h>
#define HIP_CHECK(cmd) do {                              \
    hipError_t e = (cmd);                                \
    if (e != hipSuccess) {                               \
        fprintf(stderr, "[HIP_CHECK] %s:%d  %s\n",        \
            __FILE__, __LINE__, hipGetErrorString(e));    \
        exit(EXIT_FAILURE); } } while (0)
__global__ void vec_add(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
int main() {
    const int N  = 1 << 20;            // 1,048,576 elements
    size_t bytes = N * sizeof(float);  // 4 MB per vector
    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    for (int i = 0; i < N; i++) { h_A[i] = i * 1.0f; h_B[i] = i * 2.0f; }
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));
    HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));
    vec_add<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    HIP_CHECK(hipGetLastError());       // catch bad launch config
    HIP_CHECK(hipDeviceSynchronize());  // catch runtime kernel errors
    HIP_CHECK(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));
    int errs = 0;
    for (int i = 0; i < N; i++)
        if (h_C[i] != i * 3.0f) errs++;
    printf("Vector Add: N=%d, Errors=%d - %s\n", N, errs, errs ? "FAIL":"PASS");
    for (int i = 0; i < 5; i++) printf("  C[%d] = %.1f\n", i, h_C[i]);
    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C));
    delete[] h_A; delete[] h_B; delete[] h_C;
    return errs;
}
