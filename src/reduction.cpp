
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(cmd)                                   \
do {                                                     \
    hipError_t e = cmd;                                  \
    if (e != hipSuccess) {                               \
        printf("HIP error: %s\n",                        \
               hipGetErrorString(e));                    \
        return 1;                                        \
    }                                                    \
} while(0)

//
// Block reduction.
//
// Each block produces one partial sum.
//
__global__ void reduce_sum(
    const float* input,
    float* output,
    int N)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] =
        (gid < N) ? input[gid] : 0.0f;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main()
{
    const int N = 1 << 20;

    const int block = 256;

    float *h_in =
        new float[N];

    float sum_ref = 0.0f;

    for (int i = 0; i < N; i++)
    {
        h_in[i] = 1.0f;
        sum_ref += h_in[i];
    }

    float *d_in;
    float *d_tmp1;
    float *d_tmp2;

    HIP_CHECK(
        hipMalloc(&d_in,
                  N * sizeof(float)));

    HIP_CHECK(
        hipMalloc(&d_tmp1,
                  N * sizeof(float)));

    HIP_CHECK(
        hipMalloc(&d_tmp2,
                  N * sizeof(float)));

    HIP_CHECK(
        hipMemcpy(d_in,
                  h_in,
                  N * sizeof(float),
                  hipMemcpyHostToDevice));

    //
    // Multi-pass reduction.
    //
    int current_size = N;

    float* src = d_in;
    float* dst = d_tmp1;

    while (current_size > 1)
    {
        int grid =
            (current_size + block - 1)
            / block;

        reduce_sum<<<grid, block>>>(
            src,
            dst,
            current_size);

        HIP_CHECK(
            hipGetLastError());

        std::swap(src, dst);

        current_size = grid;
    }

    float result;

    HIP_CHECK(
        hipMemcpy(&result,
                  src,
                  sizeof(float),
                  hipMemcpyDeviceToHost));

    printf("\n=== HIP Reduction ===\n");
    printf("Elements : %d\n", N);
    printf("GPU Sum  : %.0f\n", result);
    printf("CPU Sum  : %.0f\n", sum_ref);
    printf("Status   : %s\n",
           (result == sum_ref)
           ? "PASS"
           : "FAIL");

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_tmp1));
    HIP_CHECK(hipFree(d_tmp2));

    delete[] h_in;

    return 0;
}
