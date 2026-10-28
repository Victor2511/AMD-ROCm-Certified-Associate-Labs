#include <hip/hip_runtime.h>
#include <cstdio>

//
// SAXPY = Single-Precision A*X Plus Y
//
// One of the classic BLAS operations:
//
//     y = a * x + y
//
// The kernel performs very little computation
// per element and is therefore typically
// memory-bandwidth limited.
//
__global__ void saxpy(
    float a,
    const float* x,
    float* y,
    int N)
{
    int i =
        blockDim.x * blockIdx.x +
        threadIdx.x;

    if (i < N)
    {
        y[i] =
            a * x[i] + y[i];
    }
}

int main()
{
    //
    // Number of elements.
    //
    // 2^24 = 16,777,216 floats
    //
    int N = 1 << 24;

    size_t bytes =
        N * sizeof(float);

    //
    // Device memory.
    //
    float *d_x, *d_y;

    hipMalloc(&d_x, bytes);
    hipMalloc(&d_y, bytes);

    //
    // Initialize vectors.
    //
    // Exact values are unimportant;
    // we only care about performance.
    //
    hipMemset(d_x, 1, bytes);
    hipMemset(d_y, 1, bytes);

    //
    // Candidate block sizes.
    // For Instinct MI300X
    // 64 threads  = 1 wavefront
    // 128 threads = 2 wavefronts
    // 256 threads = 4 wavefronts
    // 512 threads = 8 wavefronts
    // 1024 threads = 16 wavefronts
    //
    int sizes[] =
    {
        64,
        128,
        256,
        512,
        1024
    };

    printf(
        "%-15s %-12s %-12s\n",
        "Block Size",
        "Time (ms)",
        "BW (GB/s)");

    printf(
        "%-15s %-12s %-12s\n",
        "----------",
        "---------",
        "---------");

    //
    // Sweep through several
    // workgroup sizes and compare
    // performance.
    //
    for (int b = 0; b < 5; b++)
    {
        int bs =
            sizes[b];

        int grid =
            (N + bs - 1) / bs;

        //
        // Warm-up launch.
        //
        saxpy<<<grid, bs>>>(
            2.0f,
            d_x,
            d_y,
            N);

        hipDeviceSynchronize();

        hipEvent_t t0, t1;

        hipEventCreate(&t0);
        hipEventCreate(&t1);

        //
        // Timed launches.
        //
        hipEventRecord(t0);

        for (int i = 0; i < 50; i++)
        {
            saxpy<<<grid, bs>>>(
                2.0f,
                d_x,
                d_y,
                N);
        }

        hipEventRecord(t1);
        hipEventSynchronize(t1);

        float ms;

        hipEventElapsedTime(
            &ms,
            t0,
            t1);

        ms /= 50;

        //
        // Effective bandwidth.
        //
        // For each element:
        //
        //   Read x[i]
        //   Read y[i]
        //   Write y[i]
        //
        // Total traffic:
        //
        //   3 × N × sizeof(float)
        //
        double bw =
            3.0 * N * sizeof(float)
            / (ms * 1e-3)
            / 1e9;

        printf(
            "%-15d %-12.3f %-12.1f\n",
            bs,
            ms,
            bw);

        hipEventDestroy(t0);
        hipEventDestroy(t1);
    }

    hipFree(d_x);
    hipFree(d_y);

    return 0;
}
