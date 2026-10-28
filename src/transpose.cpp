#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

//
// BASELINE VERSION
//
// Simple matrix transpose.
//
// Reads are contiguous and therefore reasonably efficient.
//
// Writes, however, are strided:
//
//     out[x * H + y]
//
// Neighboring threads write to memory locations that are
// far apart, producing uncoalesced memory accesses.
//
__global__ void transpose_naive(
    const float* in,
    float* out,
    int W,
    int H)
{
    int x =
        blockIdx.x * blockDim.x + threadIdx.x;

    int y =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H)
    {
        //
        // Read:
        //   contiguous
        //
        // Write:
        //   strided / uncoalesced
        //
        out[x * H + y] =
            in[y * W + x];
    }
}

//
// OPTIMIZED VERSION
//
// Uses LDS (Local Data Share / shared memory)
// as a temporary staging buffer.
//
// Threads first load a tile of the matrix
// into shared memory, then write it back
// in transposed order.
//
// This converts the global-memory accesses
// into largely coalesced reads and writes.
//
#define TILE 16

__global__ void transpose_tiled(
    const float* in,
    float* out,
    int W,
    int H)
{
    //
    // Shared-memory tile.
    //
    // The extra column (+1) avoids
    // shared-memory bank conflicts when
    // threads access the tile transposed.
    //
    __shared__ float tile[TILE][TILE + 1];

    //
    // Coordinates in the input matrix.
    //
    int xIn =
        blockIdx.x * TILE + threadIdx.x;

    int yIn =
        blockIdx.y * TILE + threadIdx.y;

    //
    // Load a tile from global memory
    // into shared memory.
    //
    if (xIn < W && yIn < H)
    {
        tile[threadIdx.y][threadIdx.x] =
            in[yIn * W + xIn];
    }

    //
    // Wait until the entire tile has
    // been loaded.
    //
    __syncthreads();

    //
    // Coordinates in the output matrix.
    //
    int xOut =
        blockIdx.y * TILE + threadIdx.x;

    int yOut =
        blockIdx.x * TILE + threadIdx.y;

    //
    // Store the transposed tile.
    //
    // Accessing the tile through
    // [x][y] performs the transpose.
    //
    if (xOut < H && yOut < W)
    {
        out[yOut * H + xOut] =
            tile[threadIdx.x][threadIdx.y];
    }
}

//
// Benchmark helper.
//
// Runs the kernel multiple times and
// reports average execution time and
// effective memory bandwidth.
//
void run_kernel(
    const char* name,
    void(*kernel)(const float*,float*,int,int),
    float* d_in,
    float* d_out,
    int W,
    int H)
{
    dim3 block(TILE, TILE);

    dim3 grid(
        (W + TILE - 1) / TILE,
        (H + TILE - 1) / TILE);

    //
    // Warm-up launch.
    //
    // Removes one-time startup effects
    // from timing measurements.
    //
    kernel<<<grid, block>>>(
        d_in, d_out, W, H);

    hipDeviceSynchronize();

    hipEvent_t t0, t1;

    hipEventCreate(&t0);
    hipEventCreate(&t1);

    //
    // Timed runs.
    //
    hipEventRecord(t0);

    for (int i = 0; i < 50; i++)
    {
        kernel<<<grid, block>>>(
            d_in, d_out, W, H);
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
    // Effective bandwidth:
    //
    // Read matrix + write matrix.
    //
    double bw =
        2.0 * W * H * sizeof(float)
        / (ms * 1e-3)
        / 1e9;

    printf(
        "%-20s Time: %7.3f ms   Bandwidth: %6.1f GB/s\n",
        name,
        ms,
        bw);

    hipEventDestroy(t0);
    hipEventDestroy(t1);
}

int main()
{
    //
    // Matrix dimensions.
    //
    //int W = 4096;
    //int H = 4096;
    int W = 16384;
    int H = 16384;

    size_t bytes =
        W * H * sizeof(float);

    //
    // Host memory.
    //
    float* h_in =
        (float*)malloc(bytes);

    //
    // Initialize input matrix.
    //
    for (int i = 0; i < W * H; i++)
    {
        h_in[i] = (float)i;
    }

    //
    // Device memory.
    //
    float *d_in, *d_out;

    hipMalloc(&d_in, bytes);
    hipMalloc(&d_out, bytes);

    hipMemcpy(
        d_in,
        h_in,
        bytes,
        hipMemcpyHostToDevice);

    printf(
        "Matrix Transpose %dx%d\n\n",
        W,
        H);

    //
    // Compare naive and optimized versions.
    //
    run_kernel(
        "Naive (uncoalesced)",
        transpose_naive,
        d_in,
        d_out,
        W,
        H);

    run_kernel(
        "Tiled (LDS)",
        transpose_tiled,
        d_in,
        d_out,
        W,
        H);

    hipFree(d_in);
    hipFree(d_out);

    free(h_in);

    return 0;
}
