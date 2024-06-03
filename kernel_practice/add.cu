#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__
void init_kernel(long long int n, long long int *y)
{
    long long int global_idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (global_idx<n){
        y[global_idx] = global_idx;
    }
}

__global__
void kernel2(long long int n, long long int *y, unsigned long long int *sum)
{
    long long int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    long long int val = 0;

    while (global_idx < n) {
        val += static_cast<unsigned long long int>(y[global_idx]);
        global_idx += gridDim.x * blockDim.x;
    }

    atomicAdd(sum, static_cast<unsigned long long int>(val));

}

// __global__
// void kernel3(long long int n, long long int *y, unsigned long long int *sum) {
//     // Declare shared memory for warp reduction results
//     __shared__ unsigned long long int sdata[32]; // Each warp will have one element in shared memory

//     // Thread and block indices
//     int tid = threadIdx.x;
//     long long int idx = threadIdx.x + blockDim.x * blockIdx.x;

//     // Initialize value
//     unsigned long long int val = 0;

//     // Mask for shuffle operations
//     unsigned mask = 0xFFFFFFFF;

//     // Lane and warp IDs
//     int lane = tid % warpSize;
//     int warpID = tid / warpSize;

//     // Grid stride loop to load data
//     while (idx < n) {
//         val += static_cast<unsigned long long int>(y[idx]);
//         idx += gridDim.x * blockDim.x;
//     }

//     // First warp shuffle reduction
//     for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
//         val += __shfl_down_sync(mask, val, offset);
//     }

//     // Store the result of each warp in shared memory
//     if (lane == 0) sdata[warpID] = val;

//     // Synchronize threads to ensure all warp results are written to shared memory
//     __syncthreads();

//     // Final reduction within warp 0
//     if (warpID == 0) {
//         // Reload val from shared memory if warp existed
//         val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;

//         // Final warp-shuffle reduction
//         for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
//             val += __shfl_down_sync(mask, val, offset);
//         }

//         // The first thread writes the result to global memory
//         if (tid == 0) atomicAdd(sum, val);
//     }
// }

__device__ void warpReduce(volatile long long int *sdata, unsigned int tid) {
    unsigned int blockSize = blockDim.x;

    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void kernel3(long long int n, long long int *g_idata, unsigned long long int *g_odata) {
    extern __shared__ long long int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned long long int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned long long int gridSize = blockSize * 2 * gridDim.x;

    // Initialize shared memory
    sdata[tid] = 0;

    // Grid stride loop
    while (i < n) {
        sdata[tid] += g_idata[i];
        if (i + blockSize < n) {
            sdata[tid] += g_idata[i + blockSize];
        }
        i += gridSize;
    }

    __syncthreads();

    // Reduce in shared memory
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (blockSize >= 64) { if (tid < 32) { sdata[tid] += sdata[tid + 32]; } __syncthreads(); }

    if (tid < 32) warpReduce(sdata, tid);

    // Write the result for this block to global memory
    if (tid == 0) atomicAdd(g_odata, sdata[0]);
}


void measureKernelExecutionTime(void (*kernel)(long long int, long long int*, unsigned long long int*), int numBlocks, int blockSize, long long int N, long long int *y, unsigned long long int *sum, const char* kernelName) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    *sum = 0;  // Reset sum

    // Record the start event
    cudaEventRecord(start);

    // Run the kernel on the GPU
    kernel<<<numBlocks, blockSize>>>(N, y, sum);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the sum and the elapsed time
    std::cout << kernelName << " - Sum: " << *sum << ", Time elapsed: " << milliseconds << " ms" << std::endl;

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void measureInit(void (*kernel)(long long int, long long int*), long long int numBlocks, int blockSize, long long int N, long long int *y, unsigned long long int *sum, const char* kernelName) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    *sum = 0;  // Reset sum

    // Record the start event
    cudaEventRecord(start);

    // Run the kernel on the GPU
    kernel<<<numBlocks, blockSize>>>(N, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the sum and the elapsed time
    std::cout << kernelName << " - Sum: " << *sum << ", Time elapsed: " << milliseconds << " ms" << std::endl;

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(void)
{
    long long int N = 1LL << 30;
    long long int *y;
    unsigned long long int *sum;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&y, N * sizeof(long long int));
    cudaMallocManaged(&sum, sizeof(unsigned long long int));
    
    // Initialize y array on the host
    // for (long long int i = 0; i < N; i++) {
    //     y[i] = i;
    // }
    // init_kernel<<<(N+1023)/1024, 1024>>>(N, y);
    measureInit(init_kernel, (N+1023)/1024, 1024, N, y, sum, "INIT");


    // Kernel launch parameters
    int blockSize = 1024;
    int numBlocks = 256;

    // Measure time for kernel2 (atomicAdd)
    measureKernelExecutionTime(kernel2, numBlocks, blockSize, N, y, sum, "atomicAdd");
    unsigned long long int temp = *sum;
    *sum = 0;

    // Measure time for kernel3 (warpShuffle)
    measureKernelExecutionTime(kernel3, numBlocks, blockSize, N, y, sum, "warpShuffle");

    if (temp == *sum) {
        cout << "EQUAL" << endl;
    } else {
        cout << "NOT EQUAL" << endl;
    }

    *sum = 0;
    measureKernelExecutionTime(kernel2, numBlocks, blockSize, N, y, sum, "atomicAdd");
    temp = *sum;
    *sum = 0;
    measureKernelExecutionTime(kernel3, numBlocks, blockSize, N, y, sum, "warpShuffle");

    if (temp == *sum) {
        cout << "EQUAL" << endl;
    } else {
        cout << "NOT EQUAL" << endl;
    }

    *sum = 0;
    measureKernelExecutionTime(kernel3, numBlocks, blockSize, N, y, sum, "warpShuffle");
    temp = *sum;
    *sum = 0;
    measureKernelExecutionTime(kernel2, numBlocks, blockSize, N, y, sum, "atomicAdd");

    if (temp == *sum) {
        cout << "EQUAL" << endl;
    } else {
        cout << "NOT EQUAL" << endl;
    }

    // Free allocated memory
    cudaFree(y);
    cudaFree(sum);

    return 0;
}
