#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>  // For std::sort

using namespace std;

__global__
void odd_even_kernel(long int n, long int *y, int phase)
{
    long int array_index = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + phase;

    // times two because each kernel can handle 2 elements
    long int stride = gridDim.x * blockDim.x * 2;

    while (array_index < n - 1){
        long int idx_a = array_index;
        long int idx_b = idx_a + 1;

        if (y[idx_a] > y[idx_b]) {
            // Perform the swap
            long int temp = y[idx_a];
            y[idx_a] = y[idx_b];
            y[idx_b] = temp;
        }

        array_index += stride;
    }
}

class CudaTimer {
public:
    CudaTimer(const std::string& name) : name_(name) {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~CudaTimer() {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        std::cout << name_ << " - Time elapsed: " << milliseconds / 1000.0 << " seconds" << std::endl;
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

private:
    std::string name_;
    cudaEvent_t start_, stop_;
};

/**
 * Return a random integer in the range [a..b]
 */
int randab(int a, int b){
    return a + (rand() % (b-a+1));
}

/**
 * Fill vector x with a random permutation of the integers 0..n-1
 */
void fill(long int *x, long int n ){
    int i;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
    for(i=0; i<n-1; i++) {
        const int j = randab(i, n-1);
        const int tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

int check( const long int *x, long int n )
{
    int i;
    for (i=1; i<n; i++) {
        if (x[i] < x[i-1]) {
            cout << "Check Failed at Index " << to_string(i) << endl;
            return 0;
        }
    }
    printf("Check OK -- SUCCESSFULLY SORTED\n");
    return 1;
}

void printArr(long int *&y, long int N){
    for (long int i = 0; i < N; ++i){
        cout << y[i] << ", ";
    }
    cout << endl;
}

int main(void)
{
    long int N = 1LL << 15;
    long int *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&y, N * sizeof(long int));
    {
        CudaTimer timer("Array Fill");
        fill(y, N);
    }

    printArr(y, 25);

    // Kernel launch parameters
    int blockSize = 256;
    int numBlocks = 256;

    // Timing GPU sort
    {
        CudaTimer timer("GPU Sort");
        for (int i = 0; i < N; i++) {
            int phase = i & 1;
            odd_even_kernel<<<numBlocks, blockSize>>>(N, y, phase);
            cudaDeviceSynchronize();
        }
    }

    check(y, N);
    printArr(y, 75);

    // Refill array for CPU sort
    fill(y, N);

    // Timing CPU sort
    auto start = chrono::high_resolution_clock::now();
    std::sort(y, y + N);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    std::cout << "CPU Sort - Time elapsed: " << duration.count() << " seconds" << std::endl;

    check(y, N);

    // Free allocated memory
    cudaFree(y);

    return 0;
}
