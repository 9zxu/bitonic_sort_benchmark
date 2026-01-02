#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include "sort_interface.h"
#define checkCudaErrors(call)                                 \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            printf("CUDA error at %s:%d: %s\n", __FILE__,     \
                   __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

#define THREADS 1024 // Max threads per block for RTX 3050

//////////////////////////////////////////////////////////////////////

// 1. Kernel for small strides (Shared Memory)
// Handles j = 512 down to 1 entirely in fast memory
__global__ void bitonicSortShared(int* arr, int k) {
    __shared__ int shared_arr[THREADS];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    shared_arr[tid] = arr[i];
    __syncthreads();

    // Internal bitonic merge
    // FIX: Cap the starting stride 'j' to fit within shared memory capabilities
    // If k=2048 and blockDim=512, we must start j at 256, not 1024.
    unsigned int max_shared_j = blockDim.x >> 1;
    unsigned int initial_j = k >> 1;
    
    if (initial_j > max_shared_j) {
        initial_j = max_shared_j;
    }

    for (int j = initial_j; j > 0; j >>= 1) {
        unsigned int ij = tid ^ j;
        
        // Only one thread in the pair performs the swap logic
        if (ij > tid) {
            bool ascending = ((i & k) == 0); // 'i' is global index, so 'k' check is correct
            int a = shared_arr[tid];
            int b = shared_arr[ij];

            if ((ascending && a > b) || (!ascending && a < b)) {
                shared_arr[tid] = b;
                shared_arr[ij] = a;
            }
        }
        __syncthreads();
    }

    // Write back to global memory
    arr[i] = shared_arr[tid];
}

// 2. Kernel for large strides (Global Memory)
__global__ void bitonicSortGlobal(int* arr, int k, int j) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int ij = i ^ j;

    if (ij > i) {
        bool ascending = ((i & k) == 0);
        int a = arr[i];
        int b = arr[ij];

        if ((ascending && a > b) || (!ascending && a < b)) {
            arr[i] = b;
            arr[ij] = a;
        }
    }
}

void bitonicSortEngine(int* h_arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);
    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    dim3 blocks(n / THREADS);
    dim3 threads(THREADS);

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j >= THREADS) {
                // If stride is larger than block, must use Global Memory
                bitonicSortGlobal<<<blocks, threads>>>(d_arr, k, j);
            } else {
                // If stride fits in block, do ALL remaining j-steps in Shared Memory
                bitonicSortShared<<<blocks, threads>>>(d_arr, k);
                break; // The shared kernel handled all j = stride...1
            }
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

//////////////////////////////// bubble sort ////////////////////////////////////////////////

// CUDA Kernel for Odd-Even Sort step
__global__ void oddEvenStep(int* dev_arr, int n, int phase) {
    // Each thread handles one comparison and potential swap
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the index to compare based on the phase (0 for even, 1 for odd)
    int idx = 2 * i + phase;

    if (idx < n - 1) {
        if (dev_arr[idx] > dev_arr[idx + 1]) {
            int temp = dev_arr[idx];
            dev_arr[idx] = dev_arr[idx + 1];
            dev_arr[idx + 1] = temp;
        }
    }
}

// Parallel CUDA version of Bubble Sort (Odd-Even Sort)
void bubbleSortEngine(int* arr, int n) {
    int *dev_arr;
    size_t size = n * sizeof(int);

    // Allocate and copy memory to device
    cudaMalloc(&dev_arr, size);
    cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice);

    // Configuration: 
    // Since each thread handles a pair (2 elements), we need n/2 threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;

    // The algorithm requires n passes to guarantee sorting in the worst case
    for (int i = 0; i < n; i++) {
        // Phase 0: Even indices (0,1), (2,3)...
        // Phase 1: Odd indices (1,2), (3,4)...
        oddEvenStep<<<blocksPerGrid, threadsPerBlock>>>(dev_arr, n, i % 2);
        
        // Ensure the current phase is finished before starting the next
        // (Kernel launches on the same stream are serialized automatically)
    }

    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(arr, dev_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
}

///////////////////////////////////// thrust sort ///////////////////////////////////////

void baselineSortEngine(int* arr, int n) {
    int *dev_arr;
    cudaMalloc(&dev_arr, n * sizeof(int));
    cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    
    thrust::device_ptr<int> dev_ptr(dev_arr);
    thrust::sort(dev_ptr, dev_ptr + n);
    
    cudaDeviceSynchronize();
    cudaMemcpy(arr, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
}
