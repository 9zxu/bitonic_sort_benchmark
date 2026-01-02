#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include "sort_interface.h"

// Bitonic and Baseline Sort engines
__global__ void bitonicSortStep(int* dev_arr, int k, int j) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ij = i ^ j; 
    if (ij > i) {
        bool ascending = ((i & k) == 0);
        int a = dev_arr[i];
        int b = dev_arr[ij];
        if (ascending) {
            if (a > b) { dev_arr[i] = b; dev_arr[ij] = a; }
        } else {
            if (a < b) { dev_arr[i] = b; dev_arr[ij] = a; }
        }
    }
}

void bitonicSortEngine(int* arr, int n) {
    int *dev_arr;
    cudaMalloc(&dev_arr, n * sizeof(int));
    cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<blocksPerGrid, threadsPerBlock>>>(dev_arr, k, j);
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(arr, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
}

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

void baselineSortEngine(int* arr, int n) {
    int *dev_arr;
    cudaMalloc(&dev_arr, n * sizeof(int));
    cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    
    thrust::device_ptr<int> dev_ptr(dev_arr);
    thrust::sort(dev_ptr, dev_ptr + n);
    
    cudaMemcpy(arr, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
}