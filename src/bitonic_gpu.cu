#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "sort_interface.h"

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

void baselineSortEngine(int* arr, int n) {
    int *dev_arr;
    cudaMalloc(&dev_arr, n * sizeof(int));
    cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    
    thrust::device_ptr<int> dev_ptr(dev_arr);
    thrust::sort(dev_ptr, dev_ptr + n);
    
    cudaMemcpy(arr, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
}