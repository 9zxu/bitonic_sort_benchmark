#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include "sort_interface.h"

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
