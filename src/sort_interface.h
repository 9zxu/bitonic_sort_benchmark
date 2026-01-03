/*
This header allows benchmark.cpp to call a single function regardless of whether it's CPU or GPU code.
*/
#ifndef SORT_INTERFACE_H
#define SORT_INTERFACE_H

extern double g_kernel_time_ms;

// This will call the specific bitonic version compiled for the binary
void bitonicSortEngine(int* arr, int n);

// This will call the baseline version (std::sort or thrust::sort)
void baselineSortEngine(int* arr, int n);


#endif