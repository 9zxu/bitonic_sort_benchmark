#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

// double g_kernel_time_ms = 0.0;

void bitonicSortEngine(int* h_arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);
    cudaMalloc(&d_arr, size);
    
    // 1. Data Transfer (Not part of kernel time)
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    dim3 blocks(n / THREADS);
    dim3 threads(THREADS);

    // 2. Setup Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording
    cudaEventRecord(start);

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            // ... your kernel launch logic ...
            // bitonicSortGlobal<<<blocks, threads>>>(d_arr, k, j);
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

    // Stop recording
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    g_kernel_time_ms = (double)milliseconds; // Store in global for benchmark.cpp

    // 3. Data Transfer Back (Not part of kernel time)
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


/* not calculating kernel time
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
*/


// 1. Global Kernel for large strides
__global__ void oddEvenMergeGlobal(int* arr, int n, int p, int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // In Batcher's, not every index is a 'start' index. 
    // We map thread 'i' to the logic j + i from the CPU version.
    // Each thread handles one comparison.
    unsigned int group_size = 2 * k;
    unsigned int chunk_id = i / k;
    unsigned int inner_idx = i % k;
    unsigned int start_idx = (chunk_id * group_size) + (k % p) + inner_idx;

    unsigned int idx1 = start_idx;
    unsigned int idx2 = start_idx + k;

    if (idx2 < n && (idx1 / (2 * p) == idx2 / (2 * p))) {
        int a = arr[idx1];
        int b = arr[idx2];
        if (a > b) {
            arr[idx1] = b;
            arr[idx2] = a;
        }
    }
}

// 2. Shared Memory Kernel for small strides
__global__ void oddEvenMergeShared(int* arr, int n, int p, int k_start) {
    __shared__ int s_arr[THREADS * 2];
    
    unsigned int tid = threadIdx.x;
    unsigned int block_offset = blockIdx.x * (THREADS * 2);

    // Load two elements per thread into shared memory
    if (block_offset + tid < n) s_arr[tid] = arr[block_offset + tid];
    if (block_offset + tid + THREADS < n) s_arr[tid + THREADS] = arr[block_offset + tid + THREADS];
    __syncthreads();

    // Process all remaining strides (k_start, k_start/2, ... 1)
    for (int k = k_start; k >= 1; k >>= 1) {
        // Logic similar to global but relative to shared memory
        unsigned int group_size = 2 * k;
        
        // Each thread handles one comparison in the shared block
        // Since we have THREADS*2 elements, we have THREADS comparisons
        unsigned int chunk_id = tid / k;
        unsigned int inner_idx = tid % k;
        unsigned int s_idx1 = (chunk_id * group_size) + (k % p) + inner_idx;
        unsigned int s_idx2 = s_idx1 + k;

        unsigned int g_idx1 = block_offset + s_idx1;
        unsigned int g_idx2 = block_offset + s_idx2;

        if (s_idx2 < (THREADS * 2) && g_idx2 < n) {
            if ((g_idx1 / (2 * p)) == (g_idx2 / (2 * p))) {
                if (s_arr[s_idx1] > s_arr[s_idx2]) {
                    int tmp = s_arr[s_idx1];
                    s_arr[s_idx1] = s_arr[s_idx2];
                    s_arr[s_idx2] = tmp;
                }
            }
        }
        __syncthreads();
    }

    // Write back
    if (block_offset + tid < n) arr[block_offset + tid] = s_arr[tid];
    if (block_offset + tid + THREADS < n) arr[block_offset + tid + THREADS] = s_arr[tid + THREADS];
}

void baselineSortEngine(int* h_arr, int n) {
    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    for (int p = 1; p < n; p <<= 1) {
        for (int k = p; k >= 1; k >>= 1) {
            if (k >= THREADS) {
                // Stride is too large for shared memory, use 1 thread per comparison
                int num_comparisons = n / 2; 
                int blocks = (num_comparisons + THREADS - 1) / THREADS;
                oddEvenMergeGlobal<<<blocks, THREADS>>>(d_arr, n, p, k);
            } else {
                // Stride fits in shared memory. 
                // Each block handles 2*THREADS elements.
                int blocks = (n + (THREADS * 2) - 1) / (THREADS * 2);
                oddEvenMergeShared<<<blocks, THREADS>>>(d_arr, n, p, k);
                break; // Shared kernel finishes all smaller k for this p
            }
        }
    }

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}