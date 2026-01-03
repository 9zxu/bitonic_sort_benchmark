#include <vector>
#include <algorithm>
#include <omp.h>
#include "sort_interface.h"

void bitonicSortEngine(int* arr, int n) {
    // 1. Determine grain size. 
    // Small blocks fit in L1/L2 cache.
    const int GRAIN_SIZE = 4096; 

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            
            // OPTIMIZATION: High-level parallelization for large strides
            if (j >= GRAIN_SIZE) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; i += (j << 1)) {
                    bool ascending = ((i & k) == 0);
                    for (int l = i; l < i + j; l++) {
                        if (ascending) {
                            if (arr[l] > arr[l + j]) std::swap(arr[l], arr[l + j]);
                        } else {
                            if (arr[l] < arr[l + j]) std::swap(arr[l], arr[l + j]);
                        }
                    }
                }
            } 
            // KERNEL: Cache-friendly sequential processing for small strides
            else {
                // We define the block length as 2 * GRAIN_SIZE
                // BUT we must ensure we don't exceed 'n' if n is small
                int block_len = GRAIN_SIZE << 1; 

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; i += block_len) {
                    
                    // FIX: Calculate the actual end of this block to prevent overflow
                    // If n=1024 and block_len=8192, actual_len is 1024.
                    int actual_len = (n - i < block_len) ? (n - i) : block_len;

                    // Iterate j locally down to 1
                    for (int local_j = j; local_j > 0; local_j >>= 1) {
                        
                        // FIX: Limit block_off to actual_len
                        for (int block_off = 0; block_off < actual_len; block_off += (local_j << 1)) {
                            int start_idx = i + block_off;
                            
                            // Check ascending based on the global index concept
                            bool ascending = ((start_idx & k) == 0);
                            
                            for (int l = start_idx; l < start_idx + local_j; l++) {
                                if (ascending) {
                                    if (arr[l] > arr[l + local_j]) std::swap(arr[l], arr[l + local_j]);
                                } else {
                                    if (arr[l] < arr[l + local_j]) std::swap(arr[l], arr[l + local_j]);
                                }
                            }
                        }
                    }
                }
                // Important: we handled all j steps down to 1 for this k
                break; 
            }
        }
    }
}


void baselineSortEngine(int* arr, int n) {
    const int GRAIN_SIZE = 4048; // Fits in L1/L2 cache

    // p is the size of the sorted blocks being merged
    for (int p = 1; p < n; p <<= 1) {
        // k is the stride/distance between elements being compared
        for (int k = p; k >= 1; k >>= 1) {
            
            if (k >= GRAIN_SIZE) {
                #pragma omp parallel for schedule(static)
                for (int j = k % p; j < n - k; j += 2 * k) {
                    // This inner loop ensures we only compare elements 
                    // that belong to the same merge-group (of size 2*p)
                    for (int i = 0; i < k; ++i) {
                        int idx1 = j + i;
                        int idx2 = j + i + k;
                        if (idx1 / (p * 2) == idx2 / (p * 2)) {
                            if (arr[idx1] > arr[idx2]) {
                                std::swap(arr[idx1], arr[idx2]);
                            }
                        }
                    }
                }
            } 
            else {
                // CACHE-FRIENDLY KERNEL
                // When k is small, we process blocks of data in parallel
                #pragma omp parallel for schedule(static)
                for (int block_start = 0; block_start < n; block_start += (GRAIN_SIZE * 2)) {
                    // Each thread takes a chunk of the array and finishes 
                    // ALL remaining k-steps (k, k/2, k/4... 1) for the current p-stage.
                    for (int local_k = k; local_k >= 1; local_k >>= 1) {
                        for (int j = block_start + (local_k % p); j < block_start + (GRAIN_SIZE * 2) - local_k; j += 2 * local_k) {
                            // Ensure we don't go out of array bounds
                            if (j >= n) break; 
                            for (int i = 0; i < local_k; ++i) {
                                int idx1 = j + i;
                                int idx2 = j + i + local_k;
                                if (idx2 < n && (idx1 / (p * 2) == idx2 / (p * 2))) {
                                    if (arr[idx1] > arr[idx2]) std::swap(arr[idx1], arr[idx2]);
                                }
                            }
                        }
                    }
                }
                break; // Current p-stage finished by the kernel
            }
        }
    }
}