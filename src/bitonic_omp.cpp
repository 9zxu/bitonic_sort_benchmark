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
    std::sort(arr, arr + n);
}

void bubbleSortEngine(int* arr, int n) {
    // Standard Bubble Sort is inherently sequential.
    // We use Odd-Even Transposition Sort for parallelization.
    
    for (int i = 0; i < n; i++) {
        bool swapped = false;

        // Phase 1: Even-indexed pairs (0,1), (2,3), (4,5)...
        #pragma omp parallel for reduction(|:swapped)
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        // Phase 2: Odd-indexed pairs (1,2), (3,4), (5,6)...
        #pragma omp parallel for reduction(|:swapped)
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }

        // If no swaps occurred in either phase, the array is sorted
        if (!swapped) break;
    }
}