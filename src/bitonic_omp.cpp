#include <vector>
#include <algorithm>
#include <omp.h>
#include "sort_interface.h"

// void bitonicSortEngine(int* arr, int n) {
//     for (int k = 2; k <= n; k <<= 1) {
//         for (int j = k >> 1; j > 0; j >>= 1) {
//             #pragma omp parallel for
//             for (int i = 0; i < n; i++) {
//                 int ij = i ^ j;
//                 if (ij > i) {
//                     bool ascending = ((i & k) == 0);
//                     if (ascending) {
//                         if (arr[i] > arr[ij]) std::swap(arr[i], arr[ij]);
//                     } else {
//                         if (arr[i] < arr[ij]) std::swap(arr[i], arr[ij]);
//                     }
//                 }
//             }
//         }
//     }
// }

// Bitonic Sort Engine
void bitonicSortEngine(int* arr, int n) {
    #pragma omp parallel
    {
        for (int k = 2; k <= n; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                #pragma omp for schedule(static)
                for (int i = 0; i < n; i++) {
                    int ij = i ^ j;
                    if (ij > i) {
                        bool ascending = ((i & k) == 0);
                        if (ascending) {
                            if (arr[i] > arr[ij])
                                std::swap(arr[i], arr[ij]);
                        } else {
                            if (arr[i] < arr[ij])
                                std::swap(arr[i], arr[ij]);
                        }
                    }
                }
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