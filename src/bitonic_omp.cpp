#include <vector>
#include <algorithm>
#include <omp.h>
#include "sort_interface.h"

void bitonicSortEngine(int* arr, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                int ij = i ^ j;
                if (ij > i) {
                    bool ascending = ((i & k) == 0);
                    if (ascending) {
                        if (arr[i] > arr[ij]) std::swap(arr[i], arr[ij]);
                    } else {
                        if (arr[i] < arr[ij]) std::swap(arr[i], arr[ij]);
                    }
                }
            }
        }
    }
}

void baselineSortEngine(int* arr, int n) {
    std::sort(arr, arr + n);
}