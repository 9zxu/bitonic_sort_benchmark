#include "sort_interface.h"
#include <algorithm>

void bitonicMerge(int a[], int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
            if (dir == (a[i] > a[i+k])) std::swap(a[i], a[i+k]);
        bitonicMerge(a, low, k, dir);
        bitonicMerge(a, low + k, k, dir);
    }
}

void bitonicSortRecursive(int a[], int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSortRecursive(a, low, k, true);
        bitonicSortRecursive(a, low + k, k, false);
        bitonicMerge(a, low, cnt, dir);
    }
}

void bitonicSortEngine(int* arr, int n) { bitonicSortRecursive(arr, 0, n, true); }
void baselineSortEngine(int* arr, int n) { std::sort(arr, arr + n); }

// Bubble sort
void bubbleSortEngine(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}