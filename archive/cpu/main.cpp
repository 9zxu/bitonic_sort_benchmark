#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "sort_interface.h"

using namespace std;
using namespace std::chrono;

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
void bubbleSortEngine(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

void generateData(vector<int>& data, int n, int type) {
    if (type == 1) for (int i = 0; i < n; i++) data[i] = i;              // Sorted
    else if (type == 2) for (int i = 0; i < n; i++) data[i] = n - i;    // Reverse
    else if (type == 3) for (int i = 0; i < n; i++) data[i] = i % 10;   // Duplicate
    else {                                                              // Random
        static mt19937 eng(42);
        uniform_int_distribution<int> dist(0, 1000000);
        for (int i = 0; i < n; i++) data[i] = dist(eng);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) return 1;
    int n = atoi(argv[1]);
    int type_idx = atoi(argv[2]);
    string labels[] = {"Random", "Sorted", "Reverse", "Duplicate"};

    constexpr int REPEAT = 10;
    int total_runs = REPEAT + 1;

    vector<int> data_original(n);
    generateData(data_original, n, type_idx);

    vector<int> data_bitonic(n);
    vector<int> data_baseline(n);
    vector<int> data_bubble(n);

    double bitonic_total = 0.0;
    double baseline_total = 0.0;
    double bubble_total = 0.0;

    for (int i = 0; i < total_runs; ++i) {
        copy(data_original.begin(), data_original.end(), data_bitonic.begin());
        copy(data_original.begin(), data_original.end(), data_baseline.begin());
        copy(data_original.begin(), data_original.end(), data_bubble.begin());

        // Bitonic sort
        auto s1 = high_resolution_clock::now();
        bitonicSortEngine(data_bitonic.data(), n);
        auto e1 = high_resolution_clock::now();

        // std::sort
        auto s2 = high_resolution_clock::now();
        baselineSortEngine(data_baseline.data(), n);
        auto e2 = high_resolution_clock::now();

        // Bubble sort (only for small n)
        double bubble_time = 0.0;
        if (n <= 32768) {
            auto s3 = high_resolution_clock::now();
            bubbleSortEngine(data_bubble.data(), n);
            auto e3 = high_resolution_clock::now();
            bubble_time = duration<double, milli>(e3 - s3).count();
        }

        // discard warm-up
        if (i > 0) {
            bitonic_total += duration<double, milli>(e1 - s1).count();
            baseline_total += duration<double, milli>(e2 - s2).count();
            if (n <= 32768)
                bubble_total += bubble_time;
        }
    }

    double bitonic_avg = bitonic_total / REPEAT;
    double baseline_avg = baseline_total / REPEAT;
    double bubble_avg = (n <= 32768) ? (bubble_total / REPEAT) : -1.0;

    // CSV Output
    // Size,Type,Bitonic_ms,StdSort_ms,Bubble_ms
    cout << n << "," << labels[type_idx] << ","
         << bitonic_avg << ","
         << baseline_avg << ","
         << bubble_avg << "\n";

    return 0;
}
