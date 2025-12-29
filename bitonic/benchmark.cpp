#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "sort_interface.h"

using namespace std;

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

    vector<int> data_bitonic(n);
    generateData(data_bitonic, n, type_idx);
    vector<int> data_baseline = data_bitonic;

    // Bitonic Test
    auto s1 = chrono::high_resolution_clock::now();
    bitonicSortEngine(data_bitonic.data(), n);
    auto e1 = chrono::high_resolution_clock::now();
    
    // Baseline Test
    auto s2 = chrono::high_resolution_clock::now();
    baselineSortEngine(data_baseline.data(), n);
    auto e2 = chrono::high_resolution_clock::now();

    double t_bitonic = chrono::duration<double, milli>(e1 - s1).count();
    double t_baseline = chrono::duration<double, milli>(e2 - s2).count();

    // Size, Type, Bitonic_ms, Baseline_ms
    cout << n << "," << labels[type_idx] << "," << t_bitonic << "," << t_baseline << endl;

    return 0;
}