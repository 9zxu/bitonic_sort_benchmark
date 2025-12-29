/*
random data only
*/
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;

// Bitonic Sort Implementation
void compare(int a[], int i, int j, bool dir) {
    if (dir == (a[i] > a[j])) {
        swap(a[i], a[j]);
    }
}

void bitonicMerge(int a[], int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
            compare(a, i, i + k, dir);
        bitonicMerge(a, low, k, dir);
        bitonicMerge(a, low + k, k, dir);
    }
}

void bitonicSort(int a[], int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSort(a, low, k, true);
        bitonicSort(a, low + k, k, false);
        bitonicMerge(a, low, cnt, dir);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <size_n>" << endl;
        return 1;
    }

    // 1. Setup Size (Must be power of 2 for this Bitonic implementation)
    int n = atoi(argv[1]);

    // 2. Generate Random Data
    random_device rd;
    mt19937 engine(rd());
    uniform_int_distribution<int> dist(0, 100000);
    
    vector<int> data_bitonic(n);
    for(int i=0; i<n; i++) data_bitonic[i] = dist(engine);
    
    // Copy data for std::sort comparison
    vector<int> data_std = data_bitonic;

    // 3. Benchmark Bitonic Sort
    auto start = chrono::high_resolution_clock::now();
    bitonicSort(data_bitonic.data(), 0, n, true);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> bitonic_time = end - start;

    // 4. Benchmark std::sort
    start = chrono::high_resolution_clock::now();
    std::sort(data_std.begin(), data_std.end());
    end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> std_time = end - start;

    // 5. Output result in CSV format: Size, BitonicTime, StdSortTime
    cout << n << "," << bitonic_time.count() << "," << std_time.count() << endl;

    return 0;
}