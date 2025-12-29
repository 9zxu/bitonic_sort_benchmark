#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

void bitonicSortOpenMP(vector<int>& arr, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                int ij = i ^ j;
                if (ij > i) {
                    bool ascending = ((i & k) == 0);
                    if (ascending) {
                        if (arr[i] > arr[ij]) swap(arr[i], arr[ij]);
                    } else {
                        if (arr[i] < arr[ij]) swap(arr[i], arr[ij]);
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <size_n>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);

    // random data
    random_device rd;
    mt19937 engine(rd());
    uniform_int_distribution<int> dist(0, 100000);

    vector<int> data(n), data_std(n);
    for (int i = 0; i < n; i++) {
        data[i] = data_std[i] = dist(engine);
    }

    // ---------------- Bitonic Sort ----------------
    auto start = chrono::high_resolution_clock::now();
    bitonicSortOpenMP(data, n);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> bitonic_time = end - start;

    // ---------------- Std Sort ----------------
    start = chrono::high_resolution_clock::now();
    sort(data_std.begin(), data_std.end());
    end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> std_time = end - start;

    // output CSV: Size, BitonicTime, StdSortTime
    cout << n << "," << bitonic_time.count() << "," << std_time.count() << endl;

    return 0;
}