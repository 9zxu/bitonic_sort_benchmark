# bitonic sort

```cpp
// 核心：比較與交換
void compare(int a[], int i, int j, bool dir) {
    if (dir == (a[i] > a[j])) {
        std::swap(a[i], a[j]);
    }
}

// 雙調合併
void bitonicMerge(int a[], int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
            compare(a, i, i + k, dir);
        bitonicMerge(a, low, k, dir);
        bitonicMerge(a, low + k, k, dir);
    }
}

// 雙調排序主體
void bitonicSort(int a[], int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSort(a, low, k, true);  // 升序
        bitonicSort(a, low + k, k, false); // 降序
        bitonicMerge(a, low, cnt, dir);
    }
}
```
## omp
### naive
```c++
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
```

## cuda
### naive
> often bottlenecked by memory latency rather than computation.
```c++
// Bitonic and Baseline Sort engines
__global__ void bitonicSortStep(int* dev_arr, int k, int j) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ij = i ^ j; 
    if (ij > i) {
        bool ascending = ((i & k) == 0);
        int a = dev_arr[i];
        int b = dev_arr[ij];
        if (ascending) {
            if (a > b) { dev_arr[i] = b; dev_arr[ij] = a; }
        } else {
            if (a < b) { dev_arr[i] = b; dev_arr[ij] = a; }
        }
    }
}
void bitonicSortEngine(int* arr, int n) {
    int *dev_arr;
    cudaMalloc(&dev_arr, n * sizeof(int));
    cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<blocksPerGrid, threadsPerBlock>>>(dev_arr, k, j);
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(arr, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
}
```
# Data

- [x] **Random Data**：最常見的平均情況。
- [x] **Sorted Data**：測試演算法是否能處理已排序的情況（Bitonic Sort 依然會執行所有比較）。
- [x] **Reverse Sorted Data**：測試最差情況。
- [x] **Duplicate Data**：含大量重複數值的資料。
- [x] **Different Scale** : 測試不同規模的數據

## type

```cpp
enum DataType { RANDOM, SORTED, REVERSE, DUPLICATE };

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
```

## size
```
# Powers of 2 from 2^10 to 2^26, r = 4
SIZES = 1024 4096 16384 65536 262144 1048576 4194304 16777216 67108864
```
## range
> 不要只用 `rand()`，因為它的範圍可能只有 32767。建議使用 C++11 的 `<random>` 生成全範圍整數。
```c++
        static mt19937 eng(42);
        uniform_int_distribution<int> dist(0, 1000000);
```

## average for 10 times
```c++
    double bitonic_total = 0.0;
    double baseline_total = 0.0;
    double bubble_total = 0.0;

    for (int i = 0; i < total_runs; ++i) {
        copy(data_original.begin(), data_original.end(), data_bitonic.begin());
        copy(data_original.begin(), data_original.end(), data_baseline.begin());
        copy(data_original.begin(), data_original.end(), data_bubble.begin());

        // Bitonic sort
        // ...

        // std::sort
        // ...

        // Bubble sort (only for small n)
        double bubble_time = 0.0;
        if (n <= 32768) {
            // ...
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
```

## usage

```make
make # compile all
make bench_cpu
make bench_omp
make bench_gpu
```
check out `.csv` files

# result

[colab plot link](https://colab.research.google.com/drive/1pJ0SkZlVya0l8Te-7gVRn1oT0zH-UVOa?usp=sharing)

- Bitonic sort on single thread cpu is not faster than std::sort
- Bitonic sort on omp is ? times faster than it is on single thread, 
- Bitonic sort on GPU is not faster than thrus(optimized parallel sort by nvidia), but faster than std::sort
- Bitonic sort is also significantly faster than bubble sort.

# hardware information
### CPU

```
--- Hardware Specifications ---
--- CPU/Processor Info ---
Model Name: 12th Gen Intel(R) Core(TM) i5-12400F
Total Cores: 6
Logical Processors (Threads): 12

--- Memory Info ---
Total Physical Memory: 15.54 GB
```

### GPU

```
$ nvidia-smi
Mon Dec 29 22:24:04 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.274.02             Driver Version: 535.274.02   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3050        Off | 00000000:01:00.0  On |                  N/A |
|  0%   45C    P8              12W / 130W |    660MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      3074      G   /usr/lib/xorg/Xorg                          248MiB |
|    0   N/A  N/A      3313      G   /usr/bin/gnome-shell                        108MiB |
|    0   N/A  N/A      4108      G   /proc/self/exe                               50MiB |
|    0   N/A  N/A      4396      G   ...onService --variations-seed-version      110MiB |
|    0   N/A  N/A      4780      G   ...cess-track-uuid=3190708988185955192      132MiB |
+---------------------------------------------------------------------------------------+

```
