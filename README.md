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

## Data

- [x] **Random Data**：最常見的平均情況。
- [x] **Sorted Data**：測試演算法是否能處理已排序的情況（Bitonic Sort 依然會執行所有比較）。
- [x] **Reverse Sorted Data**：測試最差情況。
- [x] **Duplicate Data**：含大量重複數值的資料。
- [x] **Different Scale** : 測試不同規模的數據

> warning
**數據生成提示**：
不要只用 `rand()`，因為它的範圍可能只有 32767。建議使用 C++11 的 `<random>` 生成全範圍整數。

```cpp
enum DataType { RANDOM, SORTED, REVERSE, DUPLICATE };

void generateData(vector<int>& data, int n, DataType type) {
    if (type == SORTED) {
        for (int i = 0; i < n; i++) data[i] = i;
    } else if (type == REVERSE) {
        for (int i = 0; i < n; i++) data[i] = n - i;
    } else if (type == DUPLICATE) {
        for (int i = 0; i < n; i++) data[i] = i % 10;
    } else {
        static mt19937 eng(42); 
        uniform_int_distribution<int> dist(0, 1000000);
        for (int i = 0; i < n; i++) data[i] = dist(eng);
    }
}
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

### single thread CPU

|Size    |Type     |Bitonic_ms|StdSort_ms|
|--------|---------|----------|----------|
|1024    |Random   |0.627646  |0.178277  |
|1024    |Sorted   |0.139935  |0.017969  |
|1024    |Reverse  |0.096061  |0.008206  |
|1024    |Duplicate|0.471883  |0.095261  |
|4096    |Random   |1.4101    |0.353661  |
|4096    |Sorted   |0.277287  |0.03094   |
|4096    |Reverse  |0.286396  |0.022141  |
|4096    |Duplicate|0.544096  |0.127778  |
|16384   |Random   |6.81134   |1.54929   |
|16384   |Sorted   |1.42838   |0.176494  |
|16384   |Reverse  |2.21165   |0.321039  |
|16384   |Duplicate|2.69469   |0.550811  |
|65536   |Random   |21.9722   |3.32156   |
|65536   |Sorted   |3.36749   |0.372256  |
|65536   |Reverse  |3.41072   |0.263709  |
|65536   |Duplicate|5.9256    |1.07325   |
|262144  |Random   |56.2075   |13.7105   |
|262144  |Sorted   |15.1265   |1.44571   |
|262144  |Reverse  |15.2511   |0.951237  |
|262144  |Duplicate|17.0125   |2.97629   |
|1048576 |Random   |230.769   |55.9429   |
|1048576 |Sorted   |74.124    |6.47664   |
|1048576 |Reverse  |76.0361   |4.07029   |
|1048576 |Duplicate|79.6287   |10.6404   |
|4194304 |Random   |1017.79   |244.286   |
|4194304 |Sorted   |344.752   |25.8921   |
|4194304 |Reverse  |333.524   |17.4536   |
|4194304 |Duplicate|357.015   |48.1648   |
|33554432|Random   |8967.18   |1959.38   |
|33554432|Sorted   |3396.72   |239.953   |
|33554432|Reverse  |3357.24   |159.853   |
|33554432|Duplicate|3507.05   |443.228   |

### omp

|Size    |Type     |Bitonic_ms|StdSort_ms|
|--------|---------|----------|----------|
|1024    |Random   |0.264755  |0.03638   |
|1024    |Sorted   |56.4048   |0.006782  |
|1024    |Reverse  |0.255037  |0.004353  |
|1024    |Duplicate|0.252284  |0.016942  |
|4096    |Random   |0.454531  |0.158374  |
|4096    |Sorted   |0.318418  |0.022087  |
|4096    |Reverse  |0.321568  |0.01608   |
|4096    |Duplicate|0.338805  |0.053718  |
|16384   |Random   |0.971151  |0.74899   |
|16384   |Sorted   |46.6506   |0.088587  |
|16384   |Reverse  |65.317    |0.069297  |
|16384   |Duplicate|0.48824   |0.211582  |
|65536   |Random   |130.914   |3.31261   |
|65536   |Sorted   |65.929    |0.380859  |
|65536   |Reverse  |1.10446   |0.292577  |
|65536   |Duplicate|1.19436   |0.774684  |
|262144  |Random   |74.1318   |14.6045   |
|262144  |Sorted   |88.2271   |2.44896   |
|262144  |Reverse  |50.776    |1.36496   |
|262144  |Duplicate|3.98456   |3.15951   |
|1048576 |Random   |32.335    |60.7719   |
|1048576 |Sorted   |81.6902   |7.26005   |
|1048576 |Reverse  |18.2477   |5.75288   |
|1048576 |Duplicate|74.567    |12.2508   |
|4194304 |Random   |226.052   |254.661   |
|4194304 |Sorted   |87.4272   |30.1001   |
|4194304 |Reverse  |383.382   |25.8978   |
|4194304 |Duplicate|95.2587   |49.7801   |
|33554432|Random   |2283.2    |2059.03   |
|33554432|Sorted   |1657.48   |262.346   |
|33554432|Reverse  |2044.01   |200.339   |
|33554432|Duplicate|1520.86   |458.966   |

### GPU

|Size    |Type     |Bitonic_ms|ThrustSort_ms|
|--------|---------|----------|-------------|
|1024    |Random   |84.6266   |0.139694     |
|1024    |Sorted   |40.5019   |0.139715     |
|1024    |Reverse  |42.1187   |0.139839     |
|1024    |Duplicate|52.5278   |0.140618     |
|4096    |Random   |46.7303   |0.153461     |
|4096    |Sorted   |47.022    |0.148216     |
|4096    |Reverse  |47.9891   |0.150088     |
|4096    |Duplicate|47.352    |0.146821     |
|16384   |Random   |39.8688   |0.224659     |
|16384   |Sorted   |41.5234   |0.220137     |
|16384   |Reverse  |53.652    |0.220954     |
|16384   |Duplicate|46.2389   |0.22293      |
|65536   |Random   |39.7695   |0.273276     |
|65536   |Sorted   |46.8608   |0.277305     |
|65536   |Reverse  |43.5225   |0.273594     |
|65536   |Duplicate|44.0128   |0.268564     |
|262144  |Random   |44.3188   |0.596685     |
|262144  |Sorted   |46.2526   |0.600079     |
|262144  |Reverse  |44.1609   |0.600056     |
|262144  |Duplicate|54.0766   |0.583914     |
|1048576 |Random   |58.351    |1.43013      |
|1048576 |Sorted   |56.6504   |1.51589      |
|1048576 |Reverse  |51.9612   |1.44509      |
|1048576 |Duplicate|49.6266   |1.40429      |
|4194304 |Random   |80.441    |5.01272      |
|4194304 |Sorted   |79.8348   |5.11176      |
|4194304 |Reverse  |79.1573   |5.2574       |
|4194304 |Duplicate|74.1999   |5.14348      |
|33554432|Random   |418.665   |37.5251      |
|33554432|Sorted   |393.452   |37.9283      |
|33554432|Reverse  |397.848   |37.8726      |
|33554432|Duplicate|323.652   |37.5372      |


- Bitonic sort on single thread cpu is not faster than std::sort
- Bitonic sort on omp is ? times faster than it is on single thread, 
- Bitonic sort on GPU is not faster than thrus(optimized parallel sort by nvidia), but faster than std::sort

## hardware information

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
