# data analysis and visualization
- each result.csv has `Size,Type,Bitonic_ms,g_kernel_time_ms,elements_per_sec,throughput_cycles,Baseline_ms`
- rename them as bitonic_(), batcher_(), std::sort, g_kernel_time_ms, elements_per_sec_(), throughput_cycles_(), with () filled by `cpu`, `omp`, `gpu`
- `g_kernel_time_ms` is set to 0 if is not `result_gpu`, so please drop this column if is not `result_gpu`
- baseline for cpu is `std::sort`, for omp is `batcher's oddeven merge sort omp optimized`, for gpu is `batcher's oddeven merge sort gpu optimized`

I have the following instructions:
1. a line chart of `bitonic_cpu`, `bitonic_omp`, `bitonic_gpu`, `std::sort`, `batcher_omp`, `batcher_gpu`, x-axis is log scale on base 2, meaning `size` (number of elements), while y-axis is log scale on base 10, meaning executing time in unit of `ms`
please follow this template

```python
line_to_plot = [
    # `bitonic_cpu`, `bitonic_omp`, `bitonic_gpu`, `std::sort`, `batcher_omp`, `batcher_gpu`
]

types = df['Type'].unique()

for data_type in types:
  subset = df[df['Type'] == data_type]
  # ...
  
  # Log scales
  plt.xscale('log', base=2)
  plt.yscale('log')

  # ...

  plt.grid(True, which="both", linestyle="-", alpha=0.3)
  plt.legend()

  plt.show()

```
2. a chart of all `throughput_cycles`
3. a chart of all `elements_per_sec`
4. speed up factor table
   - three lines: `std::sort / (bitonic_cpu, bitonic_omp, bitonic_gpu)`
5. one line `gpu_overhead_ms gpu_overhead_ms = Bitonic_ms - g_kernel_time_ms` trends chart

write such a python script for me, and 
save each figure with snake case naming.