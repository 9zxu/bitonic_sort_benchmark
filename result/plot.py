# https://colab.research.google.com/drive/1pJ0SkZlVya0l8Te-7gVRn1oT0zH-UVOa?usp=sharing
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter

df_cpu = pd.read_csv('result_cpu.csv').rename(columns={
    'Bitonic_ms': 'bitonic_cpu',
    'StdSort_ms': 'std::sort',
    'Bubble_ms': 'bubble_cpu' # Assuming _m was a typo for _ms
})

df_omp = pd.read_csv('result_omp.csv').rename(columns={
    'Bitonic_ms': 'bitonic_omp',
    'BubbleOMP_ms': 'bubble_omp'
})[['Size', 'Type', 'bitonic_omp', 'bubble_omp']] # Only keep necessary columns

df_gpu = pd.read_csv('result_gpu.csv').rename(columns={
    'Bitonic_ms': 'bitonic_gpu',
    'ThrustSort_ms': 'thrust::sort',
    'BubbleCUDA_ms': 'bubble_gpu'
})

df = pd.merge(df_cpu, df_omp, on=['Size', 'Type'])
df = pd.merge(df, df_gpu, on=['Size', 'Type'])

df = df.sort_values(by='Size')

lines_to_plot = [
    'bitonic_cpu', 'bitonic_gpu', 'bitonic_omp',
    'bubble_cpu', 'bubble_omp', 'bubble_gpu',
    'std::sort', 'thrust::sort'
]

# logarithm
for data_type in types:
    subset = df[df['Type'] == data_type]

    plt.figure(figsize=(12, 7))

    for line in lines_to_plot:
        mask = subset[line] != -1
        plt.plot(subset['Size'][mask], subset[line][mask],
                 marker='o', label=line)

    plt.title(f'Performance Comparison - Type: {data_type}', fontsize=14)
    plt.xlabel('Size (Number of Elements)', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)

    # Log scales
    plt.xscale('log', base=2)
    plt.yscale('log')

    ax = plt.gca()

    # Major ticks at exact powers of 2
    ax.xaxis.set_major_locator(LogLocator(base=2.0, numticks=12))

    # Minor ticks between powers of 2
    ax.xaxis.set_minor_locator(LogLocator(base=2.0, subs='auto'))

    # LaTeX-style tick labels: 2^{k}
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: rf"$2^{{{int(np.log2(x))}}}$")
    )

    plt.grid(True, which="both", linestyle="-", alpha=0.3)
    plt.legend()

    plt.show()

"""
# non-logarithm x-axis
types = df['Type'].unique()

for data_type in types:
    subset = df[df['Type'] == data_type]

    plt.figure(figsize=(12, 7))

    for line in lines_to_plot:
        # Plotting against 'Size_label' ensures equal distance between x-axis ticks
        plt.plot(subset['Size'], subset[line], marker='o', label=line)

    plt.title(f'Performance Comparison - Type: {data_type}', fontsize=14)
    plt.xlabel('Size (Number of Elements)', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.yscale('log') # Using log scale because Bubble Sort is usually much slower than Bitonic/Std
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()
"""