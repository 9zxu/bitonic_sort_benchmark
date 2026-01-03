import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import data2tex
import os

# 1. Load and Clean Data
def load_and_prepare(file_path, suffix):
    df = pd.read_csv(file_path)
    
    # Define renaming map based on instruction
    rename_map = {
        'Bitonic_ms': f'bitonic_{suffix}',
        'elements_per_sec': f'elements_per_sec_{suffix}',
        'throughput_cycles': f'throughput_cycles_{suffix}',
    }
    
    # Specific Baseline renaming logic
    if suffix == 'cpu':
        rename_map['Baseline_ms'] = 'std::sort'
    elif suffix == 'omp':
        rename_map['Baseline_ms'] = 'batcher_omp'
    elif suffix == 'gpu':
        rename_map['Baseline_ms'] = 'batcher_gpu'
    
    df = df.rename(columns=rename_map)
    
    # Drop g_kernel_time_ms if not GPU
    if suffix != 'gpu':
        df = df.drop(columns=['g_kernel_time_ms'], errors='ignore')
    
    return df


# Load the three files
df_cpu = load_and_prepare('result_cpu.csv', 'cpu')
df_omp = load_and_prepare('result_omp.csv', 'omp')
df_gpu = load_and_prepare('result_gpu.csv', 'gpu')

# 2. Merge DataFrames on Size and Type
df = pd.merge(df_cpu, df_omp, on=['Size', 'Type'])
df = pd.merge(df, df_gpu, on=['Size', 'Type'])

# Calculate GPU Overhead
df['gpu_overhead_ms'] = df['bitonic_gpu'] - df['g_kernel_time_ms']

# Calculate Speedup Factors (Relative to std::sort)
df['speedup_cpu'] = df['std::sort'] / df['bitonic_cpu']
df['speedup_omp'] = df['std::sort'] / df['bitonic_omp']
df['speedup_gpu'] = df['std::sort'] / df['bitonic_gpu']

# 3. Visualization Setup
types = df['Type'].unique()

for data_type in types:
    subset = df[df['Type'] == data_type].sort_values('Size')
    type_slug = data_type.lower()

    if data_type == 'Random':
        save_it = True
    else:
        save_it = False

    # --- CHART 1: Execution Time (Log-Log) ---
    plt.figure(figsize=(10, 6))
    lines_to_plot = [
        'bitonic_cpu', 'bitonic_omp', 'bitonic_gpu', 
        'std::sort', 'batcher_omp', 'batcher_gpu'
    ]
    for line in lines_to_plot:
        plt.plot(subset['Size'], subset[line], marker='o', label=line)
    
    plt.xscale('log', base=2)
    plt.yscale('log') # Default base 10
    plt.title(f'Execution Time - {data_type}')
    plt.xlabel('Size (N)')
    plt.ylabel('Time (ms)')
    plt.grid(True, which="both", linestyle="-", alpha=0.3)
    plt.legend()
    if save_it:
        plt.savefig(f'execution_time_{type_slug}.png')
    plt.show()

    # --- CHART 2: Throughput Cycles ---
    plt.figure(figsize=(10, 6))
    cycles = ['throughput_cycles_cpu', 'throughput_cycles_omp', 'throughput_cycles_gpu']
    for line in cycles:
        plt.plot(subset['Size'], subset[line], marker='s', label=line)
    
    plt.xscale('log', base=2)
    plt.title(f'Throughput Cycles - {data_type}')
    plt.xlabel('Size (N)')
    plt.ylabel('Cycles per Element')
    plt.grid(True, which="both", linestyle="-", alpha=0.3)
    plt.legend()
    if save_it:
        plt.savefig(f'throughput_cycles_{type_slug}.png')
    plt.show()

    # --- CHART 3: Elements Per Second ---
    plt.figure(figsize=(10, 6))
    eps = ['elements_per_sec_cpu', 'elements_per_sec_omp', 'elements_per_sec_gpu']
    for line in eps:
        plt.plot(subset['Size'], subset[line], marker='^', label=line)
    
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.title(f'Throughput (Elements/Sec) - {data_type}')
    plt.xlabel('Size (N)')
    plt.ylabel('Elements / Second')
    plt.grid(True, which="both", linestyle="-", alpha=0.3)
    plt.legend()
    if save_it:
        plt.savefig(f'elements_per_sec_{type_slug}.png')
    plt.show()

    # --- CHART 5: GPU Overhead Trend ---
    plt.figure(figsize=(10, 6))
    plt.plot(subset['Size'], subset['gpu_overhead_ms'], marker='x', color='red', label='GPU Overhead (Total - Kernel)')
    plt.xscale('log', base=2)
    plt.title(f'GPU Implementation Overhead - {data_type}')
    plt.xlabel('Size (N)')
    plt.ylabel('Overhead (ms)')
    plt.grid(True, which="both", linestyle="-", alpha=0.3)
    plt.legend()
    if save_it:
        plt.savefig(f'gpu_overhead_{type_slug}.png')
    plt.show()

    # --- 4. Speedup Factor Table (Output to console/terminal) ---
    ### terminal output version
    # print(f"\n{'='*20} Speedup Table: {data_type} {'='*20}")
    # table_view = subset[['Size', 'speedup_cpu', 'speedup_omp', 'speedup_gpu']].copy()
    # # Format size for readability
    # table_view['Size'] = table_view['Size'].apply(lambda x: f"2^{int(np.log2(x))}")
    # print(table_view.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.plot(subset['Size'], subset['speedup_cpu'], marker='o', label='CPU Speedup')
    plt.plot(subset['Size'], subset['speedup_omp'], marker='s', label='OMP Speedup')
    plt.plot(subset['Size'], subset['speedup_gpu'], marker='^', label='GPU Speedup')
    
    # Baseline line at y=1 (where Bitonic performance equals std::sort)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='std::sort Baseline')
    
    plt.xscale('log', base=2)
    # Speedup can vary significantly, log scale for y is optional but often helpful
    plt.yscale('linear') 
    
    plt.title(f'Speedup Factor vs std::sort - {data_type}')
    plt.xlabel('Size (N)')
    plt.ylabel('Speedup Ratio ($T_{std::sort} / T_{bitonic}$)')
    plt.grid(True, which="both", linestyle="-", alpha=0.3)
    plt.legend()
    if save_it:
        plt.savefig(f'speedup_plot_{type_slug}.png')
    plt.show()

    # --- Generate LaTeX Table for "Random" type specifically ---
    if save_it:
        data2tex.generate_latex_file(df, 'Random', "speedup_table_random.tex")