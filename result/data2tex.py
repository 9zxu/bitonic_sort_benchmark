import numpy as np

def generate_latex_file(df, data_type, filename="table.tex"):
    """Generates a complete, compilable LaTeX document with the speedup table."""
    subset = df[df['Type'] == data_type].sort_values('Size')
    
    latex_content = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage[margin=1in]{geometry}

\begin{document}

\begin{table}[htbp]
\centering
\caption{Speedup Factor Comparison ($std::sort / Bitonic$) for """ + data_type + r""" Data}
\label{tab:speedup_""" + data_type.lower() + r"""}
\begin{tabular}{l S[table-format=1.3] S[table-format=1.3] S[table-format=2.3]}
\toprule
\textbf{Size} & \textbf{Speedup CPU} & \textbf{Speedup OMP} & \textbf{Speedup GPU} \\
\midrule
"""
    for _, row in subset.iterrows():
        size_label = f"$2^{{{int(np.log2(row['Size']))}}}$"
        latex_content += f"{size_label} & {row['speedup_cpu']:.3f} & {row['speedup_omp']:.3f} & {row['speedup_gpu']:.3f} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}

\end{document}
"""
    with open(filename, "w") as f:
        f.write(latex_content)
    print(f"LaTeX file saved as {filename}")
