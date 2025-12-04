import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

# --- Hardware Specifications (for NVIDIA H100 SXM5 - FP64 theoretical peak) ---
# Memory Bandwidth: ~3.35 TB/s
PEAK_MEMORY_BW = 3350  # GB/s 
# FP64 FLOPs: ~33.5 TFLOP/s
PEAK_FLOP_RATE = 33500 # GFLOP/s 
RIDGE_POINT = PEAK_FLOP_RATE / PEAK_MEMORY_BW
# ---

def process_data(csv_file, label_prefix):
    """
    Reads a CSV and returns a dictionary of performance data (AI, GFLOPS) for the loops.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading '{csv_file}': {e}")
        return None

    # Separate data
    df_update = df[df['loop_name'] == 'update_loop'].copy()
    df_conv = df[df['loop_name'] == 'convergence_loop'].copy()

    if df_update.empty or df_conv.empty:
        print(f"Warning: '{csv_file}' missing required loop names.")
        return None

    # Convert TFLOPS to GFLOPS
    df_update['GFLOPS'] = df_update['TFLOPS'] * 1000
    df_conv['GFLOPS'] = df_conv['TFLOPS'] * 1000

    return {
        'update': {
            'ai': df_update['AI'].mean(),
            'perf': df_update['GFLOPS'].mean(),
            'label': f'{label_prefix} Update'
        },
        'conv': {
            'ai': df_conv['AI'].mean(),
            'perf': df_conv['GFLOPS'].mean(),
            'label': f'{label_prefix} Convergence'
        }
    }

def create_roofline_plot(task1_csv, task2_csv, output_file):
    print(f"Processing Task 1: {task1_csv}")
    data_t1 = process_data(task1_csv, "Task 1 (CUDA)")
    
    data_t2 = None
    if task2_csv:
        print(f"Processing Task 2: {task2_csv}")
        data_t2 = process_data(task2_csv, "Task 2 (MPI+CUDA)")

    if not data_t1 and not data_t2:
        print("No valid data found to plot.")
        return

    # --- Create the Plot ---
    plt.figure(figsize=(12, 8))
    ai_range = np.logspace(-2, 4, 1000)
    
    # Calculate roofline
    memory_bound = ai_range * PEAK_MEMORY_BW
    compute_bound = np.full_like(ai_range, PEAK_FLOP_RATE)
    roofline = np.minimum(memory_bound, compute_bound)
    
    # Plot Roofline
    plt.loglog(ai_range, roofline, 'r-', linewidth=3, label='H100 Roofline')
    plt.loglog(RIDGE_POINT, PEAK_FLOP_RATE, 'ro', markersize=10)

    # --- Plot Task 1 ---
    if data_t1:
        # Update Loop (Blue)
        plt.loglog(data_t1['update']['ai'], data_t1['update']['perf'], 'bo', markersize=12, label=data_t1['update']['label'])
        plt.annotate(f"AI={data_t1['update']['ai']:.2f}", (data_t1['update']['ai'], data_t1['update']['perf']), xytext=(10, -15), textcoords='offset points')
        
        # Convergence Loop (Blue Triangle)
        plt.loglog(data_t1['conv']['ai'], data_t1['conv']['perf'], 'b^', markersize=12, label=data_t1['conv']['label'])

    # --- Plot Task 2 ---
    if data_t2:
        # Update Loop (Green)
        plt.loglog(data_t2['update']['ai'], data_t2['update']['perf'], 'go', markersize=12, label=data_t2['update']['label'])
        plt.annotate(f"AI={data_t2['update']['ai']:.2f}", (data_t2['update']['ai'], data_t2['update']['perf']), xytext=(10, -15), textcoords='offset points')
        
        # Convergence Loop (Green Triangle)
        plt.loglog(data_t2['conv']['ai'], data_t2['conv']['perf'], 'g^', markersize=12, label=data_t2['conv']['label'])

    # Formatting
    plt.xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
    plt.ylabel('Performance (GFLOP/s)', fontsize=12)
    plt.title('Roofline Analysis: Single GPU (Task 1) vs Multi-GPU (Task 2)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=11, loc='lower right')
    
    plt.text(0.02, 1000, 'Memory Bound', fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    plt.text(50, 20000, 'Compute Bound', fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.xlim(0.01, 1000)
    plt.ylim(10, 40000)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved as: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate comparison roofline plot')
    parser.add_argument('task1_csv', help='Path to Task 1 CSV results')
    parser.add_argument('task2_csv', nargs='?', help='Path to Task 2 CSV results (optional)')
    parser.add_argument('-o', '--output', default='roofline_compare.png', help='Output filename')
    
    args = parser.parse_args()
    create_roofline_plot(args.task1_csv, args.task2_csv, args.output)

if __name__ == "__main__":
    main()