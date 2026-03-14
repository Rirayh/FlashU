import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import time

# ==============================================================================
#  Core acceleration function: compute percentile ranks using NumPy vectorization
# ==============================================================================
def calculate_percentile_ranks_numpy(score_matrix, all_scores_global):
    """
    Efficiently compute global percentile rank for each score using NumPy.
    
    Args:
        score_matrix (np.ndarray): 2D matrix of raw scores.
        all_scores_global (np.ndarray): 1D array of all raw scores for the same component type.
        
    Returns:
        np.ndarray: Matrix of the same shape as score_matrix, with values 0-100 (percentile rank).
    """
    sorted_unique_scores = np.unique(all_scores_global)
    ranks = np.searchsorted(sorted_unique_scores, score_matrix, side='left')
    percentiles = (ranks / (len(sorted_unique_scores) - 1 + 1e-9)) * 100
    return percentiles
# ==============================================================================


def plot_heatmap(data_matrix, title, xlabel, ylabel, output_filename, cmap="viridis", vmin=0, vmax=100):
    """Plot and save a heatmap."""
    if data_matrix.size == 0:
        print(f"Skipping heatmap for '{title}' as data is empty.")
        return
        
    plt.figure(figsize=(20, 12))
    cbar_label = "Global Importance Percentile (0=Least, 100=Most)"
    sns.heatmap(data_matrix, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'label': cbar_label})
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"Heatmap saved to {output_filename}")


def plot_sorted_distribution(scores_list, title, output_filename, color='blue'):
    """Plot and save a sorted importance distribution."""
    if not scores_list:
        print(f"Skipping distribution plot for '{title}' as data is empty.")
        return
    
    sorted_scores = sorted(scores_list, reverse=True)
    total_components = len(sorted_scores)
    percentiles_x_axis = np.linspace(0, 100, num=total_components)

    plt.figure(figsize=(18, 10))
    plt.plot(percentiles_x_axis, sorted_scores, color=color, linewidth=2)
    
    plt.yscale('log')
    plt.title(title, fontsize=20)
    plt.xlabel("Component Percentile (Sorted, 0% = Most Important, 100% = Least Important)", fontsize=15)
    plt.ylabel("Importance Score (Log Scale)", fontsize=15)
    plt.grid(True, which="both", ls="--")

    ratios_to_mark = [10, 20, 30, 35, 40, 50]
    
    try:
        threshold_scores = np.percentile(scores_list, [100 - p for p in ratios_to_mark])
        
        for p, score_threshold in zip(ratios_to_mark, threshold_scores):
            plt.axvline(x=p, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            plt.axhline(y=score_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            plt.text(p + 0.5, sorted_scores[0], f'Top {p}%', rotation=90, 
                     verticalalignment='bottom', color='red', alpha=0.9)
            plt.text(percentiles_x_axis[-1] * 0.9, score_threshold * 1.1, f'Score: {score_threshold:.3f}', 
                     horizontalalignment='right', color='red', alpha=0.9)
    except IndexError:
        print(f"Warning: Could not calculate percentiles for {title}. Skipping annotations.")

    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"Distribution plot saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Visualize OBD scores from obd_scores.json")
    parser.add_argument("json_file", type=str, help="Path to the obd_scores.json file.")
    parser.add_argument("--output_dir", type=str, default="obd_visualizations", help="Directory to save the output plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        with open(args.json_file, 'r') as f:
            obd_scores = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {args.json_file} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.json_file}.")
        return

    # --- 1. Separate FFN and Attention data ---
    ffn_scores_by_layer = {}
    attn_scores_by_layer = {}
    for key, scores in obd_scores.items():
        parts = key.split('.')
        layer_idx = int(parts[2])
        component_type = parts[3]
        if component_type == 'mlp':
            ffn_scores_by_layer[layer_idx] = scores
        elif component_type == 'self_attn':
            attn_scores_by_layer[layer_idx] = scores
    
    # --- 2. Prepare all raw scores as NumPy arrays ---
    all_ffn_scores_np = np.array([score for scores in ffn_scores_by_layer.values() for score in scores], dtype=np.float32)
    all_attn_scores_np = np.array([score for scores in attn_scores_by_layer.values() for score in scores], dtype=np.float32)

    # --- 3. Compute percentile heatmaps ---
    
    # FFN
    if all_ffn_scores_np.size > 0:
        num_layers_ffn = max(ffn_scores_by_layer.keys()) + 1
        ffn_intermediate_size = len(next(iter(ffn_scores_by_layer.values())))
        ffn_raw_matrix = np.zeros((num_layers_ffn, ffn_intermediate_size), dtype=np.float32)
        for layer, scores in ffn_scores_by_layer.items():
            ffn_raw_matrix[layer, :len(scores)] = scores
        
        # Compute percentile heatmap (NumPy accelerated)
        print("Calculating FFN percentile heatmap (NumPy accelerated)...")
        start_time = time.time()
        ffn_percentile_matrix = calculate_percentile_ranks_numpy(ffn_raw_matrix, all_ffn_scores_np)
        print(f"  ...done in {time.time() - start_time:.2f} seconds.")

        plot_heatmap(ffn_percentile_matrix, "Global Importance Heatmap of FFN Neurons (by Percentile Rank)", "FFN Neuron Index", "Layer Index", os.path.join(args.output_dir, "ffn_importance_heatmap.png"))

    # Attention
    if all_attn_scores_np.size > 0:
        num_layers_attn = max(attn_scores_by_layer.keys()) + 1
        num_heads = len(next(iter(attn_scores_by_layer.values())))
        attn_raw_matrix = np.zeros((num_layers_attn, num_heads), dtype=np.float32)
        for layer, scores in attn_scores_by_layer.items():
            attn_raw_matrix[layer, :len(scores)] = scores
                
        # Compute percentile heatmap (NumPy accelerated)
        print("Calculating Attention Head percentile heatmap (NumPy accelerated)...")
        start_time = time.time()
        attn_percentile_matrix = calculate_percentile_ranks_numpy(attn_raw_matrix, all_attn_scores_np)
        print(f"  ...done in {time.time() - start_time:.2f} seconds.")

        plot_heatmap(attn_percentile_matrix, "Global Importance Heatmap of Attention Heads (by Percentile Rank)", "Attention Head Index", "Layer Index", os.path.join(args.output_dir, "attn_importance_heatmap.png"), cmap="rocket_r")

    # --- 4. Plot sorted distribution (using raw Python lists for precision) ---
    print("Calculating sorted distribution plots...")
    plot_sorted_distribution(all_ffn_scores_np.tolist(), "Sorted Importance Distribution of All FFN Neurons (Score vs. Rank)", os.path.join(args.output_dir, "ffn_sorted_distribution.png"), color='C0')
    plot_sorted_distribution(all_attn_scores_np.tolist(), "Sorted Importance Distribution of All Attention Heads (Score vs. Rank)", os.path.join(args.output_dir, "attn_sorted_distribution.png"), color='C1')
    
    print("\nVisualization complete. Check the 'obd_visualizations' directory.")


if __name__ == "__main__":
    main()
