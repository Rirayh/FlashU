import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def load_ffn_scores(json_file):
    """Load FFN scores indexed by layer."""
    with open(json_file, "r") as file:
        scores = json.load(file)

    return {
        int(key.split(".")[2]): np.asarray(values, dtype=np.float64)
        for key, values in scores.items()
        if key.endswith(".mlp")
    }


def compare_top_half(generation_scores, understanding_scores):
    """Classify each layer's top-50% FFN neurons by task."""
    results = []
    for layer in sorted(generation_scores):
        if layer not in understanding_scores:
            raise ValueError(f"Layer {layer} is missing from understanding scores")

        generation = generation_scores[layer]
        understanding = understanding_scores[layer]
        if generation.shape != understanding.shape:
            raise ValueError(f"Layer {layer} has different FFN sizes")

        neuron_count = generation.size
        top_count = neuron_count // 2
        # Select the top 50% independently for each task.
        generation_top = set(np.argsort(generation)[-top_count:])
        understanding_top = set(np.argsort(understanding)[-top_count:])

        # Split the selected neurons into the three Figure 1 groups.
        both = len(generation_top & understanding_top)
        generation_only = len(generation_top) - both
        understanding_only = len(understanding_top) - both
        scale = 100.0 / neuron_count
        results.append((
            layer,
            generation_only * scale,
            both * scale,
            understanding_only * scale,
        ))

    return results


def plot_comparison(results, output_file):
    """Create the FlashU Figure 1-style task comparison."""
    layers, generation_only, both, understanding_only = map(
        np.asarray, zip(*results)
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    # Center the shared group between the two task-specific groups.
    ax.barh(layers, -generation_only, left=-both / 2, color="#5B9BE6", label="Generation Only")
    ax.barh(layers, both, left=-both / 2, color="#A665C2", label="Both Tasks (Overlap)")
    ax.barh(layers, understanding_only, left=both / 2, color="#EF5B50", label="Understanding Only")

    for layer, gen, shared, und in results:
        ax.text(-shared / 2 - gen / 2, layer, f"{gen:.1f}", ha="center", va="center", color="white", fontsize=8)
        ax.text(0, layer, f"{shared:.1f}", ha="center", va="center", color="white", fontsize=8)
        ax.text(shared / 2 + und / 2, layer, f"{und:.1f}", ha="center", va="center", color="white", fontsize=8)

    ax.axvline(0, color="black", linewidth=0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):g}"))
    ax.set_yticks(layers)
    ax.set_xlabel("Percentage of Key FFN Neurons (%)")
    ax.set_ylabel("FFN Layer Index")
    ax.set_title("Task-Specific FFN Neuron Analysis")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8)
    fig.tight_layout()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Comparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare generation and understanding FFN scores")
    parser.add_argument("generation_json", help="Generation OBD score JSON")
    parser.add_argument("understanding_json", help="Understanding OBD score JSON")
    parser.add_argument("--output_file", default="obd_task_comparison.png")
    args = parser.parse_args()

    generation_scores = load_ffn_scores(args.generation_json)
    understanding_scores = load_ffn_scores(args.understanding_json)
    results = compare_top_half(generation_scores, understanding_scores)
    plot_comparison(results, args.output_file)


if __name__ == "__main__":
    main()
