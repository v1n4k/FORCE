"""
Plotting utilities for federated learning experiments.
Focused on single experiment analysis and visualization.
For multi-experiment comparison, use compare_experiments.py instead.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from matplotlib.ticker import MaxNLocator


def load_experiment_results(exp_dir: Path) -> Dict:
    """
    Load experiment results from directory.
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        results: Dictionary containing experiment results
    """
    # Load from JSON file
    json_path = exp_dir / "results.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    
    raise FileNotFoundError(f"No results found in {exp_dir}")


def plot_accuracy_over_rounds(results: Dict, save_path: Optional[Path] = None):
    """
    Plot accuracy over federated learning rounds.
    
    Args:
        results: Experiment results dictionary
        save_path: Path to save the plot (optional)
    """
    rounds = [r["round"] for r in results["rounds"]]
    accuracies = [r["accuracy"] for r in results["rounds"]]
    
    plt.figure(figsize=(10, 6))
    
    # Check if this is MNLI dataset with auxiliary accuracy
    is_mnli = "auxiliary_accuracy" in results["rounds"][0] if results["rounds"] else False
    
    if is_mnli:
        # For MNLI, plot both matched and mismatched accuracies
        auxiliary_accs = [r.get("auxiliary_accuracy", 0) for r in results["rounds"]]
        
        # Determine which is primary and which is auxiliary
        primary_type = results.get('primary_eval_type', 'standard')
        if primary_type == 'mismatched':
            # Primary is mismatched, auxiliary is matched
            plt.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=8, label='Mismatched (Primary)')
            plt.plot(rounds, auxiliary_accs, 'g-s', linewidth=2, markersize=8, label='Matched')
            
            # Mark best accuracies
            best_round = results["best_round"]
            best_acc = results["best_accuracy"]
            plt.plot(best_round, best_acc, 'r*', markersize=15, label=f'Best Mismatched: {best_acc:.4f}')
            
            if "best_auxiliary_round" in results:
                best_aux_round = results["best_auxiliary_round"]
                best_aux_acc = results["best_auxiliary_accuracy"]
                plt.plot(best_aux_round, best_aux_acc, 'm*', markersize=15, label=f'Best Matched: {best_aux_acc:.4f}')
        else:
            # Primary is matched, auxiliary is mismatched
            plt.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=8, label='Matched (Primary)')
            plt.plot(rounds, auxiliary_accs, 'g-s', linewidth=2, markersize=8, label='Mismatched')
            
            # Mark best accuracies
            best_round = results["best_round"]
            best_acc = results["best_accuracy"]
            plt.plot(best_round, best_acc, 'r*', markersize=15, label=f'Best Matched: {best_acc:.4f}')
            
            if "best_auxiliary_round" in results:
                best_aux_round = results["best_auxiliary_round"]
                best_aux_acc = results["best_auxiliary_accuracy"]
                plt.plot(best_aux_round, best_aux_acc, 'm*', markersize=15, label=f'Best Mismatched: {best_aux_acc:.4f}')
    else:
        # Non-MNLI dataset, plot single accuracy curve
        plt.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=8, label='Accuracy')
        
        # Mark best accuracy
        best_round = results["best_round"]
        best_acc = results["best_accuracy"]
        plt.plot(best_round, best_acc, 'r*', markersize=15, label=f'Best: {best_acc:.4f}')
        
        # Add horizontal line for best accuracy
        plt.axhline(y=best_acc, color='r', linestyle='--', alpha=0.5)
    
    # Ensure x-axis shows only integers
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlabel('Federated Learning Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    
    # Update title based on dataset
    dataset_name = results["config"]["dataset"]
    if is_mnli:
        plt.title(f'{results["config"]["method"]} on MNLI (Matched & Mismatched)', fontsize=14)
    else:
        plt.title(f'{results["config"]["method"]} on {dataset_name}', fontsize=14)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_experiment_report(exp_dir: Path, output_dir: Optional[Path] = None):
    """
    Generate a comprehensive report with all plots for a single experiment.
    
    Args:
        exp_dir: Path to experiment directory
        output_dir: Directory to save report plots (optional)
    """
    results = load_experiment_results(exp_dir)
    
    if output_dir is None:
        output_dir = exp_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Plot accuracy over rounds (handles MNLI dual curves automatically)
    plot_accuracy_over_rounds(results, save_path=output_dir / "accuracy_over_rounds.png")
    
    # Generate summary statistics
    summary_stats = {
        "Method": results["config"]["method"],
        "Dataset": results["config"]["dataset"],
        "Model": results["config"]["model_name"],
        "Num Clients": results["config"]["num_clients"],
        "Num Rounds": results["config"]["num_rounds"],
        "Best Accuracy": results["best_accuracy"],
        "Best Round": results["best_round"],
        "Final Accuracy": results["final_accuracy"],
        "Improvement": results["best_accuracy"] - results["rounds"][0]["accuracy"] if results["rounds"] else 0
    }
    
    # Add MNLI-specific information if available
    if "best_auxiliary_accuracy" in results:
        primary_type = results.get('primary_eval_type', 'standard')
        if primary_type == 'mismatched':
            summary_stats["Best Mismatched"] = results["best_accuracy"]
            summary_stats["Best Matched"] = results["best_auxiliary_accuracy"]
        else:
            summary_stats["Best Matched"] = results["best_accuracy"]
            summary_stats["Best Mismatched"] = results["best_auxiliary_accuracy"]
    
    # Save summary as text
    with open(output_dir / "summary.txt", "w") as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Report generated in {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot federated learning experiment results")
    parser.add_argument("--exp_dir", type=str, help="Path to single experiment directory")
    parser.add_argument("--output_dir", type=str, help="Directory to save plots")
    
    args = parser.parse_args()
    
    if args.exp_dir:
        # Single experiment report
        exp_dir = Path(args.exp_dir)
        output_dir = Path(args.output_dir) if args.output_dir else None
        generate_experiment_report(exp_dir, output_dir)
    else:
        print("Please provide --exp_dir for single experiment analysis")
        print("For multi-experiment comparison, use compare_experiments.py")
        parser.print_help() 