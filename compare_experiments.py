#!/usr/bin/env python3
"""
Interactive experiment comparison tool for federated learning results.
Allows users to select experiments and generate comparison plots.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class ExperimentInfo:
    """Container for experiment information."""
    def __init__(self, path: Path):
        self.path = path
        self.results = self._load_results()
        self.config = self.results.get('config', {}) if self.results else {}
        
    def _load_results(self) -> Optional[Dict]:
        """Load experiment results from JSON file."""
        results_path = self.path / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        return None
    
    @property
    def method(self) -> str:
        return self.config.get('method', 'unknown')
    
    @property
    def dataset(self) -> str:
        return self.config.get('dataset', 'unknown')
    
    @property
    def alpha(self) -> Optional[float]:
        return self.config.get('alpha', None)
    
    @property
    def num_rounds(self) -> int:
        return self.config.get('num_rounds', 0)
    
    @property
    def num_clients(self) -> int:
        return self.config.get('num_clients', 0)
    
    @property
    def best_accuracy(self) -> float:
        return self.results.get('best_accuracy', 0.0) if self.results else 0.0
    
    @property
    def timestamp(self) -> str:
        # Extract timestamp from directory name (first part before underscore)
        parts = self.path.name.split('_')
        if len(parts) >= 3:
            return f"{parts[0]}_{parts[1]}_{parts[2]}"
        return "unknown"
    
    def get_display_name(self) -> str:
        """Get a human-readable display name for the experiment."""
        return f"{self.method:15} | {self.dataset:10} | α={self.alpha:4} | rounds={self.num_rounds:3} | acc={self.best_accuracy:.4f} | {self.timestamp}"


def scan_experiments(base_dir: Path = Path("experiments")) -> List[ExperimentInfo]:
    """Scan for all valid experiments in the base directory."""
    experiments = []
    
    # Handle both direct experiment folders and alpha-grouped folders
    for item in base_dir.rglob("results.json"):
        exp_dir = item.parent
        exp_info = ExperimentInfo(exp_dir)
        if exp_info.results:  # Only add if results loaded successfully
            experiments.append(exp_info)
    
    # Sort by timestamp (newest first)
    experiments.sort(key=lambda x: x.path.name, reverse=True)
    return experiments


def display_experiments(experiments: List[ExperimentInfo]) -> None:
    """Display experiments in a readable format."""
    print("\n" + "="*100)
    print("AVAILABLE EXPERIMENTS")
    print("="*100)
    print(f"{'  #':3} | {'Method':15} | {'Dataset':10} | {'Alpha':6} | {'Rounds':10} | {'Accuracy':10} | {'Timestamp':20}")
    print("-"*100)
    
    for i, exp in enumerate(experiments):
        print(f"{i+1:3} | {exp.get_display_name()}")
    
    print("="*100)


def get_user_selections(experiments: List[ExperimentInfo]) -> List[ExperimentInfo]:
    """Get user's experiment selections."""
    print("\nSelect experiments to compare (e.g., '1,3,5' or '1-5' or 'all'):")
    selection = input("> ").strip()
    
    selected_indices = []
    
    if selection.lower() == 'all':
        return experiments
    
    # Parse selection
    for part in selection.split(','):
        part = part.strip()
        if '-' in part:
            # Range selection
            try:
                start, end = map(int, part.split('-'))
                selected_indices.extend(range(start-1, end))
            except:
                print(f"Invalid range: {part}")
        else:
            # Single selection
            try:
                idx = int(part) - 1
                if 0 <= idx < len(experiments):
                    selected_indices.append(idx)
                else:
                    print(f"Invalid index: {part}")
            except:
                print(f"Invalid selection: {part}")
    
    # Remove duplicates and sort
    selected_indices = sorted(list(set(selected_indices)))
    
    return [experiments[i] for i in selected_indices]


def validate_comparability(experiments: List[ExperimentInfo]) -> Tuple[bool, str]:
    """Check if experiments can be compared."""
    if len(experiments) < 2:
        return False, "Need at least 2 experiments to compare"
    
    # Check same dataset (e.g., all sst2, all qqp, etc.)
    datasets = set(exp.dataset for exp in experiments)
    if len(datasets) > 1:
        return False, f"Cannot compare different datasets: {datasets}. All experiments must use the same dataset."
    
    # Check same alpha value (same data distribution)
    alphas = set(exp.alpha for exp in experiments)
    if len(alphas) > 1:
        return False, f"Cannot compare different alpha values: {alphas}. All experiments must have the same non-IID distribution (alpha)."
    
    # Check same number of rounds
    rounds = set(exp.num_rounds for exp in experiments)
    if len(rounds) > 1:
        return False, f"Cannot compare different number of rounds: {rounds}. All experiments must have the same number of federated learning rounds."
    
    # All checks passed
    dataset = experiments[0].dataset
    alpha = experiments[0].alpha
    num_rounds = experiments[0].num_rounds
    return True, f"All experiments are comparable: Dataset={dataset}, Alpha={alpha}, Rounds={num_rounds}"


def plot_comparison(experiments: List[ExperimentInfo], save_path: Optional[str] = None) -> None:
    """Generate comparison plot for selected experiments."""
    plt.figure(figsize=(12, 8))
    
    # Color palette for different methods
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    # Check if this is MNLI dataset
    dataset = experiments[0].dataset
    is_mnli = dataset in ['mnli_matched', 'mnli_mismatched']
    
    for i, exp in enumerate(experiments):
        if not exp.results or 'rounds' not in exp.results:
            print(f"Warning: No round data for {exp.method}")
            continue
        
        # Extract round data
        rounds = [r["round"] for r in exp.results["rounds"]]
        accuracies = [r["accuracy"] for r in exp.results["rounds"]]
        
        if is_mnli and "auxiliary_accuracy" in exp.results["rounds"][0]:
            # For MNLI, plot both matched and mismatched
            auxiliary_accs = [r.get("auxiliary_accuracy", 0) for r in exp.results["rounds"]]
            
            # Determine which is primary
            primary_type = exp.results.get('primary_eval_type', 'standard')
            
            if primary_type == 'mismatched':
                # Primary is mismatched, auxiliary is matched
                plt.plot(rounds, accuracies, '-o', linewidth=2, markersize=6, 
                        color=colors[i % len(colors)], label=f'{exp.method} (Mismatched)', alpha=0.8)
                plt.plot(rounds, auxiliary_accs, '--s', linewidth=2, markersize=6, 
                        color=colors[i % len(colors)], label=f'{exp.method} (Matched)', alpha=0.6)
            else:
                # Primary is matched, auxiliary is mismatched
                plt.plot(rounds, accuracies, '-o', linewidth=2, markersize=6, 
                        color=colors[i % len(colors)], label=f'{exp.method} (Matched)', alpha=0.8)
                plt.plot(rounds, auxiliary_accs, '--s', linewidth=2, markersize=6, 
                        color=colors[i % len(colors)], label=f'{exp.method} (Mismatched)', alpha=0.6)
        else:
            # Non-MNLI dataset, plot single curve
            plt.plot(rounds, accuracies, '-o', linewidth=2, markersize=6, 
                    color=colors[i % len(colors)], label=exp.method, alpha=0.8)
    
    # Configure plot
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlabel('Federated Learning Round', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    
    # Title with experiment details
    alpha = experiments[0].alpha
    if is_mnli:
        plt.title(f'Method Comparison on MNLI (Matched & Mismatched) (α={alpha})', fontsize=16, fontweight='bold')
    else:
        plt.title(f'Method Comparison on {dataset} (α={alpha})', fontsize=16, fontweight='bold')
    
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_mnli:
            filename = f"comparison_mnli_alpha{alpha}_{timestamp}.png"
        else:
            filename = f"comparison_{dataset}_alpha{alpha}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {filename}")
    
    plt.show()


def main():
    """Main interactive loop."""
    print("\n" + "="*50)
    print("FEDERATED LEARNING EXPERIMENT COMPARISON TOOL")
    print("="*50)
    
    # Scan for experiments
    print("\nScanning for experiments...")
    experiments = scan_experiments()
    
    if not experiments:
        print("No experiments found in 'experiments' directory!")
        return
    
    print(f"Found {len(experiments)} experiments.")
    
    while True:
        # Display available experiments
        display_experiments(experiments)
        
        # Get user selections
        selected = get_user_selections(experiments)
        
        if not selected:
            print("No experiments selected.")
            continue
        
        print(f"\nSelected {len(selected)} experiments:")
        for exp in selected:
            print(f"  - {exp.method} on {exp.dataset} (α={exp.alpha})")
        
        # Validate comparability
        valid, message = validate_comparability(selected)
        
        if not valid:
            print(f"\n❌ Cannot compare: {message}")
            print("\n📋 Comparison Requirements:")
            print("   1. Same dataset (e.g., all sst2 or all qqp)")
            print("   2. Same alpha value (same non-IID data distribution)")
            print("   3. Same number of federated learning rounds")
            print("\nPlease select experiments that meet ALL these criteria.")
        else:
            print(f"\n✅ {message}")
            
            # Generate comparison plot
            print("\nGenerating comparison plot...")
            plot_comparison(selected)
        
        # Ask if user wants to continue
        print("\nDo you want to compare other experiments? (y/n)")
        if input("> ").strip().lower() != 'y':
            break
    
    print("\nThank you for using the comparison tool!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1) 