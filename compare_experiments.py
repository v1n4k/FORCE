#!/usr/bin/env python3
"""
Interactive experiment comparison tool for federated learning results.
Allows users to select experiments and generate comparison plots.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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
        # Truncate method name if too long and add abbreviations for common long names
        method_display = self.method
        if len(method_display) > 18:
            # Create abbreviations for common long method names
            abbreviations = {
                'soft_constraint+muon': 'soft_constr+muon',
                'Newton_shulz+muon': 'Newton_sz+muon',
                'soft_constraint': 'soft_constr',
                'Newton_shulz': 'Newton_sz'
            }
            method_display = abbreviations.get(method_display, method_display[:18])
        
        return f"{method_display:18} | {self.dataset:12} | α={self.alpha:4} | R={self.num_rounds:2} | acc={self.best_accuracy:.4f} | {self.timestamp}"


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
    """Display experiments in a readable format with grouping."""
    print("\n" + "="*105)
    print("AVAILABLE EXPERIMENTS")
    print("="*105)
    print(f"{'  #':3} | {'Method':18} | {'Dataset':12} | {'Alpha':6} | {'R':2} | {'Accuracy':8} | {'Timestamp':17}")
    print("-"*105)
    
    # Group by dataset and alpha for better readability
    current_group = None
    for i, exp in enumerate(experiments):
        group_key = f"{exp.dataset}_α{exp.alpha}"
        if group_key != current_group:
            if current_group is not None:
                print("-"*105)
            current_group = group_key
        
        print(f"{i+1:3} | {exp.get_display_name()}")
    
    print("="*105)
    
    # Show summary of available datasets and alphas
    datasets = list(set(exp.dataset for exp in experiments))
    alphas = list(set(exp.alpha for exp in experiments if exp.alpha is not None))
    print(f"\n📊 Summary: {len(experiments)} experiments | Datasets: {sorted(datasets)} | Alpha values: {sorted(alphas)}")
    print("💡 Note: Only experiments with same dataset, alpha, and rounds can be compared.")


def get_user_selections(experiments: List[ExperimentInfo]) -> List[ExperimentInfo]:
    """Get user's experiment selections."""
    print("\n" + "="*80)
    print("🎯 SELECT EXPERIMENTS TO COMPARE")
    print("="*80)
    print("Examples:")
    print("  • Single experiments:     1,3,5")
    print("  • Range of experiments:   1-5")
    print("  • All experiments:        all")
    print("  • Mixed selection:        1,3,7-10,15")
    print("-"*80)
    print("💡 Remember: Only experiments with same dataset, alpha, and rounds can be compared!")
    print("="*80)
    
    while True:
        selection = input("Enter your selection > ").strip()
        
        if not selection:
            print("❌ Please enter a selection!")
            continue
            
        selected_indices = []
        
        if selection.lower() == 'all':
            return experiments
        
        # Parse selection
        valid = True
        for part in selection.split(','):
            part = part.strip()
            if '-' in part:
                # Range selection
                try:
                    start, end = map(int, part.split('-'))
                    if start > end:
                        print(f"❌ Invalid range: {part} (start > end)")
                        valid = False
                        break
                    selected_indices.extend(range(start-1, end))
                except:
                    print(f"❌ Invalid range format: {part}")
                    valid = False
                    break
            else:
                # Single selection
                try:
                    idx = int(part) - 1
                    if 0 <= idx < len(experiments):
                        selected_indices.append(idx)
                    else:
                        print(f"❌ Invalid index: {part} (valid range: 1-{len(experiments)})")
                        valid = False
                        break
                except:
                    print(f"❌ Invalid number: {part}")
                    valid = False
                    break
        
        if not valid:
            print("🔄 Please try again with a valid selection.")
            continue
            
        # Remove duplicates and sort
        selected_indices = sorted(list(set(selected_indices)))
        
        if not selected_indices:
            print("❌ No valid experiments selected!")
            continue
            
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


def get_display_method_name(method_name: str) -> str:
    """Map internal method names to display names for legends."""
    method_mapping = {
        'QR+muon': 'FORCE(QR)',
        'Newton_shulz+muon': 'FORCE(NS)',
        'soft_constraint+muon': 'FORCE',
        'ffa_lora': 'FFA-LoRA',
        'lora': 'FedIT'
    }
    return method_mapping.get(method_name, method_name)


def get_method_color(method_name: str) -> str:
    """Map method names to consistent colors across all plots."""
    color_mapping = {
        'QR+muon': 'tab:blue',
        'Newton_shulz+muon': 'tab:orange', 
        'soft_constraint+muon': 'tab:green',
        'ffa_lora': 'tab:red',
        'lora': 'tab:purple'
    }
    return color_mapping.get(method_name, 'tab:gray')


def get_subplot_params() -> Optional[Tuple[int, int]]:
    """Ask user if they want to add a subplot and get the range parameters."""
    while True:
        choice = input("\n🔍 Do you want to add a zoomed subplot to highlight differences? (y/n) > ").strip().lower()
        if choice in ['y', 'yes']:
            print("\n📊 Subplot will zoom into a specific range of rounds to magnify differences.")
            try:
                start_round = int(input("Enter start round for subplot (e.g., 12) > ").strip())
                end_round = int(input("Enter end round for subplot (e.g., 15) > ").strip())
                
                if start_round >= end_round:
                    print("❌ Start round must be less than end round!")
                    continue
                if start_round < 1:
                    print("❌ Start round must be at least 1!")
                    continue
                    
                return (start_round, end_round)
            except ValueError:
                print("❌ Please enter valid integer values!")
                continue
        elif choice in ['n', 'no']:
            return None
        else:
            print("❌ Please enter 'y' for yes or 'n' for no!")


def create_single_plot(experiments: List[ExperimentInfo], plot_type: str, title_suffix: str, subplot_range: Optional[Tuple[int, int]] = None):
    """Create a single comparison plot for the given experiments and plot type.
    
    Args:
        experiments: List of experiment info objects
        plot_type: 'matched', 'mismatched', or 'standard'
        title_suffix: Suffix for the plot title
        subplot_range: Optional tuple (start_round, end_round) for zoomed subplot
    
    Returns:
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors will be assigned based on method name for consistency
    
    dataset = experiments[0].dataset
    alpha = experiments[0].alpha
    
    # Store data for subplot if needed
    subplot_data = []
    
    for exp in experiments:
        if not exp.results or 'rounds' not in exp.results:
            print(f"Warning: No round data for {exp.method}")
            continue
        
        # Extract round data
        rounds = [r["round"] for r in exp.results["rounds"]]
        
        # Choose which accuracy to plot based on plot_type
        if plot_type == 'matched':
            # For MNLI matched plot
            primary_type = exp.results.get('primary_eval_type', 'standard')
            if primary_type == 'matched':
                accuracies = [r["accuracy"] for r in exp.results["rounds"]]
            else:
                accuracies = [r.get("auxiliary_accuracy", 0) for r in exp.results["rounds"]]
        elif plot_type == 'mismatched':
            # For MNLI mismatched plot
            primary_type = exp.results.get('primary_eval_type', 'standard')
            if primary_type == 'mismatched':
                accuracies = [r["accuracy"] for r in exp.results["rounds"]]
            else:
                accuracies = [r.get("auxiliary_accuracy", 0) for r in exp.results["rounds"]]
        else:
            # For standard (non-MNLI) datasets
            accuracies = [r["accuracy"] for r in exp.results["rounds"]]
        
        # Get consistent color for this method
        method_color = get_method_color(exp.method)
        
        # Plot the main curve
        ax.plot(rounds, accuracies, '-o', linewidth=2, markersize=6, 
               color=method_color, label=get_display_method_name(exp.method), alpha=0.8)
        
        # Store data for subplot
        if subplot_range:
            subplot_data.append((rounds, accuracies, method_color, exp.method))
    
    # Configure main plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Federated Learning Round', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(f'Method Comparison on {title_suffix} (α={alpha})', fontsize=20, fontweight='bold')
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    
    # Add subplot if requested
    if subplot_range and subplot_data:
        start_round, end_round = subplot_range
        
        # Determine optimal subplot position based on data distribution
        # Check if curves are generally increasing (typical case)
        avg_start_accuracy = sum(data[1][0] for data in subplot_data if data[1]) / len([d for d in subplot_data if d[1]])
        avg_end_accuracy = sum(data[1][-1] for data in subplot_data if data[1]) / len([d for d in subplot_data if d[1]])
        
        if avg_end_accuracy > avg_start_accuracy + 0.05:  # Curves are increasing significantly
            # Place subplot in lower left where there's typically more space
            subplot_loc = 'lower left'
            bbox_anchor = (0.18, 0.05, 1, 1)
        else:
            # For flat or decreasing curves, place in upper left
            subplot_loc = 'upper left'
            bbox_anchor = (0.05, 0.70, 1, 1)
        
        # Create inset axes for subplot
        axins = inset_axes(ax, width="55%", height="55%", loc=subplot_loc, 
                          bbox_to_anchor=bbox_anchor, bbox_transform=ax.transAxes)
        
        # Plot data in subplot
        for rounds, accuracies, color, method in subplot_data:
            # Filter data for the specified range
            filtered_rounds = []
            filtered_accs = []
            for r, acc in zip(rounds, accuracies):
                if start_round <= r <= end_round:
                    filtered_rounds.append(r)
                    filtered_accs.append(acc)
            
            if filtered_rounds:  # Only plot if there's data in the range
                axins.plot(filtered_rounds, filtered_accs, '-o', linewidth=1.5, markersize=4, 
                          color=color, alpha=0.8)
        
        # Configure subplot
        axins.set_xlim(start_round, end_round)
        axins.xaxis.set_major_locator(MaxNLocator(integer=True))
        axins.grid(True, alpha=0.3)
        axins.tick_params(axis='both', which='major', labelsize=10)
        axins.set_title(f'Rounds {start_round}-{end_round}', fontsize=12)
        
        # Add indication lines from main plot to subplot
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_comparison(experiments: List[ExperimentInfo], save_path: Optional[str] = None) -> None:
    """Generate comparison plot(s) for selected experiments."""
    dataset = experiments[0].dataset
    alpha = experiments[0].alpha
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if this is MNLI dataset
    is_mnli = dataset in ['mnli_matched', 'mnli_mismatched']
    
    # Ask user about subplot functionality
    subplot_range = get_subplot_params()
    
    if is_mnli:
        # Create separate plots for matched and mismatched
        print("\n🎨 Generating separate plots for MNLI matched and mismatched...")
        
        # Matched plot
        fig_matched = create_single_plot(experiments, 'matched', 'MNLI Matched', subplot_range)
        if save_path:
            matched_path = save_path.replace('.png', '_matched.png')
        else:
            matched_path = f"comparison_mnli_matched_alpha{alpha}_{timestamp}.png"
        fig_matched.savefig(matched_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Matched plot saved to: {matched_path}")
        
        # Mismatched plot
        fig_mismatched = create_single_plot(experiments, 'mismatched', 'MNLI Mismatched', subplot_range)
        if save_path:
            mismatched_path = save_path.replace('.png', '_mismatched.png')
        else:
            mismatched_path = f"comparison_mnli_mismatched_alpha{alpha}_{timestamp}.png"
        fig_mismatched.savefig(mismatched_path, dpi=300, bbox_inches='tight')
        print(f"📊 Mismatched plot saved to: {mismatched_path}")
        
        plt.show()
        
    else:
        # Single plot for non-MNLI datasets
        fig = create_single_plot(experiments, 'standard', dataset, subplot_range)
        if save_path:
            filename = save_path
        else:
            filename = f"comparison_{dataset}_alpha{alpha}_{timestamp}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
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
        
        print(f"\n✅ Selected {len(selected)} experiments:")
        print("-"*60)
        for exp in selected:
            print(f"  • {exp.method:20} | {exp.dataset:10} | α={exp.alpha}")
        print("-"*60)
        
        # Validate comparability
        valid, message = validate_comparability(selected)
        
        if not valid:
            print(f"\n❌ COMPARISON ERROR: {message}")
            print("\n📋 Requirements for comparison:")
            print("   ✓ Same dataset (e.g., all qnli or all sst2)")
            print("   ✓ Same alpha value (same data distribution)")
            print("   ✓ Same number of federated learning rounds")
            print("\n🔄 Please select experiments that meet ALL these criteria.")
        else:
            print(f"\n✅ VALIDATION PASSED: {message}")
            
            # Generate comparison plot
            print("\n🎨 Generating comparison plot...")
            plot_comparison(selected)
            print("✅ Comparison completed!")
        
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