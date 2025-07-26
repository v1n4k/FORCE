"""
Data distribution utilities for federated learning.
Implements non-IID data splitting using Dirichlet distribution as described in:
"Federated Learning with Non-IID Data" (https://arxiv.org/abs/1806.00582)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import Subset
from typing import List, Dict, Tuple, Optional
import json
from PIL import Image
import os


def dirichlet_non_iid_split(dataset, num_clients: int, alpha: float = 0.5, seed: int = 42) -> List[List[int]]:
    """
    Create non-IID data split using Dirichlet distribution.
    
    Based on the method described in "Federated Learning with Non-IID Data" (arXiv:1806.00582).
    Lower alpha values create more heterogeneous (non-IID) distributions.
    
    Args:
        dataset: The dataset to split
        num_clients: Number of clients
        alpha: Dirichlet distribution concentration parameter
               - alpha = 0.1: highly non-IID
               - alpha = 1.0: moderately non-IID  
               - alpha = 10.0: close to IID
        seed: Random seed for reproducibility
        
    Returns:
        client_indices: List of data indices for each client
    """
    np.random.seed(seed)
    
    # Get labels from dataset
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # For HuggingFace datasets
        labels = np.array([example['labels'] for example in dataset])
    
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute data according to Dirichlet distribution
    for class_id in range(num_classes):
        # Get indices of samples belonging to this class
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Calculate the number of samples each client gets for this class
        class_size = len(class_indices)
        client_splits = (np.cumsum(proportions) * class_size).astype(int)[:-1]
        
        # Split the class indices among clients
        split_indices = np.split(class_indices, client_splits)
        
        # Assign to clients
        for client_id, indices in enumerate(split_indices):
            client_indices[client_id].extend(indices.tolist())
    
    # Shuffle each client's data to mix classes
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices


def biased_dirichlet_split(dataset, num_clients: int, 
                          client_quotas: List[float] = None,
                          client_alphas: List[float] = None,
                          seed: int = 42) -> List[List[int]]:
    """
    Create biased non-IID data split with configurable client quotas and individual alpha values.
    
    Enables research scenarios like:
    - Dominant client with large data share but highly skewed labels
    - Multiple balanced clients with smaller data shares but uniform labels
    
    Args:
        dataset: The dataset to split
        num_clients: Number of clients
        client_quotas: Data proportion for each client (must sum to 1.0)
                      If None, uses equal split [1/num_clients, ...]
        client_alphas: Dirichlet alpha value for each client
                      If None, uses standard alpha=0.5 for all clients
        seed: Random seed for reproducibility
        
    Returns:
        client_indices: List of data indices for each client
        
    Example:
        # Client 0: 60% data with high skew (α=0.3)
        # Others: 10% each with balanced labels (α=10.0)
        quotas = [0.6, 0.1, 0.1, 0.1, 0.1]
        alphas = [0.3, 10.0, 10.0, 10.0, 10.0]
        indices = biased_dirichlet_split(dataset, 5, quotas, alphas)
    """
    np.random.seed(seed)
    
    # Validate and set default parameters
    if client_quotas is None:
        client_quotas = [1.0 / num_clients] * num_clients
    if client_alphas is None:
        client_alphas = [0.5] * num_clients
        
    # Validation
    if len(client_quotas) != num_clients:
        raise ValueError(f"client_quotas length ({len(client_quotas)}) must equal num_clients ({num_clients})")
    if len(client_alphas) != num_clients:
        raise ValueError(f"client_alphas length ({len(client_alphas)}) must equal num_clients ({num_clients})")
    if abs(sum(client_quotas) - 1.0) > 1e-6:
        raise ValueError(f"client_quotas must sum to 1.0, got {sum(client_quotas)}")
    if any(q <= 0 for q in client_quotas):
        raise ValueError("All client_quotas must be positive")
    if any(a <= 0 for a in client_alphas):
        raise ValueError("All client_alphas must be positive")
    
    # Get labels from dataset
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # For HuggingFace datasets
        labels = np.array([example['labels'] for example in dataset])
    
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    # Calculate target samples per client based on quotas
    target_client_sizes = [int(quota * num_samples) for quota in client_quotas]
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute data according to each client's alpha and quota
    for class_id in range(num_classes):
        # Get indices of samples belonging to this class
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        class_size = len(class_indices)
        
        # Calculate how many samples each client should get from this class
        # This considers both quota (how much data they get) and alpha (how skewed their distribution is)
        
        # Step 1: For each client, sample from Dirichlet to get their preference for this class
        client_class_preferences = []
        for client_id in range(num_clients):
            # Each client has their own alpha - lower alpha = more skewed preferences
            alpha_vec = np.repeat(client_alphas[client_id], num_classes)
            preference = np.random.dirichlet(alpha_vec)[class_id]
            client_class_preferences.append(preference)
        
        # Step 2: Normalize preferences by client quotas to get actual allocations
        total_preference_weighted = sum(pref * quota for pref, quota in zip(client_class_preferences, client_quotas))
        
        if total_preference_weighted > 0:
            client_class_allocations = []
            for client_id in range(num_clients):
                allocation = (client_class_preferences[client_id] * client_quotas[client_id] / total_preference_weighted) * class_size
                client_class_allocations.append(int(allocation))
            
            # Handle rounding errors - distribute remaining samples
            allocated_total = sum(client_class_allocations)
            remaining = class_size - allocated_total
            for i in range(remaining):
                client_class_allocations[i % num_clients] += 1
        else:
            # Fallback: distribute equally
            client_class_allocations = [class_size // num_clients] * num_clients
            for i in range(class_size % num_clients):
                client_class_allocations[i] += 1
        
        # Step 3: Assign class samples to clients
        start_idx = 0
        for client_id, allocation in enumerate(client_class_allocations):
            end_idx = start_idx + allocation
            client_indices[client_id].extend(class_indices[start_idx:end_idx].tolist())
            start_idx = end_idx
    
    # Shuffle each client's data to mix classes
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices


def analyze_data_distribution(dataset, client_indices: List[List[int]], 
                            dataset_name: str) -> Dict[str, any]:
    """
    Analyze the data distribution across clients.
    
    Args:
        dataset: The original dataset
        client_indices: List of data indices for each client
        dataset_name: Name of the dataset for labeling
        
    Returns:
        analysis: Dictionary containing distribution statistics
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([example['labels'] for example in dataset])
    
    num_classes = len(np.unique(labels))
    num_clients = len(client_indices)
    
    # Calculate class distribution for each client
    client_class_counts = np.zeros((num_clients, num_classes))
    client_sizes = []
    
    for client_id, indices in enumerate(client_indices):
        client_labels = labels[indices]
        client_sizes.append(len(indices))
        
        for class_id in range(num_classes):
            client_class_counts[client_id, class_id] = np.sum(client_labels == class_id)
    
    # Calculate overall statistics
    total_samples = len(labels)
    global_class_dist = np.array([np.sum(labels == i) for i in range(num_classes)])
    global_class_ratio = global_class_dist / total_samples
    
    # Calculate entropy for each client (measure of class diversity)
    client_entropies = []
    for client_id in range(num_clients):
        client_dist = client_class_counts[client_id] / client_sizes[client_id]
        # Avoid log(0) by adding small epsilon
        client_dist = client_dist + 1e-12
        entropy = -np.sum(client_dist * np.log2(client_dist + 1e-12))
        client_entropies.append(entropy)
    
    # Calculate Earth Mover's Distance (EMD) approximation using L1 distance
    # This measures how different each client's distribution is from the global distribution
    emd_distances = []
    for client_id in range(num_clients):
        client_dist = client_class_counts[client_id] / client_sizes[client_id]
        emd = np.sum(np.abs(client_dist - global_class_ratio))
        emd_distances.append(emd)
    
    analysis = {
        'dataset_name': dataset_name,
        'num_clients': num_clients,
        'num_classes': num_classes,
        'total_samples': total_samples,
        'client_sizes': client_sizes,
        'client_class_counts': client_class_counts.tolist(),
        'global_class_distribution': global_class_dist.tolist(),
        'global_class_ratio': global_class_ratio.tolist(),
        'client_entropies': client_entropies,
        'emd_distances': emd_distances,
        'avg_entropy': np.mean(client_entropies),
        'std_entropy': np.std(client_entropies),
        'avg_emd': np.mean(emd_distances),
        'std_emd': np.std(emd_distances)
    }
    
    return analysis


def visualize_data_distribution(analysis: Dict[str, any], save_dir: Path, 
                               alpha: float, seed: int) -> None:
    """
    Create visualizations of the data distribution across clients.
    
    Args:
        analysis: Distribution analysis results
        save_dir: Directory to save plots
        alpha: Dirichlet alpha parameter used
        seed: Random seed used
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_clients = analysis['num_clients']
    num_classes = analysis['num_classes']
    client_class_counts = np.array(analysis['client_class_counts'])
    client_sizes = analysis['client_sizes']
    dataset_name = analysis['dataset_name']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Class distribution heatmap
    plt.figure(figsize=(12, 8))
    
    # Normalize by client size to show proportions
    client_proportions = client_class_counts / np.array(client_sizes).reshape(-1, 1)
    
    sns.heatmap(client_proportions, 
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd',
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Client {i}' for i in range(num_clients)],
                cbar_kws={'label': 'Proportion of Client Data'})
    
    plt.title(f'Data Distribution Heatmap\n'
              f'Dataset: {dataset_name}, Alpha: {alpha}, Clients: {num_clients}, Seed: {seed}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Clients', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / 'distribution_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Client sample sizes
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_clients), client_sizes, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Sample Size Distribution Across Clients\n'
              f'Dataset: {dataset_name}, Alpha: {alpha}, Clients: {num_clients}\n'
              f'Total Samples: {analysis["total_samples"]}, Avg: {np.mean(client_sizes):.0f}', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(num_clients))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'client_sample_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Entropy distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_clients), analysis['client_entropies'], 
            color='lightcoral', edgecolor='darkred', alpha=0.7)
    plt.axhline(y=analysis['avg_entropy'], color='red', linestyle='--', 
                label=f'Average: {analysis["avg_entropy"]:.3f}')
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Entropy (bits)', fontsize=12)
    plt.title(f'Class Diversity (Entropy) per Client\n'
              f'Dataset: {dataset_name}, Alpha: {alpha}, Clients: {num_clients}\n'
              f'Higher entropy = more diverse classes', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(num_clients))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'client_entropy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Earth Mover's Distance from global distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_clients), analysis['emd_distances'], 
            color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    plt.axhline(y=analysis['avg_emd'], color='green', linestyle='--', 
                label=f'Average: {analysis["avg_emd"]:.3f}')
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('EMD Distance from Global Distribution', fontsize=12)
    plt.title(f'Non-IID Degree per Client\n'
              f'Dataset: {dataset_name}, Alpha: {alpha}, Clients: {num_clients}\n'
              f'Higher EMD = more different from global distribution', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(num_clients))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'client_emd_distance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Summary statistics plot - combines the four individual plots
    # Make sure all individual plots have been created
    individual_plots = [
        'distribution_heatmap.png',
        'client_sample_sizes.png', 
        'client_entropy.png',
        'client_emd_distance.png'
    ]
    
    # Load the individual plot images
    images = []
    for plot_name in individual_plots:
        plot_path = save_dir / plot_name
        if plot_path.exists():
            images.append(Image.open(plot_path))
    
    if len(images) == 4:
        # Create combined figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        
        # Remove axes and display images
        for ax, img in zip(axes.flat, images):
            ax.imshow(img)
            ax.axis('off')
        
        plt.suptitle(f'Data Distribution Summary\n'
                     f'Dataset: {dataset_name}, Alpha: {alpha}, Clients: {num_clients}', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"Warning: Could not create summary statistics plot. Missing individual plots.")
    
    print(f"Data distribution visualizations saved to {save_dir}")


def save_distribution_analysis(analysis: Dict[str, any], save_path: Path) -> None:
    """
    Save the distribution analysis results to a JSON file.
    
    Args:
        analysis: Distribution analysis results
        save_path: Path to save the JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(analysis, f, indent=4)
    
    print(f"Distribution analysis saved to {save_path}")


def create_client_subsets(dataset, client_indices: List[List[int]]) -> List[Subset]:
    """
    Create PyTorch Subset objects for each client.
    
    Args:
        dataset: The original dataset
        client_indices: List of data indices for each client
        
    Returns:
        client_subsets: List of Subset objects for each client
    """
    client_subsets = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        client_subsets.append(subset)
    
    return client_subsets 