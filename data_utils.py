"""
Refactored data utilities for federated learning.
Simplified to contain only core data loading functionality.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
from data_distribution import (
    dirichlet_non_iid_split, 
    analyze_data_distribution, 
    visualize_data_distribution,
    save_distribution_analysis,
    create_client_subsets
)


def tokenize_function(examples, tokenizer, dataset_name):
    """
    Tokenize examples based on the dataset type.
    
    Args:
        examples: The examples to tokenize
        tokenizer: The tokenizer to use
        dataset_name: Name of the dataset to determine tokenization approach
    
    Returns:
        Tokenized outputs
    """
    # Dataset-specific tokenization
    if dataset_name == "sst2":
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
    elif dataset_name == "cola":
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
    elif dataset_name == "qnli":
        outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=None)
    elif dataset_name == "qqp":
        outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=None)
    else:  # MNLI datasets
        outputs = tokenizer(examples['premise'], examples['hypothesis'], truncation=True, max_length=None)
    
    return outputs


def load_glue_dataset(dataset_name, model_name_or_path="roberta-base", batch_size=32):
    """
    Load GLUE dataset for training and evaluation.
    
    Args:
        dataset_name: Name of the GLUE dataset to load
        model_name_or_path: Model name for tokenizer initialization
        batch_size: Batch size for DataLoaders
        
    Returns:
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for evaluation
        metric: Evaluation metric for the dataset
        mismatched_eval_dataloader: Additional eval dataloader for MNLI (if applicable)
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load dataset and metric
    if dataset_name in ["mnli_matched", "mnli_mismatched"]:
        datasets = load_dataset("glue", "mnli")
        metric = evaluate.load("glue", "mnli")
    else:
        datasets = load_dataset("glue", dataset_name)
        metric = evaluate.load("glue", dataset_name)
    
    # Determine columns to remove based on dataset
    if dataset_name == "sst2":
        remove_columns = ["idx", "sentence"]
    elif dataset_name == "cola":
        remove_columns = ["idx", "sentence"]
    elif dataset_name == "qqp":
        remove_columns = ["idx", "question1", "question2"]
    elif dataset_name == "qnli":
        remove_columns = ["idx", "question", "sentence"]
    else:  # MNLI datasets
        remove_columns = ["idx", "premise", "hypothesis"]
    
    # Tokenize datasets
    tokenized_datasets = datasets.map(
        lambda examples: tokenize_function(examples, tokenizer=tokenizer, dataset_name=dataset_name),
        batched=True,
        remove_columns=remove_columns,
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # Create collate function for DataLoader
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    
    # Create training DataLoader
    train_dataset = tokenized_datasets["train"]
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Create evaluation DataLoader based on dataset choice
    if dataset_name == "mnli_matched":
        # For matched: primary = validation_matched, secondary = validation_mismatched
        eval_dataset = tokenized_datasets["validation_matched"]
        auxiliary_eval_dataset = tokenized_datasets["validation_mismatched"]
    elif dataset_name == "mnli_mismatched":
        # For mismatched: primary = validation_mismatched, secondary = validation_matched
        eval_dataset = tokenized_datasets["validation_mismatched"]
        auxiliary_eval_dataset = tokenized_datasets["validation_matched"]
    else:
        eval_dataset = tokenized_datasets["validation" if "validation" in tokenized_datasets else "test"]
        auxiliary_eval_dataset = None
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Handle MNLI auxiliary evaluation set
    auxiliary_eval_dataloader = None
    if dataset_name in ["mnli_matched", "mnli_mismatched"]:
        auxiliary_eval_dataloader = DataLoader(
            auxiliary_eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        return train_dataloader, eval_dataloader, auxiliary_eval_dataloader, metric
    
    return train_dataloader, eval_dataloader, metric


def load_federated_glue_dataset(dataset_name, num_clients, model_name_or_path="roberta-base", 
                                batch_size=32, alpha=0.5, seed=42, save_dir=None, 
                                client_validation_ratio=None):
    """
    Load GLUE dataset with non-IID federated splits and data distribution visualization.
    
    Args:
        dataset_name: Name of the GLUE dataset to load
        num_clients: Number of federated learning clients
        model_name_or_path: Model name for tokenizer initialization
        batch_size: Batch size for DataLoaders
        alpha: Dirichlet distribution parameter (lower = more non-IID)
        seed: Random seed for reproducibility
        save_dir: Directory to save distribution analysis and visualizations
        client_validation_ratio: Ratio of each client's data to use for validation (0.0-1.0)
                               If None, clients get only training data (original behavior)
        
    Returns:
        client_dataloaders: List of training DataLoaders for each client
        client_val_dataloaders: List of validation DataLoaders for each client (if client_validation_ratio is set)
        eval_dataloader: DataLoader for global evaluation
        metric: Evaluation metric for the dataset
        mismatched_eval_dataloader: Additional eval dataloader for MNLI (if applicable)
        distribution_analysis: Analysis of the data distribution
    """
    # Load base dataset
    dataset_result = load_glue_dataset(dataset_name, model_name_or_path, batch_size)
    
    # Unpack results
    if len(dataset_result) == 4:
        train_dataloader, eval_dataloader, auxiliary_eval_dataloader, metric = dataset_result
    else:
        train_dataloader, eval_dataloader, metric = dataset_result
        auxiliary_eval_dataloader = None
    
    # Get the underlying dataset for splitting
    train_dataset = train_dataloader.dataset
    
    # Create non-IID splits using Dirichlet distribution
    client_indices = dirichlet_non_iid_split(
        dataset=train_dataset,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed
    )
    
    # Analyze data distribution
    distribution_analysis = analyze_data_distribution(
        dataset=train_dataset,
        client_indices=client_indices,
        dataset_name=dataset_name
    )
    
    # Add split parameters to analysis
    distribution_analysis['alpha'] = alpha
    distribution_analysis['seed'] = seed
    distribution_analysis['num_clients'] = num_clients
    
    # Save analysis and create visualizations if save_dir is provided
    if save_dir is not None:
        from pathlib import Path
        save_dir = Path(save_dir)
        
        # Check if we're in an alpha directory structure
        parent_dir = save_dir.parent
        if parent_dir.name.startswith('alpha_'):
            # Use dataset-specific and client-count-specific data_distribution folder
            # This ensures different datasets and client counts have their own distribution plots
            alpha_dist_dir = parent_dir / f"data_distribution_{dataset_name}_c{num_clients}"
            
            if alpha_dist_dir.exists():
                # Distribution data already exists for this alpha value, dataset, and client count
                print(f"Using existing data distribution from: {alpha_dist_dir}")
                
                # Create a reference file in the experiment directory
                exp_dist_ref = save_dir / "data_distribution_location.txt"
                with open(exp_dist_ref, 'w') as f:
                    f.write(f"Data distribution plots are located at: {alpha_dist_dir.absolute()}\n")
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Alpha value: {alpha}\n")
                    f.write(f"Number of clients: {num_clients}\n")
                    f.write(f"Seed: {seed}\n")
            else:
                # First experiment in this alpha folder for this dataset and client count - generate distribution data
                alpha_dist_dir.mkdir(parents=True, exist_ok=True)
                
                # Save distribution analysis
                save_distribution_analysis(distribution_analysis, alpha_dist_dir / "distribution_analysis.json")
                
                # Create visualizations
                visualize_data_distribution(distribution_analysis, alpha_dist_dir, alpha, seed)
                
                # Create a reference file in the experiment directory
                exp_dist_ref = save_dir / "data_distribution_location.txt"
                with open(exp_dist_ref, 'w') as f:
                    f.write(f"Data distribution plots are located at: {alpha_dist_dir.absolute()}\n")
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write(f"Alpha value: {alpha}\n")
                    f.write(f"Number of clients: {num_clients}\n")
                    f.write(f"Seed: {seed}\n")
                
                print(f"Created new data distribution at: {alpha_dist_dir}")
        else:
            # Not in alpha directory structure (shouldn't happen with federated split, but handle it)
            dist_dir = save_dir / f"data_distribution_c{num_clients}"
            dist_dir.mkdir(parents=True, exist_ok=True)
            
            # Save distribution analysis
            save_distribution_analysis(distribution_analysis, dist_dir / "distribution_analysis.json")
            
            # Create visualizations
            visualize_data_distribution(distribution_analysis, dist_dir, alpha, seed)
    
    # Create client subsets
    client_subsets = create_client_subsets(train_dataset, client_indices)
    
    # Create DataLoaders for each client with optional train/validation splitting
    client_dataloaders = []
    client_val_dataloaders = []
    
    if client_validation_ratio is not None and client_validation_ratio > 0:
        # Split each client's data into train/validation
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        for i, subset in enumerate(client_subsets):
            # Get indices and labels for stratified splitting
            subset_indices = list(range(len(subset)))
            subset_labels = [subset[idx]['labels'].item() for idx in subset_indices]
            
            # Stratified split to maintain label distribution
            train_indices, val_indices = train_test_split(
                subset_indices,
                test_size=client_validation_ratio,
                stratify=subset_labels,
                random_state=seed + i  # Different seed per client for diversity
            )
            
            # Create train subset
            train_subset = torch.utils.data.Subset(subset, train_indices)
            train_dataloader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=train_dataloader.collate_fn
            )
            client_dataloaders.append(train_dataloader)
            
            # Create validation subset
            val_subset = torch.utils.data.Subset(subset, val_indices)
            val_dataloader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,  # No need to shuffle validation data
                collate_fn=train_dataloader.collate_fn
            )
            client_val_dataloaders.append(val_dataloader)
            
            print(f"Client {i}: {len(train_subset)} train, {len(val_subset)} validation samples")
    else:
        # Original behavior - no client-side validation splitting
        for subset in client_subsets:
            client_dataloader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=train_dataloader.collate_fn
            )
            client_dataloaders.append(client_dataloader)
    
    # Return results
    if auxiliary_eval_dataloader is not None:
        if client_validation_ratio is not None and client_validation_ratio > 0:
            return client_dataloaders, client_val_dataloaders, eval_dataloader, auxiliary_eval_dataloader, metric, distribution_analysis
        else:
            return client_dataloaders, eval_dataloader, auxiliary_eval_dataloader, metric, distribution_analysis
    else:
        if client_validation_ratio is not None and client_validation_ratio > 0:
            return client_dataloaders, client_val_dataloaders, eval_dataloader, metric, distribution_analysis
        else:
            return client_dataloaders, eval_dataloader, metric, distribution_analysis


def load_tokenizer(model_name_or_path):
    """
    Load and configure tokenizer for the specified model.
    
    Args:
        model_name_or_path: Model name or path for tokenizer
        
    Returns:
        tokenizer: Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Set padding token if not already set
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer 