"""
Refactored main script for federated learning experiments.
Supports both FORCE and baseline methods with unified federated learning round logic.
"""

import os
import argparse
import torch
import copy
import json
import pickle
import uuid
import random
import hashlib
import logging
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer_utils import set_seed
from peft import LoraConfig, get_peft_model

from data_utils import load_glue_dataset, load_federated_glue_dataset
from client import ForceClient, BaselineClient
from server import Server


def initialize_model(model_name, num_labels, use_dora=False, lora_rank=4, lora_alpha=16, lora_dropout=0.1):
    """
    Initialize model with LoRA/DoRA configuration.
    
    Args:
        model_name: Pretrained model name
        num_labels: Number of output labels
        use_dora: Whether to use DoRA (for FORCE) or standard LoRA (for baseline)
        lora_rank: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        
    Returns:
        model: Model with LoRA/DoRA applied
    """
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        return_dict=True
    )
    
    # Configure LoRA/DoRA
    peft_config = LoraConfig(
        use_dora=use_dora,
        task_type="SEQ_CLS",
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    # Apply LoRA/DoRA to model
    model = get_peft_model(base_model, peft_config)
    
    return model


def create_experiment_directory(base_dir="experiments", args=None):
    """
    Create unique directory for saving experiment results with descriptive naming.
    Supports multi-process environments with conflict avoidance.
    
    Args:
        base_dir: Base directory for experiments
        args: Command line arguments for descriptive naming
        
    Returns:
        exp_dir: Path to unique experiment directory
    """
    # High-precision timestamp (millisecond precision)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    # Build descriptive directory name parts
    dir_parts = [timestamp]
    
    if args:
        # Add method name (replace + with _ for filename compatibility)
        method_name = args.method.replace("+", "_")
        dir_parts.append(method_name)
        
        # Add dataset name
        dir_parts.append(args.dataset)
        
        # Add MNLI training key for reuse identification
        if args.dataset in ["mnli_matched", "mnli_mismatched"]:
            training_key = get_mnli_training_key(args)
            dir_parts.append(training_key)
        
        # Add number of clients
        dir_parts.append(f"c{args.num_clients}")
        
        # Add number of rounds
        dir_parts.append(f"r{args.num_rounds}")
        
        # Add GPU ID if using GPU
        if hasattr(args, 'cuda_device') and args.cuda_device >= 0:
            dir_parts.append(f"gpu{args.cuda_device}")
        
        # Don't add alpha to dir name if using federated split (it will be in parent dir)
        # if hasattr(args, 'enable_federated_split') and args.enable_federated_split:
        #     alpha_str = f"alpha{args.alpha}".replace(".", "p")  # Replace . with p for filename
        #     dir_parts.append(f"fed_{alpha_str}")
    
    # Add short UUID for guaranteed uniqueness
    unique_id = str(uuid.uuid4())[:8]
    dir_parts.append(unique_id)
    
    # Combine directory name
    dir_name = "_".join(dir_parts)
    
    # Check if we need to create alpha parent directory for federated split experiments
    if args and hasattr(args, 'enable_federated_split') and args.enable_federated_split:
        # Create alpha parent directory
        alpha_str = f"alpha_{args.alpha}"
        parent_dir = Path(base_dir) / alpha_str
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment directory goes inside alpha directory
        exp_dir = parent_dir / dir_name
    else:
        # Normal directory structure for non-federated experiments
        exp_dir = Path(base_dir) / dir_name
    
    # Ensure directory uniqueness (additional safety against race conditions)
    counter = 1
    original_exp_dir = exp_dir
    while exp_dir.exists():
        exp_dir = Path(f"{original_exp_dir}_dup{counter}")
        counter += 1
    
    # Create directory (exist_ok=False ensures no overwrite)
    exp_dir.mkdir(parents=True, exist_ok=False)
    
    # Create process lock file for multi-process safety
    process_id = os.getpid()
    lock_file = exp_dir / f".lock_pid_{process_id}"
    
    # Write process information to lock file
    lock_info = {
        "pid": process_id,
        "start_time": datetime.now().isoformat(),
        "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
        "working_dir": os.getcwd(),
        "cuda_device": getattr(args, 'cuda_device', None) if args else None
    }
    
    with open(lock_file, 'w') as f:
        json.dump(lock_info, f, indent=2)
    
    # Set up temporary logger for directory creation
    setup_logger = logging.getLogger("Setup")
    setup_logger.info(f"Created experiment directory: {exp_dir}")
    
    return exp_dir


def setup_muon_environment(cuda_device, method):
    """
    Set up environment variables for Muon optimizer with multi-process conflict avoidance.
    
    Args:
        cuda_device: CUDA device ID
        method: Training method name
    """
    if "muon" in method:
        os.environ['MASTER_ADDR'] = 'localhost'
        
        # Enhanced port allocation to avoid conflicts
        base_port = 12355
        
        # Multi-layer port offset calculation:
        # 1. GPU-based offset (1000 per GPU)
        # 2. Process-based offset (hash of PID modulo 900)
        # 3. Random offset (0-99) for additional safety
        gpu_offset = cuda_device * 1000 if cuda_device >= 0 else 0
        pid_offset = hash(os.getpid()) % 900
        random_offset = random.randint(0, 99)
        
        final_port = base_port + gpu_offset + pid_offset + random_offset
        
        # Ensure port is within valid range
        final_port = max(12355, min(65535, final_port))
        
        os.environ['MASTER_PORT'] = str(final_port)
        
        print(f"Muon distributed setup:")
        print(f"  MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        print(f"  MASTER_PORT: {os.environ['MASTER_PORT']}")
        print(f"  Port calculation: base({base_port}) + gpu({gpu_offset}) + pid({pid_offset}) + random({random_offset}) = {final_port}")


def save_experiment_config(exp_dir, args):
    """
    Save comprehensive experiment configuration to JSON file.
    
    Args:
        exp_dir: Experiment directory
        args: Command line arguments
    """
    config = vars(args).copy()
    
    # Add system information
    config['system_info'] = {
        'python_version': f"{os.sys.version}",
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'hostname': os.uname().nodename if hasattr(os, 'uname') else "unknown",
        'working_directory': os.getcwd(),
        'process_id': os.getpid(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Add CUDA device info if available
    if torch.cuda.is_available() and hasattr(args, 'cuda_device') and args.cuda_device >= 0:
        try:
            device_props = torch.cuda.get_device_properties(args.cuda_device)
            config['system_info']['cuda_device_name'] = device_props.name
            config['system_info']['cuda_device_memory'] = device_props.total_memory
        except:
            config['system_info']['cuda_device_info'] = "unavailable"
    
    # Add environment variables for Muon if relevant
    if "muon" in getattr(args, 'method', ''):
        config['muon_env'] = {
            'MASTER_ADDR': os.environ.get('MASTER_ADDR'),
            'MASTER_PORT': os.environ.get('MASTER_PORT')
        }
    
    config_path = exp_dir / "config.json"
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Set up temporary logger for config saving
    setup_logger = logging.getLogger("Setup")
    setup_logger.info(f"Saved experiment configuration: {config_path}")


def save_experiment_results(exp_dir, results):
    """
    Save comprehensive experiment results to multiple formats.
    
    Args:
        exp_dir: Experiment directory
        results: Dictionary containing experiment results
    """
    # Add completion timestamp
    results['completion_info'] = {
        'end_time': datetime.now().isoformat(),
        'total_duration_seconds': results.get('experiment_duration_seconds', None),
        'process_id': os.getpid()
    }
    
    # Save as JSON for easy reading
    json_path = exp_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Saved experiment results: {json_path}")


def get_mnli_training_key(args):
    """
    Generate a unique key for MNLI training configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Unique key for this training configuration
    """
    # Create key based on training-relevant parameters (excluding dataset variant)
    key_params = {
        'method': args.method,
        'model_name': args.model_name,
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'enable_federated_split': args.enable_federated_split,
        'alpha': args.alpha if args.enable_federated_split else None,
        'seed': args.seed
    }
    
    # Create hash of the parameters
    key_str = json.dumps(key_params, sort_keys=True)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    return f"mnli_train_{key_hash}"


def find_mnli_checkpoint(args, base_experiments_dir="experiments"):
    """
    Find existing MNLI training checkpoint for the same configuration.
    
    Args:
        args: Command line arguments
        base_experiments_dir: Base directory to search for experiments
        
    Returns:
        Path or None: Path to the checkpoint directory if found
    """
    base_dir = Path(base_experiments_dir)
    if not base_dir.exists():
        return None
    
    training_key = get_mnli_training_key(args)
    
    # Search for existing experiments with the same training key
    for exp_dir in base_dir.iterdir():
        if exp_dir.is_dir() and training_key in exp_dir.name:
            checkpoint_file = exp_dir / "mnli_training_checkpoint.pkl"
            if checkpoint_file.exists():
                return exp_dir
    
    return None


def save_mnli_checkpoint(exp_dir, clients, server, round_results, args):
    """
    Save MNLI training checkpoint for future reuse.
    
    Args:
        exp_dir: Experiment directory
        clients: List of trained clients
        server: Trained server
        round_results: Training results
        args: Command line arguments
    """
    checkpoint_data = {
        'server_state': server.get_model_state(),
        'round_results': round_results,
        'training_config': {
            'method': args.method,
            'model_name': args.model_name,
            'num_clients': args.num_clients,
            'num_rounds': args.num_rounds,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'lora_rank': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'enable_federated_split': args.enable_federated_split,
            'alpha': args.alpha if args.enable_federated_split else None,
            'seed': args.seed
        },
        'save_timestamp': datetime.now().isoformat()
    }
    
    checkpoint_file = exp_dir / "mnli_training_checkpoint.pkl"
    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"MNLI training checkpoint saved: {checkpoint_file}")


def load_mnli_checkpoint(checkpoint_dir):
    """
    Load MNLI training checkpoint.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        
    Returns:
        dict: Checkpoint data
    """
    checkpoint_file = checkpoint_dir / "mnli_training_checkpoint.pkl"
    with open(checkpoint_file, "rb") as f:
        checkpoint_data = pickle.load(f)
    
    print(f"MNLI training checkpoint loaded from: {checkpoint_file}")
    return checkpoint_data


def evaluate_mnli_with_checkpoint(args, exp_dir, checkpoint_data):
    """
    Evaluate MNLI using existing checkpoint with different evaluation focus.
    
    Args:
        args: Command line arguments
        exp_dir: Experiment directory for new results
        checkpoint_data: Loaded checkpoint data
        
    Returns:
        dict: Evaluation results
    """
    print(f"Reusing MNLI training for {args.dataset} evaluation...")
    
    # Set up device
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() and args.cuda_device >= 0 else "cpu")
    
    # Load dataset for evaluation
    dataset_result = load_glue_dataset(
        dataset_name=args.dataset,
        model_name_or_path=args.model_name,
        batch_size=args.batch_size
    )
    
    if len(dataset_result) == 4:
        _, eval_dataloader, auxiliary_eval_dataloader, metric = dataset_result
    else:
        _, eval_dataloader, metric = dataset_result
        auxiliary_eval_dataloader = None
    
    # Initialize model with same configuration
    num_labels = 3  # MNLI always has 3 labels
    use_dora = args.method in ["soft_constraint", "Newton_shulz", "QR"] or "muon" in args.method
    
    model = initialize_model(
        model_name=args.model_name,
        num_labels=num_labels,
        use_dora=use_dora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Initialize server and load checkpoint
    server = Server(global_model=model, device=device)
    server.set_model_state(checkpoint_data['server_state'])
    
    # Evaluate on both validation sets
    print("Evaluating on validation sets...")
    eval_results = server.evaluate(eval_dataloader, metric=metric)
    primary_accuracy = eval_results.get("accuracy", 0.0)
    
    auxiliary_results = server.evaluate(auxiliary_eval_dataloader, metric=metric)
    auxiliary_accuracy = auxiliary_results.get("accuracy", 0.0)
    
    # Determine evaluation focus based on dataset choice
    is_mnli_mismatched = (args.dataset == "mnli_mismatched")
    primary_eval_name = "mismatched" if is_mnli_mismatched else "matched"
    auxiliary_eval_name = "matched" if is_mnli_mismatched else "mismatched"
    
    # Create results structure
    results = {
        "config": vars(args),
        "rounds": checkpoint_data['round_results']['rounds'],  # Reuse training rounds
        "best_accuracy": primary_accuracy,  # Final accuracy on chosen evaluation
        "best_round": len(checkpoint_data['round_results']['rounds']),  # Last round
        "final_accuracy": primary_accuracy,
        "best_auxiliary_accuracy": auxiliary_accuracy,
        "best_auxiliary_round": len(checkpoint_data['round_results']['rounds']),
        "final_auxiliary_accuracy": auxiliary_accuracy,
        "primary_eval_type": primary_eval_name,
        "data_distribution": checkpoint_data['round_results'].get('data_distribution'),
        "reused_checkpoint": True,
        "original_checkpoint_from": str(checkpoint_data.get('save_timestamp')),
        "training_config_reused": checkpoint_data['training_config']
    }
    
    # Save evaluation results
    save_experiment_config(exp_dir, args)
    save_experiment_results(exp_dir, results)
    
    # Print summary
    print("\n" + "="*50)
    print("MNLI CHECKPOINT REUSE SUMMARY")
    print("="*50)
    print(f"Dataset Focus: {args.dataset}")
    print(f"Primary Evaluation ({primary_eval_name.capitalize()}): {primary_accuracy:.4f}")
    print(f"Auxiliary Evaluation ({auxiliary_eval_name.capitalize()}): {auxiliary_accuracy:.4f}")
    print(f"Checkpoint Reused From: {checkpoint_data.get('save_timestamp')}")
    print(f"Results saved to: {exp_dir}")
    print("="*50)
    
    return results


def setup_experiment_logging(exp_dir):
    """
    Set up comprehensive logging for the experiment.
    
    Args:
        exp_dir: Experiment directory for log files
    """
    # Create logs directory
    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(log_dir / "experiment.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler for user-facing messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Create experiment-specific logger
    exp_logger = logging.getLogger("Experiment")
    exp_logger.info(f"Experiment logging initialized. Log directory: {log_dir}")
    
    return exp_logger


def run_federated_learning(args, exp_dir):
    """
    Main federated learning loop supporting both FORCE and baseline methods.
    
    Args:
        args: Command line arguments
        exp_dir: Directory to save results
        
    Returns:
        results: Dictionary containing training results
    """
    # Record experiment start time
    experiment_start_time = datetime.now()
    
    # Set up experiment logging
    exp_logger = setup_experiment_logging(exp_dir)
    exp_logger.info(f"Experiment started at: {experiment_start_time.isoformat()}")
    
    # Check for MNLI checkpoint reuse
    if args.dataset in ["mnli_matched", "mnli_mismatched"]:
        checkpoint_dir = find_mnli_checkpoint(args)
        if checkpoint_dir is not None:
            exp_logger.info(f"Found existing MNLI checkpoint: {checkpoint_dir}")
            checkpoint_data = load_mnli_checkpoint(checkpoint_dir)
            return evaluate_mnli_with_checkpoint(args, exp_dir, checkpoint_data)
        else:
            exp_logger.info("No existing MNLI checkpoint found, starting fresh training...")
    
    # Use separate seeds for data distribution and model initialization
    # This allows fair comparison: same data split, different initializations
    data_seed = args.data_seed if args.data_seed is not None else args.seed
    model_seed = args.model_seed if args.model_seed is not None else args.seed
    
    exp_logger.info(f"Seeds - Data distribution: {data_seed}, Model initialization: {model_seed}")
    
    # Set random seed for model initialization and other operations
    set_seed(model_seed)
    
    # Set up Muon environment if needed
    setup_muon_environment(args.cuda_device, args.method)
    
    # Set up device
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() and args.cuda_device >= 0 else "cpu")
    exp_logger.info(f"Using device: {device}")
    
    # Determine number of labels based on dataset
    num_labels = 3 if "mnli" in args.dataset else 2
    
    # Load dataset
    exp_logger.info(f"Loading {args.dataset} dataset...")
    
    if args.enable_federated_split:
        exp_logger.info(f"Using federated non-IID data splitting with alpha={args.alpha}")
        
        # Determine if client evaluation is enabled
        client_val_ratio = args.client_validation_ratio if args.enable_client_evaluation else None
        if args.enable_client_evaluation:
            exp_logger.info(f"Client evaluation enabled with validation ratio: {client_val_ratio}")
        
        dataset_result = load_federated_glue_dataset(
            dataset_name=args.dataset,
            num_clients=args.num_clients,
            model_name_or_path=args.model_name,
            batch_size=args.batch_size,
            alpha=args.alpha,
            seed=data_seed,  # Use data_seed for data distribution
            save_dir=exp_dir,
            client_validation_ratio=client_val_ratio
        )
        
        # Unpack federated dataset results based on client evaluation
        if args.enable_client_evaluation:
            # With client evaluation: expect client_val_dataloaders in results
            if len(dataset_result) == 6:
                client_data_splits, client_val_dataloaders, eval_dataloader, auxiliary_eval_dataloader, metric, distribution_analysis = dataset_result
            else:
                client_data_splits, client_val_dataloaders, eval_dataloader, metric, distribution_analysis = dataset_result
                auxiliary_eval_dataloader = None
        else:
            # Without client evaluation: original behavior
            if len(dataset_result) == 5:
                client_data_splits, eval_dataloader, auxiliary_eval_dataloader, metric, distribution_analysis = dataset_result
            else:
                client_data_splits, eval_dataloader, metric, distribution_analysis = dataset_result
                auxiliary_eval_dataloader = None
            client_val_dataloaders = None
            
        exp_logger.info(f"Data distribution analysis:")
        exp_logger.info(f"  - Average entropy: {distribution_analysis['avg_entropy']:.3f} ± {distribution_analysis['std_entropy']:.3f}")
        exp_logger.info(f"  - Average EMD: {distribution_analysis['avg_emd']:.3f} ± {distribution_analysis['std_emd']:.3f}")
        exp_logger.info(f"  - Client sample sizes: {distribution_analysis['client_sizes']}")
        
    else:
        exp_logger.info("Using simple data loading (all clients share same data)")
        dataset_result = load_glue_dataset(
            dataset_name=args.dataset,
            model_name_or_path=args.model_name,
            batch_size=args.batch_size
        )
        
        # Unpack simple dataset results
        if len(dataset_result) == 4:
            train_dataloader, eval_dataloader, auxiliary_eval_dataloader, metric = dataset_result
        else:
            train_dataloader, eval_dataloader, metric = dataset_result
            auxiliary_eval_dataloader = None
        
        # All clients use the same training data
        client_data_splits = [train_dataloader for _ in range(args.num_clients)]
        distribution_analysis = None
    
    # Determine if using FORCE or baseline method
    is_force_method = args.method in ["soft_constraint", "Newton_shulz", "QR", "muon", 
                                      "soft_constraint+muon", "Newton_shulz+muon", "QR+muon"]
    
    # Initialize model with appropriate configuration
    exp_logger.info(f"Initializing model with {'DoRA' if is_force_method else 'LoRA'}...")
    model = initialize_model(
        model_name=args.model_name,
        num_labels=num_labels,
        use_dora=is_force_method,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Initialize server
    server = Server(global_model=model, device=device)
    
    # Initialize clients based on method type
    exp_logger.info(f"Initializing {args.num_clients} clients...")
    clients = []
    
    if is_force_method:
        # Create FORCE clients
        for i in range(args.num_clients):
            val_data = client_val_dataloaders[i] if client_val_dataloaders else None
            client = ForceClient(
                client_id=i,
                model=copy.deepcopy(model),
                data=client_data_splits[i],
                device=device,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                lambda_ortho=args.lambda_ortho,
                val_data=val_data
            )
            clients.append(client)
    else:
        # Create baseline clients
        baseline_method = "ffa_lora" if args.method == "ffa_lora" else "lora"
        for i in range(args.num_clients):
            val_data = client_val_dataloaders[i] if client_val_dataloaders else None
            client = BaselineClient(
                client_id=i,
                model=copy.deepcopy(model),
                data=client_data_splits[i],
                device=device,
                baseline_method=baseline_method,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                val_data=val_data
            )
            clients.append(client)
    
    # Determine primary evaluation strategy based on dataset
    is_mnli_mismatched = (args.dataset == "mnli_mismatched")
    primary_eval_name = "mismatched" if is_mnli_mismatched else "matched"
    auxiliary_eval_name = "matched" if is_mnli_mismatched else "mismatched"
    
    # Initialize results tracking
    results = {
        "config": vars(args),
        "rounds": [],
        "best_accuracy": 0.0,
        "best_round": 0,
        "final_accuracy": 0.0,
        "data_distribution": distribution_analysis,
        "primary_eval_type": primary_eval_name if auxiliary_eval_dataloader is not None else "standard"
    }
    
    if auxiliary_eval_dataloader is not None:
        results["best_auxiliary_accuracy"] = 0.0
        results["best_auxiliary_round"] = 0
        results["final_auxiliary_accuracy"] = 0.0
    
    # Federated learning rounds
    exp_logger.info(f"\nStarting federated learning for {args.num_rounds} rounds...")
    rounds_pbar = tqdm(range(args.num_rounds), desc="FL Rounds", leave=True, ncols=100)
    
    for round_num in rounds_pbar:
        round_results = {
            "round": round_num + 1,
            "client_losses": []
        }
        
        # Client evaluation tracking
        if args.enable_client_evaluation:
            round_results["client_evaluations"] = {
                "after_local_training": [],
                "after_aggregation": []
            }
        
        # Train each client
        client_models = []
        client_data_sizes = []
        
        for client in clients:
            # Train client
            if is_force_method:
                client.train(
                    epochs=args.num_epochs,
                    train_method=args.method,
                    gradient_accumulation_steps=args.gradient_accumulation_steps
                )
            else:
                # Baseline training
                client.train(
                    epochs=args.num_epochs,
                    gradient_accumulation_steps=args.gradient_accumulation_steps
                )
            
            # Evaluate client AFTER local training (before aggregation)
            if args.enable_client_evaluation:
                client_eval_results = client.evaluate(metric=metric)
                if client_eval_results:
                    client_eval_results["client_id"] = client.client_id
                    round_results["client_evaluations"]["after_local_training"].append(client_eval_results)
            
            # Collect client model and data size
            client_models.append(client.get_parameters())
            client_data_sizes.append(len(client.data))
        
        # Server aggregation with proper FedAvg weighting
        server.aggregate(client_models, client_data_sizes)
        
        # Evaluate global model
        eval_results = server.evaluate(eval_dataloader, metric=metric)
        if eval_results:
            # Store all metrics returned by the evaluator
            for metric_name, metric_value in eval_results.items():
                round_results[metric_name] = metric_value
            # Ensure backward compatibility with accuracy
            if "accuracy" not in eval_results and "matthews_correlation" in eval_results:
                round_results["accuracy"] = eval_results["matthews_correlation"]
        else:
            round_results["accuracy"] = 0.0
        
        # Evaluate on auxiliary set if MNLI
        if auxiliary_eval_dataloader is not None:
            auxiliary_results = server.evaluate(auxiliary_eval_dataloader, metric=metric)
            if auxiliary_results:
                # Store auxiliary metrics with prefix
                for metric_name, metric_value in auxiliary_results.items():
                    round_results[f"auxiliary_{metric_name}"] = metric_value
                # Ensure backward compatibility
                round_results["auxiliary_accuracy"] = auxiliary_results.get("accuracy", auxiliary_results.get("matthews_correlation", 0.0))
            else:
                round_results["auxiliary_accuracy"] = 0.0
            # Also store with specific names for clarity
            round_results[f"{auxiliary_eval_name}_accuracy"] = round_results["auxiliary_accuracy"]
            round_results[f"{primary_eval_name}_accuracy"] = round_results.get("accuracy", 0.0)
        
        # Update best results based on primary evaluation metric
        if round_results["accuracy"] > results["best_accuracy"]:
            results["best_accuracy"] = round_results["accuracy"]
            results["best_round"] = round_num + 1
        
        if auxiliary_eval_dataloader is not None and "auxiliary_accuracy" in round_results:
            if round_results["auxiliary_accuracy"] > results["best_auxiliary_accuracy"]:
                results["best_auxiliary_accuracy"] = round_results["auxiliary_accuracy"]
                results["best_auxiliary_round"] = round_num + 1
        
        # Update progress bar
        postfix = {"acc": f"{round_results['accuracy']:.4f}", "best": f"{results['best_accuracy']:.4f}"}
        if auxiliary_eval_dataloader is not None and "auxiliary_accuracy" in round_results:
            aux_label = f"{auxiliary_eval_name[:3]}_acc"  # "mat_acc" or "mis_acc"
            postfix[aux_label] = f"{round_results['auxiliary_accuracy']:.4f}"
        rounds_pbar.set_postfix(postfix)
        
        # Save round results
        results["rounds"].append(round_results)
        
        # Update clients with new global model
        global_state = server.get_model_state()
        for client in clients:
            client.set_parameters(global_state)
            
            # Evaluate client AFTER aggregation (with new global model)
            if args.enable_client_evaluation:
                client_eval_results = client.evaluate(metric=metric)
                if client_eval_results:
                    client_eval_results["client_id"] = client.client_id
                    round_results["client_evaluations"]["after_aggregation"].append(client_eval_results)
        
        # Add client evaluation statistics if enabled
        if args.enable_client_evaluation:
            # Calculate statistics for after_local_training
            pre_agg_accs = [eval_res.get("accuracy", 0.0) for eval_res in round_results["client_evaluations"]["after_local_training"]]
            post_agg_accs = [eval_res.get("accuracy", 0.0) for eval_res in round_results["client_evaluations"]["after_aggregation"]]
            
            if pre_agg_accs and post_agg_accs:
                round_results["client_stats"] = {
                    "pre_aggregation": {
                        "mean_accuracy": sum(pre_agg_accs) / len(pre_agg_accs),
                        "std_accuracy": (sum([(acc - sum(pre_agg_accs)/len(pre_agg_accs))**2 for acc in pre_agg_accs]) / len(pre_agg_accs))**0.5 if len(pre_agg_accs) > 1 else 0.0,
                        "min_accuracy": min(pre_agg_accs),
                        "max_accuracy": max(pre_agg_accs)
                    },
                    "post_aggregation": {
                        "mean_accuracy": sum(post_agg_accs) / len(post_agg_accs),
                        "std_accuracy": (sum([(acc - sum(post_agg_accs)/len(post_agg_accs))**2 for acc in post_agg_accs]) / len(post_agg_accs))**0.5 if len(post_agg_accs) > 1 else 0.0,
                        "min_accuracy": min(post_agg_accs),
                        "max_accuracy": max(post_agg_accs)
                    }
                }
    
    # Save final results
    results["final_accuracy"] = results["rounds"][-1]["accuracy"] if results["rounds"] else 0.0
    if auxiliary_eval_dataloader is not None:
        results["final_auxiliary_accuracy"] = results["rounds"][-1].get("auxiliary_accuracy", 0.0) if results["rounds"] else 0.0
    
    # Save MNLI checkpoint for future reuse
    if args.dataset in ["mnli_matched", "mnli_mismatched"]:
        save_mnli_checkpoint(exp_dir, clients, server, results, args)
    
    # Calculate and log experiment duration
    experiment_end_time = datetime.now()
    experiment_duration = experiment_end_time - experiment_start_time
    
    # Add duration to results
    results['experiment_start_time'] = experiment_start_time.isoformat()
    results['experiment_duration_seconds'] = experiment_duration.total_seconds()
    
    # Save experiment results
    save_experiment_results(exp_dir, results)
    
    # Print summary
    exp_logger.info("\n" + "="*50)
    exp_logger.info("EXPERIMENT SUMMARY")
    exp_logger.info("="*50)
    exp_logger.info(f"Method: {args.method}")
    exp_logger.info(f"Dataset: {args.dataset}")
    exp_logger.info(f"Model: {args.model_name}")
    exp_logger.info(f"Best {primary_eval_name.capitalize()} Accuracy: {results['best_accuracy']:.4f} (Round {results['best_round']})")
    exp_logger.info(f"Final {primary_eval_name.capitalize()} Accuracy: {results['final_accuracy']:.4f}")
    
    if auxiliary_eval_dataloader is not None:
        exp_logger.info(f"Best {auxiliary_eval_name.capitalize()} Accuracy: {results['best_auxiliary_accuracy']:.4f} (Round {results['best_auxiliary_round']})")
        exp_logger.info(f"Final {auxiliary_eval_name.capitalize()} Accuracy: {results['final_auxiliary_accuracy']:.4f}")
    
    # Log duration information
    hours = int(experiment_duration.total_seconds() // 3600)
    minutes = int((experiment_duration.total_seconds() % 3600) // 60)
    seconds = int(experiment_duration.total_seconds() % 60)
    exp_logger.info(f"Experiment duration: {hours}h {minutes}m {seconds}s")
    
    exp_logger.info(f"Results saved to: {exp_dir}")
    exp_logger.info("="*50)
    
    return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Refactored Federated Learning with FORCE and Baseline Methods")
    
    # Method selection
    parser.add_argument("--method", type=str, required=True,
                        choices=["soft_constraint", "Newton_shulz", "QR", "muon",
                                "soft_constraint+muon", "Newton_shulz+muon", "QR+muon",
                                "lora", "ffa_lora"],
                        help="Training method to use")
    
    # Dataset and model
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["sst2", "cola", "qqp", "qnli", "mnli_matched", "mnli_mismatched"],
                        help="GLUE dataset to use")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Pretrained model name from HuggingFace")
    
    # Federated learning parameters
    parser.add_argument("--num_clients", type=int, default=3,
                        help="Number of federated learning clients")
    parser.add_argument("--num_rounds", type=int, default=5,
                        help="Number of federated learning rounds")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of local training epochs per round")
    
    # Model hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate for optimizers")
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")
    
    # Training optimization parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    
    # FORCE-specific parameters
    parser.add_argument("--lambda_ortho", type=float, default=0.1,
                        help="Weight for orthogonality regularization loss in soft_constraint method")
    
    # Data distribution parameters
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet distribution parameter for non-IID split (lower = more non-IID)")
    parser.add_argument("--enable_federated_split", action="store_true",
                        help="Enable federated non-IID data splitting with visualization")
    
    # Client evaluation parameters
    parser.add_argument("--enable_client_evaluation", action="store_true",
                        help="Enable client-side evaluation on local validation data")
    parser.add_argument("--client_validation_ratio", type=float, default=0.2,
                        help="Ratio of each client's data to use for validation (0.0-1.0)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--data_seed", type=int, default=None,
                        help="Separate seed for data distribution (if None, uses --seed)")
    parser.add_argument("--model_seed", type=int, default=None,
                        help="Separate seed for model initialization (if None, uses --seed)")
    parser.add_argument("--cuda_device", type=int, default=0,
                        help="CUDA device to use (-1 for CPU)")
    parser.add_argument("--exp_dir", type=str, default="experiments",
                        help="Base directory for saving experiment results")
    
    args = parser.parse_args()
    
    # Create experiment directory
    exp_dir = create_experiment_directory(args.exp_dir, args)
    
    # Save configuration
    save_experiment_config(exp_dir, args)
    
    # Set up basic logging for main function
    main_logger = logging.getLogger("Main")
    main_logger.info("Starting experiment...")
    main_logger.info(f"Experiment directory: {exp_dir}")
    
    try:
        # Run federated learning (this will set up detailed experiment logging)
        results = run_federated_learning(args, exp_dir)
        main_logger.info("Experiment completed successfully!")
        
    except Exception as e:
        main_logger.error(f"Experiment failed with error: {str(e)}")
        # Log the full traceback to file
        exp_logger = logging.getLogger("Experiment")
        exp_logger.exception("Full traceback:")
        raise
    
    return results


if __name__ == "__main__":
    main() 