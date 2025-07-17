"""
Refactored server implementation for federated learning.
Handles model aggregation and evaluation with minimal changes from original.
"""

import copy
import torch
from tqdm import tqdm


class Server:
    """
    Central server for federated learning.
    Manages global model aggregation and evaluation.
    """
    
    def __init__(self, global_model, device):
        """
        Initialize the server with a global model and device.
        
        Args:
            global_model: The initial model (typically with PEFT/LoRA configuration)
            device: The device to run computations on (CPU or GPU)
        """
        self.global_model = global_model
        self.device = device
        # Track round number for logging
        self.current_round = 0
    
    def aggregate(self, client_models):
        """
        Aggregate models using Federated Averaging (FedAvg).
        
        This is the core of federated learning - combining multiple locally trained
        models into a single global model by averaging their parameters.
        
        Args:
            client_models: List of model state dictionaries from clients
        """
        if not client_models:
            print("Warning: No client models to aggregate")
            return
        
        # Start with a copy of the first client's model parameters
        global_params = copy.deepcopy(client_models[0])
        
        # Average the parameters of all client models
        for key in global_params.keys():
            # Sum all client parameters for this key
            for i in range(1, len(client_models)):
                global_params[key] += client_models[i][key]
            # Average by dividing by number of clients
            global_params[key] = torch.div(global_params[key], len(client_models))
        
        # Update global model with averaged parameters
        self.global_model.load_state_dict(global_params)
        
        # Increment round counter
        self.current_round += 1
    
    def evaluate(self, eval_dataloader, metric):
        """
        Evaluate the global model on the validation dataset.
        
        Runs inference on the evaluation dataset and computes metrics
        (e.g., accuracy for GLUE tasks) to assess model performance.
        
        Args:
            eval_dataloader: DataLoader containing validation data
            metric: The evaluation metric object (from HuggingFace evaluate)
            
        Returns:
            eval_metric: Dictionary containing evaluation results
        """
        if eval_dataloader is None:
            print("Warning: No evaluation dataloader provided")
            return None
        
        # Move model to device and set to evaluation mode
        self.global_model.to(self.device)
        self.global_model.eval()
        
        # Create a fresh metric instance for each evaluation to avoid accumulation
        # This is necessary because Glue metric objects don't have a reset() method
        import evaluate
        dataset_name = metric.config_name if hasattr(metric, 'config_name') else 'sst2'
        fresh_metric = evaluate.load("glue", dataset_name)
        
        # Run evaluation
        eval_iter = tqdm(eval_dataloader, desc="Global Evaluation", leave=False, ncols=100, mininterval=1.0)
        
        with torch.no_grad():
            for step, batch in enumerate(eval_iter):
                # Move batch to device
                batch.to(self.device)
                
                # Forward pass
                outputs = self.global_model(**batch)
                
                # Get predictions
                predictions = outputs.logits.argmax(dim=-1)
                references = batch["labels"]
                
                # Add batch to fresh metric
                fresh_metric.add_batch(
                    predictions=predictions,
                    references=references,
                )
        
        # Compute final metric using fresh metric
        eval_metric = fresh_metric.compute()
        
        # Log evaluation results
        tqdm.write(f"Round {self.current_round} - Global Evaluation Result: {eval_metric}")
        
        return eval_metric
    
    def get_model_state(self):
        """
        Get the current global model state dictionary.
        
        Returns:
            State dictionary of the global model
        """
        return copy.deepcopy(self.global_model.state_dict())
    
    def set_model_state(self, state_dict):
        """
        Set the global model state dictionary.
        
        Args:
            state_dict: Model state dictionary to load
        """
        self.global_model.load_state_dict(state_dict)
        self.global_model.to(self.device)
    
    def reset_round_counter(self):
        """Reset the round counter (useful when starting new experiments)"""
        self.current_round = 0 