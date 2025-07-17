"""
Refactored client implementation for federated learning experiments.
Contains both FORCE method client and baseline client implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import copy
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
try:
    from muon import Muon, zeropower_via_newtonschulz5
    MUON_AVAILABLE = True
except ImportError:
    from simple_muon import SimpleMuon, simple_zeropower_via_newtonschulz5
    MUON_AVAILABLE = False
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # File handler will be added per experiment in main.py if needed
    ]
)


class ForceClient:
    """
    FORCE method client implementation specifically designed for DoRA (Weight-Decomposed Low-Rank Adaptation).
    
    Supports the following FORCE methods:
    - Soft constraint: Orthogonality regularization during training (Algorithm 1)
    - Newton-Schulz orthogonalization: Post-training orthogonal repair (Algorithm 2)
    - QR decomposition: Post-training orthogonal repair (Algorithm 2)
    - Muon optimizer: Matrix parameter optimization
    
    This implementation requires DoRA models (use_dora=True) and will raise errors
    if DoRA magnitude vectors are not found.
    """
    
    def __init__(self, client_id, model, data, device, learning_rate=3e-4, num_epochs=2, lambda_ortho=0.1):
        """
        Initialize FORCE client with DoRA model, data, and training configuration.
        
        Args:
            client_id: Client identifier
            model: DoRA model (must have magnitude vectors)
            data: Local dataset for training
            device: Device for computation (CPU/GPU)
            learning_rate: Learning rate for optimizers
            num_epochs: Number of training epochs per round
            lambda_ortho: Weight for orthogonality regularization loss
            
        Note:
            The model must be configured with DoRA (use_dora=True) as FORCE methods
            specifically require magnitude vectors for proper orthogonality constraints.
        """
        self.client_id = client_id
        self.model = model
        self.device = device
        self.data = data
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.lambda_ortho = lambda_ortho  # Orthogonality regularization weight
        
        # Set up client-specific logger
        self.logger = logging.getLogger(f"ForceClient_{client_id}")
        
        # Memory optimization settings
        self.gradient_accumulation_steps = 1
        
        # Track if optimizers need to be initialized for specific method
        self.train_method = None
        
        # Initialize optimizers
        self._initialize_optimizers()
        
        # Initialize learning rate schedulers
        self._initialize_lr_schedulers()
    
    def _initialize_optimizers(self):
        """Initialize optimizers based on parameter dimensions"""
        # Separate parameters for different optimizers
        matrix_params = []
        vector_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.ndim >= 2:
                    matrix_params.append(param)  # Matrix parameters
                else:
                    vector_params.append(param)  # Vector parameters
        
        self.optimizers = []
        self.use_muon = False  # Track if we're using muon
        
        # Add AdamW optimizer for all parameters (default for all methods)
        if matrix_params or vector_params:
            all_params = matrix_params + vector_params
            self.adamw_optimizer = AdamW(all_params, lr=self.learning_rate)
            self.optimizers.append(self.adamw_optimizer)
        
        if not self.optimizers:
            raise ValueError("No trainable parameters found for any optimizer!")
    
    def _initialize_optimizers_for_method(self, train_method):
        """Initialize optimizers specifically for the training method"""
        # Reset optimizers
        self.optimizers = []
        self.use_muon = False
        
        # Separate parameters for different optimizers
        matrix_params = []
        vector_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.ndim >= 2:
                    matrix_params.append(param)  # Matrix parameters
                else:
                    vector_params.append(param)  # Vector parameters
        
        # Only use Muon optimizer if the method explicitly contains "muon"
        if "muon" in train_method and matrix_params:
            try:
                # Try to use Muon if available and distributed training is available
                if MUON_AVAILABLE and torch.distributed.is_initialized():
                    self.logger.info(f"Using original MUON optimizer (distributed training initialized)")
                    self.muon_optimizer = Muon(matrix_params, lr=self.learning_rate, momentum=0.95)
                    self.optimizers.append(self.muon_optimizer)
                    self.use_muon = True
                else:
                    # Use simplified Muon for single-process environments
                    dist_status = "available" if torch.distributed.is_available() else "not available"
                    dist_init_status = "initialized" if torch.distributed.is_initialized() else "not initialized"
                    self.logger.info(f"Using SimpleMuon optimizer for single-process environment")
                    self.logger.info(f"  - Distributed training: {dist_status}, {dist_init_status}")
                    self.logger.info(f"  - Original MUON available: {MUON_AVAILABLE}")
                    self.simple_muon_optimizer = SimpleMuon(matrix_params, lr=self.learning_rate, momentum=0.95)
                    self.optimizers.append(self.simple_muon_optimizer)
                    self.use_muon = True
            except Exception as e:
                # Fallback to AdamW if Muon fails
                self.adamw_matrix_optimizer = AdamW(matrix_params, lr=self.learning_rate)
                self.optimizers.append(self.adamw_matrix_optimizer)
        
        # Add AdamW optimizer for vector parameters (if not using muon) or as fallback
        if vector_params and not self.use_muon:
            self.adamw_vector_optimizer = AdamW(vector_params, lr=self.learning_rate)
            self.optimizers.append(self.adamw_vector_optimizer)
        
        # If no optimizers were added, use AdamW for all parameters
        if not self.optimizers:
            all_params = matrix_params + vector_params
            self.adamw_optimizer = AdamW(all_params, lr=self.learning_rate)
            self.optimizers.append(self.adamw_optimizer)
        
        if not self.optimizers:
            raise ValueError("No trainable parameters found for any optimizer!")
    
    def _initialize_lr_schedulers(self):
        """Initialize learning rate schedulers for all optimizers"""
        num_training_steps = len(self.data) * self.num_epochs
        
        self.lr_schedulers = []
        for optimizer in self.optimizers:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps
            )
            self.lr_schedulers.append(lr_scheduler)
    
    def train(self, epochs, train_method, gradient_accumulation_steps=1):
        """
        Main training method supporting:
        - soft_constraint: Orthogonality regularization during training
        - Newton_shulz: Newton-Schulz orthogonalization after each epoch
        - QR: QR decomposition for orthogonalization
        - muon: Using Muon optimizer for matrix parameters
        """
        # Only initialize optimizers if the training method has changed
        # This prevents resetting optimizer state between federated learning rounds
        if self.train_method != train_method:
            # Initialize optimizers based on the specific training method
            self._initialize_optimizers_for_method(train_method)
            # Re-initialize learning rate schedulers for the new optimizers
            self._initialize_lr_schedulers()
            # Update the current training method
            self.train_method = train_method
            self.logger.info(f"Initialized optimizers for method: {train_method}")
        
        # Set memory optimization parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Move model to device
        self.model.to(self.device)
        self.model.train()
        
        total_loss = 0
        epoch_pbar = tqdm(range(epochs), desc=f"Client {self.client_id} Epochs", leave=True, ncols=100, mininterval=1.0)
        
        for epoch in epoch_pbar:
            epoch_loss = 0
            batch_iter = tqdm(enumerate(self.data), desc=f"  Training Batch", total=len(self.data), leave=False, ncols=100, mininterval=1.0)
            
            for step, batch in batch_iter:
                # Move batch tensors to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                else:
                    batch.to(self.device)
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Add orthogonality regularization for soft_constraint method
                if "soft_constraint" in train_method:
                    orth_loss = self._compute_orthogonal_loss()
                    loss = loss + self.lambda_ortho * orth_loss
                    
                    # Log orthogonality loss periodically for debugging
                    if step % max(1, len(self.data) // 4) == 0:  # Log 4 times per epoch
                        orth_loss_value = orth_loss.item() if torch.is_tensor(orth_loss) else float(orth_loss)
                        tqdm.write(f"  Client {self.client_id}, Step {step}: Task Loss = {outputs.loss.item():.6f}, "
                                  f"Ortho Loss = {orth_loss_value:.6f}, λ = {self.lambda_ortho}")
                
                # Scale loss for gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                
                # Backpropagation
                loss.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.data):
                    for opt in self.optimizers:
                        opt.step()
                        opt.zero_grad()
                    for lr_scheduler in self.lr_schedulers:
                        lr_scheduler.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                current_lr = self.lr_schedulers[0].get_last_lr()[0] if self.lr_schedulers else self.learning_rate
                batch_iter.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/(step+1):.4f}',
                    'lr': f'{current_lr:.6f}'
                })
            
            total_loss += epoch_loss / len(self.data)
            
            # Apply orthogonal repair after each epoch for Newton_shulz or QR methods
            if "Newton_shulz" in train_method:
                self.logger.debug(f"Applying Newton-Schulz orthogonal repair")
                self._orthogonal_repair_newton_schulz()
            elif "QR" in train_method:
                self.logger.debug(f"Applying QR orthogonal repair")
                self._orthogonal_repair_qr()
        
        # Log final training stats
        final_avg_loss = total_loss / epochs if epochs > 0 else 0
        self.logger.info(f"Training completed - Method: {train_method}, Average loss: {final_avg_loss:.4f}")
        
        # Clean up distributed training if used
        if "muon" in train_method and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    
    def _compute_orthogonal_loss(self, lora_alpha=16):
        """
        Calculate orthogonality regularization loss for DoRA FORCE method.
        Implements Algorithm 1, eq.(16): L_ortho = sum ‖ (W⁰ + B·A)^T (W⁰ + B·A) - I ‖_F^2
        
        IMPORTANT: In DoRA, orthogonality constraints are applied to the DIRECTION component (W⁰ + B·A),
        NOT to the magnitude-scaled component (mi * (W⁰ + B·A)). This ensures that the directional
        part maintains orthogonal properties while magnitude vectors handle scaling independently.
        
        Args:
            lora_alpha: LoRA alpha parameter (not used in DoRA but kept for compatibility)
            
        Returns:
            ortho_loss: Orthogonality regularization loss with gradients enabled
            
        Raises:
            RuntimeError: If DoRA magnitude parameters are not found
        """
        ortho_loss = 0.0
        
        # Get all parameters as a dictionary for easy access
        params = dict(self.model.named_parameters())
        
        for name, B in params.items():
            # Find LoRA B matrices
            if 'lora_B' in name:
                # Get related parameters
                if '.default.weight' in name:
                    # Remove both lora_B and .default.weight to get the base layer path
                    base_layer_path = name.replace('lora_B.default.weight', '')
                    
                    # Construct correct parameter names
                    A_name = base_layer_path + 'lora_A.default.weight'
                    weight_name = base_layer_path + 'base_layer.weight'
                    magnitude_name = base_layer_path + 'lora_magnitude_vector.default.weight'
                else:
                    # Fallback to old logic if naming convention is different
                    base_name = name.replace('lora_B', '')
                    A_name = base_name + 'lora_A'
                    weight_name = base_name + 'base_layer.weight'
                    magnitude_name = base_name + 'lora_magnitude_vector'
                
                A = params.get(A_name)
                W0 = params.get(weight_name)
                magnitude = params.get(magnitude_name)
                
                # If we can't find A, skip this layer
                if A is None:
                    continue
                    
                # If no pretrained weight found, use zeros (backward compatibility)
                if W0 is None:
                    W0 = torch.zeros_like(B @ A)
                
                # DoRA requires magnitude vector - raise error if not found
                if magnitude is None:
                    raise RuntimeError(f"FORCE method requires DoRA magnitude vector for layer {name}, "
                                     f"but none found. Please ensure the model uses DoRA (use_dora=True).")
                
                # CORRECTED: Apply orthogonality constraint to DIRECTION component only
                # DoRA direction component: W⁰ + B·A (before magnitude scaling)
                direction_component = W0 + B @ A  # This is the directional part
                
                # Compute Gram matrix for DIRECTION component: (W⁰ + B·A)^T @ (W⁰ + B·A)
                # This is the correct implementation according to Algorithm 1, eq.(16)
                gram_matrix = direction_component.transpose(-2, -1) @ direction_component
                
                # Create identity matrix of appropriate size
                identity = torch.eye(
                    gram_matrix.size(-1), 
                    device=gram_matrix.device, 
                    dtype=gram_matrix.dtype
                )
                
                # Compute Frobenius norm squared: ‖G - I‖_F^2
                # This is the exact formula from paper eq.(16) applied to direction component
                ortho_term = torch.norm(gram_matrix - identity, p='fro') ** 2
                ortho_loss = ortho_loss + ortho_term
        
        return ortho_loss
    
    def _orthogonal_repair_newton_schulz(self, lora_alpha=16, lr_AB=0.01, ns_steps=5):
        """
        Repair orthogonality using Newton-Schulz algorithm for DoRA.
        Implements Algorithm 2 with DoRA-specific processing:
        1. Compute Mi = W⁰ + Bi*Ai (line 12)
        2. Compute Qi via Newton-Schulz (line 15) 
        3. Compute Δi = Qi - W⁰ (line 16)
        4. Solve argmin ||Δi - Bi*Ai||_F^2 (line 17)
        
        This implementation is specifically designed for DoRA and requires magnitude vectors.
        
        Raises:
            RuntimeError: If DoRA magnitude parameters are not found
        """
        with torch.no_grad():
            params = dict(self.model.named_parameters())
            
            for name, B in params.items():
                if 'lora_B' in name:
                    # Get related parameters
                    # Fix: Handle the .default.weight suffix correctly
                    if '.default.weight' in name:
                        # Remove both lora_B and .default.weight to get the base layer path
                        base_layer_path = name.replace('lora_B.default.weight', '')
                        
                        # Construct correct parameter names
                        A_name = base_layer_path + 'lora_A.default.weight'
                        weight_name = base_layer_path + 'base_layer.weight'
                        magnitude_name = base_layer_path + 'lora_magnitude_vector.default.weight'
                    else:
                        # Fallback to old logic if naming convention is different
                        base_name = name.replace('lora_B', '')
                        A_name = base_name + 'lora_A'
                        weight_name = base_name + 'base_layer.weight'
                        magnitude_name = base_name + 'lora_magnitude_vector'
                    
                    A = params.get(A_name)
                    W0 = params.get(weight_name)
                    magnitude = params.get(magnitude_name)
                    
                    if A is None or W0 is None:
                        continue
                    
                    # DoRA requires magnitude vector
                    if magnitude is None:
                        raise RuntimeError(f"FORCE Newton-Schulz requires DoRA magnitude vector for layer {name}, "
                                         f"but none found. Please ensure the model uses DoRA (use_dora=True).")
                    
                    # Step 1: Compute Mi = W⁰ + Bi*Ai (Algorithm 2, line 12)
                    Mi = W0 + B @ A
                    
                    # Step 2: Orthogonalize using Newton-Schulz algorithm (Algorithm 2, line 15)
                    if MUON_AVAILABLE:
                        Qi = zeropower_via_newtonschulz5(Mi, steps=ns_steps)
                    else:
                        Qi = simple_zeropower_via_newtonschulz5(Mi, steps=ns_steps)
                    
                    # Step 3: Compute target update: Δi = Qi - W⁰ (Algorithm 2, line 16)
                    # This is the key correction - subtract W0, not the current combined weight
                    delta_i = Qi - W0
                    
                    # Step 4: Solve optimization problem (Algorithm 2, line 17)
                    # argmin ||Δi - Bi*Ai||_F^2
                    # For DoRA, we need to consider the magnitude scaling in the optimization
                    
                    # Current B*A product
                    current_BA = B @ A
                    
                    # For DoRA, the relationship is more complex due to magnitude scaling
                    # We approximate the optimization by minimizing ||target_BA - B*A||_F^2
                    # where target_BA is derived from the orthogonalization target
                    target_BA = delta_i  # Simplified approximation for DoRA
                    
                    # Calculate residual for gradient computation
                    residual = current_BA - target_BA
                    
                    # Update B and A to minimize ||residual||_F^2
                    B_grad = residual @ A.T
                    A_grad = B.T @ residual
                    
                    # Apply gradient descent updates
                    B.add_(-lr_AB * B_grad)
                    A.add_(-lr_AB * A_grad)
    
    def _orthogonal_repair_qr(self, lora_alpha=16, lr_AB=0.01):
        """
        Repair orthogonality using QR decomposition for DoRA.
        Implements Algorithm 2 with DoRA-specific processing:
        1. Compute Mi = W⁰ + Bi*Ai (line 12)
        2. Compute Qi via QR decomposition (line 14)
        3. Compute Δi = Qi - W⁰ (line 16) 
        4. Solve argmin ||Δi - Bi*Ai||_F^2 (line 17)
        
        This implementation is specifically designed for DoRA and requires magnitude vectors.
        
        Raises:
            RuntimeError: If DoRA magnitude parameters are not found
        """
        with torch.no_grad():
            params = dict(self.model.named_parameters())
            
            for name, B in params.items():
                if 'lora_B' in name:
                    # Get related parameters
                    # Fix: Handle the .default.weight suffix correctly
                    if '.default.weight' in name:
                        # Remove both lora_B and .default.weight to get the base layer path
                        base_layer_path = name.replace('lora_B.default.weight', '')
                        
                        # Construct correct parameter names
                        A_name = base_layer_path + 'lora_A.default.weight'
                        weight_name = base_layer_path + 'base_layer.weight'
                        magnitude_name = base_layer_path + 'lora_magnitude_vector.default.weight'
                    else:
                        # Fallback to old logic if naming convention is different
                        base_name = name.replace('lora_B', '')
                        A_name = base_name + 'lora_A'
                        weight_name = base_name + 'base_layer.weight'
                        magnitude_name = base_name + 'lora_magnitude_vector'
                    
                    A = params.get(A_name)
                    W0 = params.get(weight_name)
                    magnitude = params.get(magnitude_name)
                    
                    if A is None or W0 is None:
                        continue
                    
                    # DoRA requires magnitude vector
                    if magnitude is None:
                        raise RuntimeError(f"FORCE QR requires DoRA magnitude vector for layer {name}, "
                                         f"but none found. Please ensure the model uses DoRA (use_dora=True).")
                    
                    # Step 1: Compute Mi = W⁰ + Bi*Ai (Algorithm 2, line 12)
                    Mi = W0 + B @ A
                    
                    # Step 2: Orthogonalize using QR decomposition (Algorithm 2, line 14)
                    Qi, _ = torch.qr(Mi)
                    
                    # Step 3: Compute target update: Δi = Qi - W⁰ (Algorithm 2, line 16)
                    # This is the key correction - subtract W0, not the current combined weight
                    delta_i = Qi - W0
                    
                    # Step 4: Solve optimization problem (Algorithm 2, line 17)
                    # argmin ||Δi - Bi*Ai||_F^2
                    # For DoRA, we need to consider the magnitude scaling in the optimization
                    
                    # Current B*A product
                    current_BA = B @ A
                    
                    # For DoRA, the relationship is more complex due to magnitude scaling
                    # We approximate the optimization by minimizing ||target_BA - B*A||_F^2
                    # where target_BA is derived from the orthogonalization target
                    target_BA = delta_i  # Simplified approximation for DoRA
                    
                    # Calculate residual for gradient computation
                    residual = current_BA - target_BA
                    
                    # Update B and A to minimize ||residual||_F^2
                    B_grad = residual @ A.T
                    A_grad = B.T @ residual
                    
                    # Apply gradient descent updates
                    B.add_(-lr_AB * B_grad)
                    A.add_(-lr_AB * A_grad)
    
    def get_parameters(self):
        """Return model parameters for federated averaging"""
        return copy.deepcopy(self.model.state_dict())
    
    def set_parameters(self, new_params):
        """Set model parameters received from the server"""
        self.model.load_state_dict(new_params)
        # Re-initialize optimizers after loading new parameters
        # This is crucial for federated learning to work properly
        if self.train_method and "muon" in self.train_method:
            # If using a muon method, use the method-specific initialization
            self._initialize_optimizers_for_method(self.train_method)
        else:
            # Otherwise use the default initialization
            self._initialize_optimizers()
        # Re-initialize learning rate schedulers after loading new parameters
        self._initialize_lr_schedulers()
        print(f"Client {self.client_id}: Reinitialized optimizers and schedulers after receiving new parameters")


class BaselineClient:
    """
    Baseline client implementation for standard federated learning.
    Supports standard Fed-LoRA and FFA-LoRA methods.
    """
    
    def __init__(self, client_id, model, data, device, baseline_method="lora", learning_rate=3e-4, num_epochs=2):
        """
        Initialize baseline client for comparison experiments
        
        Args:
            client_id: Client identifier
            model: Pre-trained model with LoRA already applied
            data: Local dataset
            device: Computing device
            baseline_method: "lora" for standard LoRA (FedIT), "ffa_lora" for FFA-LoRA
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
        """
        self.client_id = client_id
        self.device = device
        self.data = data
        self.baseline_method = baseline_method
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Use the model as-is (LoRA already applied)
        self.model = model
        
        # For FFA-LoRA, freeze A matrices after initialization
        if baseline_method == "ffa_lora":
            self._freeze_lora_a_matrices()
        
        # Set up client-specific logger
        self.logger = logging.getLogger(f"BaselineClient_{client_id}")
        
        # Initialize optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Initialize learning rate scheduler
        num_training_steps = len(self.data) * self.num_epochs
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
    
    def _freeze_lora_a_matrices(self):
        """Freeze all LoRA A matrices for FFA-LoRA implementation"""
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = False
    
    def train(self, epochs, gradient_accumulation_steps=1, **kwargs):
        """
        Local training on client's data - standard federated learning
        No orthogonal constraints or special regularization
        """
        self.model.to(self.device)
        self.model.train()
        
        # Track parameter changes
        initial_params = {name: param.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        
        total_loss = 0
        total_samples = 0
        epoch_pbar = tqdm(range(epochs), desc=f"Client {self.client_id} Training", leave=True)
        
        for epoch in epoch_pbar:
            epoch_loss = 0
            epoch_samples = 0
            batch_iter = tqdm(enumerate(self.data), desc=f"  Epoch {epoch+1}", total=len(self.data), leave=False)
            
            for step, batch in batch_iter:
                # Move batch tensors to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                else:
                    batch.to(self.device)
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                # Verify valid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss {loss.item()} at client {self.client_id}, step {step}")
                    continue
                
                # Backpropagation
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(self.data):
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                epoch_samples += batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
                
                batch_iter.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{self.lr_scheduler.get_last_lr()[0]:.6f}'})
            
            avg_epoch_loss = epoch_loss / len(self.data) if len(self.data) > 0 else 0
            total_loss += avg_epoch_loss
            epoch_pbar.set_postfix({'avg_loss': f'{avg_epoch_loss:.4f}', 'samples': epoch_samples})
        
        # Check parameter changes
        param_changes = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in initial_params:
                change = torch.norm(param - initial_params[name]).item()
                param_changes[name] = change
        
        max_change = max(param_changes.values()) if param_changes else 0.0
        avg_change = sum(param_changes.values()) / len(param_changes) if param_changes else 0.0
        
        final_avg_loss = total_loss / epochs if epochs > 0 else 0
        print(f"Client {self.client_id} ({self.baseline_method}) - Final avg loss: {final_avg_loss:.4f}, "
              f"Total samples: {total_samples}, Max param change: {max_change:.6f}, Avg param change: {avg_change:.6f}")
        
        if max_change < 1e-6:
            print(f"WARNING: Very small parameter changes detected for client {self.client_id}. Model may not be learning properly.")
    
    def get_parameters(self):
        """Return model parameters for federated averaging"""
        return copy.deepcopy(self.model.state_dict())
    
    def set_parameters(self, new_params):
        """Set model parameters received from the server"""
        self.model.load_state_dict(new_params)
        
        # Re-initialize optimizer and scheduler after loading new parameters
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Re-initialize learning rate scheduler
        num_training_steps = len(self.data) * self.num_epochs
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        print(f"Client {self.client_id}: Reinitialized optimizer and scheduler after receiving new parameters") 