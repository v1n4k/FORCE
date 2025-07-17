"""
Simplified Muon Optimizer for Single-Process Environments

This module provides a simplified version of the Muon optimizer that works
in single-process federated learning environments, removing the distributed
training dependencies while maintaining the core optimization logic.

The simplified version includes:
- Momentum-based updates for matrix parameters
- Matrix-specific optimization strategies
- Compatibility with single-process federated learning
- Similar API to the original Muon optimizer
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from typing import List, Optional, Dict, Any


class SimpleMuon(Optimizer):
    """
    Simplified Muon optimizer for single-process environments.
    
    This optimizer is designed for matrix parameters and provides:
    - Momentum-based updates
    - Matrix-specific optimization strategies
    - Compatibility with single-process federated learning
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum: Momentum factor (default: 0.9)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        dampening: Dampening for momentum (default: 0)
        nesterov: Enable Nesterov momentum (default: False)
        matrix_scale: Scaling factor for matrix updates (default: 1.0)
    """
    
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0,
                 dampening=0, nesterov=False, matrix_scale=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= dampening:
            raise ValueError(f"Invalid dampening value: {dampening}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov,
                       matrix_scale=matrix_scale)
        super(SimpleMuon, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SimpleMuon, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            matrix_scale = group['matrix_scale']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                # Apply Muon orthogonalization for matrix parameters
                if p.dim() >= 2:  # Matrix parameters
                    # Apply Newton-Schulz orthogonalization
                    d_p_ortho = simple_zeropower_via_newtonschulz5(d_p)
                    
                    # Scale by matrix dimensions and gradient norm
                    d_out, d_in = p.shape[0], p.shape[1]
                    scale = (d_out * d_in) ** 0.5
                    grad_norm = d_p.norm()
                    
                    if grad_norm > 1e-7:
                        d_p = d_p_ortho * (scale / grad_norm) * matrix_scale
                
                # Update parameters
                p.data.add_(d_p, alpha=-group['lr'])
        
        return loss


def simple_zeropower_via_newtonschulz(matrix, steps=5, tolerance=1e-6):
    """
    Simplified Newton-Schulz algorithm for computing matrix power.
    
    This is a simplified version that works in single-process environments.
    
    Args:
        matrix: Input matrix (torch.Tensor)
        steps: Number of Newton-Schulz iterations (default: 5)
        tolerance: Convergence tolerance (default: 1e-6)
    
    Returns:
        Approximated matrix power result
    """
    if matrix.dim() != 2:
        raise ValueError("Input matrix must be 2-dimensional")
    
    # Initialize with identity matrix
    n = matrix.size(0)
    result = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
    
    # Newton-Schulz iteration
    for _ in range(steps):
        # Compute next iteration
        next_result = 0.5 * (3 * result - result @ matrix @ result)
        
        # Check convergence
        if torch.norm(next_result - result) < tolerance:
            break
        
        result = next_result
    
    return result


def simple_zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration for orthogonalization (5th order polynomial).
    This is the correct implementation following the original Muon algorithm.
    
    Args:
        G: Input gradient matrix
        steps: Number of Newton-Schulz iterations
        eps: Small value for numerical stability
        
    Returns:
        Orthogonalized matrix
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimized coefficients
    
    # Normalize by Frobenius norm
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    
    # Handle rectangular matrices
    if G.size(0) > G.size(1):
        X = X.T
        
    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # b*XX^T + c*(XX^T)^2
        X = a * X + B @ X      # aX + (bXX^T + c(XX^T)^2)X
        
    # Restore original shape
    if G.size(0) > G.size(1):
        X = X.T
        
    return X


class SimpleMuonMatrixOptimizer:
    """
    Matrix-specific optimizer that combines SimpleMuon with matrix operations.
    
    This class provides additional matrix-specific optimization features
    that are commonly used with Muon optimizer.
    """
    
    def __init__(self, matrix_params, lr=1e-3, momentum=0.9, matrix_scale=1.0):
        """
        Initialize the matrix optimizer.
        
        Args:
            matrix_params: List of matrix parameters to optimize
            lr: Learning rate
            momentum: Momentum factor
            matrix_scale: Scaling factor for matrix updates
        """
        self.optimizer = SimpleMuon(
            matrix_params,
            lr=lr,
            momentum=momentum,
            matrix_scale=matrix_scale
        )
        self.matrix_params = matrix_params
    
    def step(self):
        """Perform optimization step."""
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def get_last_lr(self):
        """Get the last learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """Get optimizer state."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)


# Compatibility functions for existing code
def create_simple_muon_optimizer(params, lr=1e-3, momentum=0.9):
    """
    Create a SimpleMuon optimizer for compatibility with existing code.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate
        momentum: Momentum factor
    
    Returns:
        SimpleMuon optimizer instance
    """
    return SimpleMuon(params, lr=lr, momentum=momentum)


def is_matrix_parameter(param):
    """
    Check if a parameter is a matrix parameter.
    
    Args:
        param: PyTorch parameter
    
    Returns:
        True if parameter is a matrix (2D or higher), False otherwise
    """
    return param.dim() >= 2


def separate_matrix_and_vector_params(model_params):
    """
    Separate matrix and vector parameters.
    
    Args:
        model_params: Model parameters
    
    Returns:
        tuple: (matrix_params, vector_params)
    """
    matrix_params = []
    vector_params = []
    
    for param in model_params:
        if param.requires_grad:
            if is_matrix_parameter(param):
                matrix_params.append(param)
            else:
                vector_params.append(param)
    
    return matrix_params, vector_params 