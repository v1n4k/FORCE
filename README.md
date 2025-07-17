# FORCE: Federated Learning with Orthogonal Regularization and Constraint Enforcement

A comprehensive implementation of FORCE (Federated Learning with Orthogonal Regularization and Constraint Enforcement) methods for federated learning experiments.

## Overview

This project implements various FORCE methods for federated learning, including:

- **Soft Constraint**: Orthogonality regularization during training
- **Newton-Schulz**: Post-training orthogonal repair using Newton-Schulz algorithm
- **QR Decomposition**: Post-training orthogonal repair using QR decomposition
- **Muon Optimizer**: Matrix parameter optimization for better convergence

## Features

- **Multiple FORCE Methods**: Implementation of all major FORCE algorithms
- **Non-IID Data Support**: Dirichlet distribution-based data splitting for realistic federated scenarios
- **DoRA Integration**: Support for Weight-Decomposed Low-Rank Adaptation
- **Comprehensive Evaluation**: Support for GLUE benchmark datasets
- **Experiment Management**: Automated experiment tracking and result comparison
- **Multi-GPU Support**: Distributed training capabilities

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd FORCE

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Individual Experiments

```bash
python main.py \
    --method soft_constraint \
    --dataset qnli \
    --num_clients 3 \
    --num_rounds 10 \
    --alpha 0.5 \
    --cuda_device 0
```

### Running Multiple Experiments

```bash
# Run experiments from configuration file
./run_experiments.sh --queue --config experiments_config.txt
```

### Comparing Results

```bash
# Interactive comparison tool
python compare_experiments.py
```

## Project Structure

```
FORCE/
├── main.py                 # Main experiment runner
├── client.py              # FORCE and baseline client implementations
├── server.py              # Federated learning server
├── data_utils.py          # Data loading and preprocessing
├── data_distribution.py   # Non-IID data splitting utilities
├── plotting.py            # Visualization and reporting
├── compare_experiments.py # Experiment comparison tool
├── simple_muon.py         # Simplified Muon optimizer implementation
├── run_experiments.sh     # Batch experiment runner
└── experiments_config.txt # Experiment configuration
```

## Supported Methods

### FORCE Methods
- `soft_constraint`: Orthogonality regularization during training
- `Newton_shulz`: Newton-Schulz orthogonalization after each epoch
- `QR`: QR decomposition for orthogonalization
- `muon`: Muon optimizer for matrix parameters

### Combined Methods
- `soft_constraint+muon`: Soft constraint with Muon optimizer
- `Newton_shulz+muon`: Newton-Schulz with Muon optimizer
- `QR+muon`: QR decomposition with Muon optimizer

### Baseline Methods
- `lora`: Standard LoRA (FedIT)
- `ffa_lora`: FFA-LoRA implementation

## Supported Datasets

- **SST-2**: Stanford Sentiment Treebank
- **QQP**: Quora Question Pairs
- **QNLI**: Question Natural Language Inference
- **MNLI**: Multi-Genre Natural Language Inference (matched/mismatched)

## Configuration

Experiments can be configured using the `experiments_config.txt` file:

```
# Format: method:dataset:gpu_id
soft_constraint:qnli:0
QR+muon:mnli_matched:1
lora:sst2:2
```

## Results

Experiment results are automatically saved to the `experiments/` directory with:
- Configuration files
- Training logs
- Accuracy plots
- Performance metrics
- Data distribution analysis