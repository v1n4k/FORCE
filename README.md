# FORCE: Federated Learning with Orthogonal Regularization and Constraint Enforcement

A comprehensive implementation of FORCE methods for federated learning experiments. This project implements various orthogonal regularization techniques to improve federated learning performance on non-IID data distributions.

## Project Overview

This codebase provides:
- **FORCE Methods**: Soft constraint, Newton-Schulz, QR decomposition, and Muon optimizer implementations
- **Baseline Methods**: Standard LoRA and FFA-LoRA for comparison
- **Non-IID Data Support**: Dirichlet distribution-based data splitting
- **Automated Experiment Management**: Batch execution and result comparison tools

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd FORCE

# Create and activate conda environment
conda env create -f environment.yml
conda activate FORCE
```

**Note**: This experiment is designed for single-process environments. The Muon optimizer implementation used here is from `simple_muon.py`, which is a simplified version that works without distributed training dependencies.

### Running Experiments

1. **Configure your experiments** in `experiments_config.txt`:
```
# Format: method:dataset:gpu_id
soft_constraint:qnli:0
QR:qnli:1
lora:qnli:2
```

2. **Run all experiments**:
```bash
./run_experiments.sh --queue --config experiments_config.txt
```

3. **Monitor progress**:
```bash
./run_experiments.sh --status
```

4. **Compare results**:
```bash
python compare_experiments.py
```

## Configuration File

The `experiments_config.txt` file uses simple format: `method:dataset:gpu_id`

### Supported Methods
- **FORCE**: `soft_constraint`, `Newton_shulz`, `QR`, `muon`
- **Combined**: `soft_constraint+muon`, `Newton_shulz+muon`, `QR+muon`
- **Baseline**: `lora`, `ffa_lora`

### Supported Datasets
- `sst2`, `qqp`, `qnli`, `mnli_matched`, `mnli_mismatched`

### Example Configurations

**Method Comparison:**
```
lora:qnli:0
soft_constraint:qnli:1
QR:qnli:2
```

**Multi-Dataset:**
```
soft_constraint:sst2:0
soft_constraint:qqp:1
soft_constraint:qnli:2
```

## Important Notes

- **Single-Process Design**: This experiment is designed for single-process environments. The Muon optimizer used is from `simple_muon.py`, not the distributed version from the original Muon repository.
- **GPU Requirements**: Ensure sufficient GPU memory for your batch size and model configuration
- **Results**: All experiment results are automatically saved to `experiments/` directory

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
├── simple_muon.py         # Simplified Muon optimizer for single-process
├── run_experiments.sh     # Batch experiment runner
├── experiments_config.txt # Experiment configuration
└── environment.yml        # Conda environment file
```