# FORCE: Federated Orthogonalized Representation for Communication-Efficient LLMs Adaptation

## Overview

FORCE is a novel federated learning framework that enhances parameter-efficient fine-tuning (PEFT) by integrating Weight-Decomposed Low-Rank Adaptation (DoRA) with orthogonal constraints.

### Key Features

- **DoRA Integration**: Decomposes weight updates into low-rank matrices (A, B) and magnitude vectors (m) for efficient adaptation
- **Orthogonal Constraints**: Enforces orthogonality on directional components to ensure stable and consistent adaptations across clients
- **Multiple Enforcement Methods**: Supports both soft constraint (regularization) and hard constraint (post-training orthogonalization) approaches
- **Muon Optimizer**: Leverages matrix-aware optimization for accelerated convergence in non-IID settings, in this experiment we only adopt its logic in single process.
- **Communication Efficient**: Minimizes overhead by transmitting only low-rank adapter parameters

## Experiment Structure

This codebase provides a comprehensive federated learning experiment framework with the following components:

```
FORCE/
├── main.py                 # Main experiment runner with federated learning logic
├── client.py              # FORCE and baseline client implementations
├── server.py              # Federated learning server for model aggregation
├── data_utils.py          # GLUE dataset loading and preprocessing
├── data_distribution.py   # Non-IID data splitting with Dirichlet distribution
├── simple_muon.py         # Simplified Muon optimizer for single-process environments
├── plotting.py            # Individual experiment visualization
├── compare_experiments.py # Multi-experiment comparison and plotting tool
├── run_experiments.sh     # Automated batch experiment runner
├── experiments_config.txt # Experiment configuration file
└── environment.yml        # Conda environment dependencies
```

### Supported Methods

- **FORCE Methods**: `soft_constraint`, `Newton_shulz`, `QR`, `muon`
- **Combined Methods**: `soft_constraint+muon`, `Newton_shulz+muon`, `QR+muon`
- **Baseline Methods**: `lora` (FedIT), `ffa_lora` (FFA-LoRA)

### Supported Datasets

- GLUE benchmark tasks: `sst2`, `qqp`, `qnli`, `mnli_matched`, `mnli_mismatched`

## Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd FORCE
```

### 2. Setup Environment
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate FORCE
```
- **Note**: For using complete Muon Optimizer, please refer to https://github.com/KellerJordan/Muon.
- In this experiment setting, we simulate Muon's logic in the `simple_muon.py`

## Running Experiments

### Step 1: Change the Default parameter 

Edit `run_experiments.sh` with your desired hyperparameter

```bash
NUM_EPOCHS=2
NUM_ROUNDS=15
NUM_CLIENTS=8
LEARNING_RATE=3e-4
LORA_RANK=4
LORA_ALPHA=16
LORA_DROPOUT=0.1
BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=2
LAMBDA_ORTHO=0.0000005
ALPHA=0.3
MODEL_NAME="roberta-base"
BASE_EXP_DIR="experiments"
SEED=42
ENABLE_FEDERATED_SPLIT="--enable_federated_split"
BASE_PORT=12355
```

Edit `experiments_config.txt` with your desired experiments using the format: `method:dataset:gpu_id`

```bash
# Example configuration
soft_constraint:sst2:0
Newton_shulz:qqp:1
QR:mnli_matched:2
lora:qnli:3
                           <-- keep one empty line
```
- **Note**: DON'T FORGET to keep one empty line in your configuration, as shows above

### Key Parameters 
#### In `experiments_config.txt`:
- `--method`: Training method (see supported methods above)
- `--dataset`: GLUE dataset to use
#### In `run_experiments.sh`:
- `--num_clients`: Number of federated learning clients (default: 8)
- `--num_rounds`: Number of federated learning rounds (default: 15)
- `--num_epochs`: Local training epochs per round (default: 2)
- `--alpha`: Dirichlet distribution parameter for non-IID split (lower = more non-IID)
- `--enable_federated_split`: Enable realistic non-IID data distribution
- `--lambda_ortho`: Orthogonality regularization weight for soft constraint methods

### Step 2: Run

```bash
# Run all experiments in configuration file
./run_experiments.sh --config experiments_config.txt

# Monitor experiment status
./run_experiments.sh --status

# View help for additional options
./run_experiments.sh --help
```

This bash script based experiment runner provides:
- **Smart GPU Management**: Automatically queues experiments when GPUs are busy
- **Memory Monitoring**: Prevents OOM by checking GPU memory usage
- **Progress Tracking**: Real-time monitoring of experiment completion
- **Automatic Plotting**: Generates individual plots after each experiment

## Results Comparison and Visualization

### Interactive Comparison Tool

After running experiments, use the interactive comparison tool:

```bash
python compare_experiments.py
```

The tool provides:

1. **Experiment Scanning**: Automatically finds all completed experiments
2. **Grouped Display**: Shows experiments organized by dataset and parameters
3. **Selection Interface**: Multiple selection methods:
   - Single experiments: `1,3,5`
   - Ranges: `1-5`
   - All experiments: `all`
   - Mixed: `1,3,7-10,15`

4. **Validation**: Ensures experiments are comparable (same dataset, alpha, rounds)
5. **Automatic Plotting**: Generates comparison plots with proper legends and formatting

### Comparison Requirements

For meaningful comparison, experiments must have:
- ✅ **Same dataset** (e.g., all `qnli` or all `sst2`)
- ✅ **Same alpha value** (same data distribution)
- ✅ **Same number of rounds** (same training duration)

## Example Workflow

1. **Run experiments**:
```bash
./run_experiments.sh --config experiments_config.txt
```

2. **Compare results**:
```bash
python compare_experiments.py
```

3. **Select experiments** (example):
```
Enter your selection > 1-5,8,10-12
```

4. **View generated plots**: Comparison plots are automatically saved with timestamps

### MNLI Special Handling

For MNLI datasets, the framework automatically:
- Handles both matched and mismatched validation sets
- Plots dual accuracy curves for comprehensive evaluation
- Saves checkpoints for reuse between `mnli_matched` and `mnli_mismatched` experiments

## Output Structure

```
experiments/
├── alpha_0.3/                          # Non-IID parameter grouping
│   ├── data_distribution_qnli_c8/       # Data distribution visualizations
│   └── 20241219_143022_soft_constraint_qnli_c8_r15_gpu0_a1b2c3d4/
│       ├── config.json                  # Experiment configuration
│       ├── results.json                 # Training results and metrics
│       ├── logs/                        # Detailed training logs
│       ├── plots/                       # Individual experiment plots
│       └── data_distribution_location.txt
```

## Advanced Features

### Non-IID Data Distribution
- Implements Dirichlet distribution for realistic federated settings
- Visualizes data heterogeneity across clients
- Configurable via `--alpha` parameter (0.1 = highly non-IID, 1.0 = moderately non-IID)

### GPU Management
- Automatic GPU availability detection
- Process-based locking to prevent conflicts
- Memory usage monitoring
- Queue-based experiment scheduling

### Experiment Reproducibility
- Separate seeds for data distribution and model initialization
- Comprehensive configuration logging
- Automatic checkpoint saving and reuse (for MNLI dataset, since both matched and mismatched are using same training set.)

<!-- ## Citation

If you use this code in your research, please cite:

```bibtex
@article{force2024,
  title={FORCE: Federated Orthogonalized Representation for Communication-Efficient LLMs Adaptation},
  author={[Authors]},
  journal={[Venue]},
  year={2024}
}
``` -->

## Troubleshooting

**GPU Memory Issues**: Reduce `--batch_size` or increase `--gradient_accumulation_steps`

**MNLI Experiments**: The framework automatically reuses training between `mnli_matched` and `mnli_mismatched` for efficiency

**Comparison Errors**: Ensure experiments have the same dataset, alpha value, and number of rounds

**Environment Issues**: Use the provided `environment.yml` for exact dependency versions