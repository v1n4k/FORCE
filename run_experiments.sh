#!/bin/bash

# ===============================================================
# Easy run experiments bash script
# ===============================================================
# This script provides flexible execution of federated learning experiments:
# - Run experiments on specific GPUs with method/dataset combinations
# - Queue multiple experiments to run sequentially across available GPUs
# - Automatic plotting after each experiment completion
# - Support for both FORCE and baseline methods
# ===============================================================
# ############ Attention ############
# ===============================================================
# 1. This script is designed for single-process environments.
# 2. BEFEORE you start running experiments, you should check the 
#    default configuration parameters in the script.
#    see below in ---- DEFAULT CONFIGURATION PARAMETERS ---- section.
#    you can use "ctrl + f"/"cmd + f" to find the section.
# ===============================================================
#
# USAGE MODES:
# 1. Load experiments from configuration file (!!!!!Highly recommended!!!!!):
#    This is the most flexible way to run experiments, go to experiments_config.txt to arrange experiments.
#    ./run_experiments.sh --queue --config experiments_config.txt
#
# 2. Single experiment:
#    ./run_experiments.sh --method QR --dataset sst2 --gpu 0
#
# 3. Queue multiple experiments:
#    ./run_experiments.sh --queue --experiments "QR:sst2:0,lora:qqp:2,ffa_lora:qnli:1"
#
# 4. Batch experiments with method/dataset combinations:
#    ./run_experiments.sh --queue --methods "QR,lora" --datasets "sst2,qqp" --gpus "0,1,2,3"
#
# ===============================================================

# ---- HELPER FUNCTIONS ----
print_error() {
    echo -e "\e[31mERROR: $1\e[0m"
}

print_success() {
    echo -e "\e[32m$1\e[0m"
}

print_info() {
    echo -e "\e[34m$1\e[0m"
}

print_warning() {
    echo -e "\e[33m$1\e[0m"
}

# ---- MONITORING HELPER FUNCTIONS ----
show_running_experiments() {
    local exp_base_dir="${BASE_EXP_DIR}"
    if [[ ! -d "$exp_base_dir" ]]; then
        echo "No experiments have been run yet."
        return
    fi
    
    echo "==============================================================="
    echo "CURRENTLY RUNNING EXPERIMENTS"
    echo "==============================================================="
    
    local found_running=false
    for exp_dir in "$exp_base_dir"/*; do
        if [[ -d "$exp_dir" ]]; then
            local python_log="$exp_dir/logs/experiment.log"
            if [[ -f "$python_log" ]]; then
                # Check if experiment is still running (log file being written to)
                local exp_name=$(basename "$exp_dir")
                local last_modified=$(stat -c %Y "$python_log" 2>/dev/null || echo 0)
                local current_time=$(date +%s)
                local time_diff=$((current_time - last_modified))
                
                # If log was modified within last 60 seconds, consider it running
                if [[ $time_diff -lt 60 ]]; then
                    found_running=true
                    echo "🔄 $exp_name"
                    echo "   Log: $python_log"
                    echo "   Monitor: tail -f $python_log"
                    echo "   Last activity: $time_diff seconds ago"
                    echo ""
                fi
            fi
        fi
    done
    
    if [[ "$found_running" = false ]]; then
        echo "No experiments currently running."
    fi
    
    echo "==============================================================="
}

show_recent_experiments() {
    local exp_base_dir="${BASE_EXP_DIR}"
    local count=${1:-5}
    
    if [[ ! -d "$exp_base_dir" ]]; then
        echo "No experiments have been run yet."
        return
    fi
    
    echo "==============================================================="
    echo "RECENT EXPERIMENTS (Last $count)"
    echo "==============================================================="
    
    # Get most recent experiment directories
    local recent_dirs=($(ls -dt "$exp_base_dir"/* 2>/dev/null | head -n $count))
    
    for exp_dir in "${recent_dirs[@]}"; do
        if [[ -d "$exp_dir" ]]; then
            local exp_name=$(basename "$exp_dir")
            local python_log="$exp_dir/logs/experiment.log"
            local status="Unknown"
            local last_line=""
            
            if [[ -f "$python_log" ]]; then
                # Check if experiment completed successfully
                if grep -q "EXPERIMENT SUMMARY" "$python_log" 2>/dev/null; then
                    if tail -n 20 "$python_log" | grep -q "Results saved to" 2>/dev/null; then
                        status="✅ Completed"
                    else
                        status="❌ Failed"
                    fi
                else
                    # Check if still running
                    local last_modified=$(stat -c %Y "$python_log" 2>/dev/null || echo 0)
                    local current_time=$(date +%s)
                    local time_diff=$((current_time - last_modified))
                    
                    if [[ $time_diff -lt 60 ]]; then
                        status="🔄 Running"
                    else
                        status="⏸️ Stopped"
                    fi
                fi
                
                # Get last meaningful line
                last_line=$(tail -n 10 "$python_log" | grep -E "(FL Rounds|Best.*Accuracy|Final.*Accuracy|Error|ERROR)" | tail -n 1)
            fi
            
            echo "$status $exp_name"
            if [[ -n "$last_line" ]]; then
                echo "   Latest: $last_line"
            fi
            echo "   Log: $python_log"
            echo ""
        fi
    done
    
    echo "==============================================================="
}

# ---- EXPERIMENT MONITORING COMMANDS ----
if [[ "$1" == "--status" ]]; then
    show_running_experiments
    echo ""
    show_recent_experiments
    exit 0
fi

if [[ "$1" == "--monitor" ]]; then
    if [[ -z "$2" ]]; then
        print_error "Usage: $0 --monitor <experiment_name_pattern>"
        echo "Available experiments:"
        ls -1 "${BASE_EXP_DIR}/" 2>/dev/null | head -10
        exit 1
    fi
    
    local pattern="$2"
    local exp_base_dir="${BASE_EXP_DIR}"
    local matching_logs=($(find "$exp_base_dir" -name "*${pattern}*" -path "*/logs/experiment.log" 2>/dev/null))
    
    if [[ ${#matching_logs[@]} -eq 0 ]]; then
        print_error "No experiments found matching pattern: $pattern"
        echo "Available experiments:"
        ls -1 "$exp_base_dir" 2>/dev/null | head -10
        exit 1
    elif [[ ${#matching_logs[@]} -eq 1 ]]; then
        echo "Monitoring: ${matching_logs[0]}"
        echo "Press Ctrl+C to stop monitoring"
        tail -f "${matching_logs[0]}"
    else
        echo "Multiple experiments found matching pattern '$pattern':"
        for log in "${matching_logs[@]}"; do
            local exp_name=$(dirname $(dirname "$log"))
            exp_name=$(basename "$exp_name")
            echo "  - $exp_name: $log"
        done
        echo ""
        echo "Please be more specific with the pattern."
        exit 1
    fi
    exit 0
fi

# ---- DEFAULT CONFIGURATION PARAMETERS ----
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

# ---- EXECUTION MODE ----
QUEUE_MODE=false
SINGLE_MODE=true

# ---- EXPERIMENT QUEUE ----
EXPERIMENT_QUEUE=()

# ---- PORT CONFIGURATION FOR MUON METHODS ----
BASE_PORT=12355

# ---- PARSE COMMAND LINE ARGUMENTS ----
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --method)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --method"
        exit 1
      fi
      METHOD="$2"
      shift 2
      ;;
    --dataset)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --dataset"
        exit 1
      fi
      DATASET="$2"
      shift 2
      ;;
    --gpu)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --gpu"
        exit 1
      fi
      GPU_ID="$2"
      shift 2
      ;;
    --experiments)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --experiments"
        exit 1
      fi
      EXPERIMENTS_STRING="$2"
      shift 2
      ;;
    --methods)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --methods"
        exit 1
      fi
      METHODS_STRING="$2"
      shift 2
      ;;
    --datasets)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --datasets"
        exit 1
      fi
      DATASETS_STRING="$2"
      shift 2
      ;;
    --gpus)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --gpus"
        exit 1
      fi
      GPUS_STRING="$2"
      shift 2
      ;;
    --config)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --config"
        exit 1
      fi
      CONFIG_FILE="$2"
      shift 2
      ;;
    --queue)
      QUEUE_MODE=true
      SINGLE_MODE=false
      shift
      ;;
    --epochs)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --epochs"
        exit 1
      fi
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --rounds)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --rounds"
        exit 1
      fi
      NUM_ROUNDS="$2"
      shift 2
      ;;
    --clients)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --clients"
        exit 1
      fi
      NUM_CLIENTS="$2"
      shift 2
      ;;
    --lr)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --lr"
        exit 1
      fi
      LEARNING_RATE="$2"
      shift 2
      ;;
    --lora-rank)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --lora-rank"
        exit 1
      fi
      LORA_RANK="$2"
      shift 2
      ;;
    --lora-alpha)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --lora-alpha"
        exit 1
      fi
      LORA_ALPHA="$2"
      shift 2
      ;;
    --lora-dropout)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --lora-dropout"
        exit 1
      fi
      LORA_DROPOUT="$2"
      shift 2
      ;;
    --batch-size)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --batch-size"
        exit 1
      fi
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad-accum)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --grad-accum"
        exit 1
      fi
      GRADIENT_ACCUMULATION_STEPS="$2"
      shift 2
      ;;
    --lambda-ortho)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --lambda-ortho"
        exit 1
      fi
      LAMBDA_ORTHO="$2"
      shift 2
      ;;
    --alpha)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --alpha"
        exit 1
      fi
      ALPHA="$2"
      shift 2
      ;;
    --model)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --model"
        exit 1
      fi
      MODEL_NAME="$2"
      shift 2
      ;;
    --exp-dir)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --exp-dir"
        exit 1
      fi
      BASE_EXP_DIR="$2"
      shift 2
      ;;
    --seed)
      if [[ -z "$2" || "$2" == --* ]]; then
        print_error "Missing value for --seed"
        exit 1
      fi
      SEED="$2"
      shift 2
      ;;
    --enable-federated-split)
      ENABLE_FEDERATED_SPLIT="--enable_federated_split"
      shift
      ;;
    --help)
      echo "==============================================================="
      echo "FLEXIBLE FEDERATED LEARNING EXPERIMENT MANAGER - HELP"
      echo "==============================================================="
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "EXECUTION MODES:"
      echo "  1. Single experiment:"
      echo "     $0 --method <method> --dataset <dataset> --gpu <gpu_id>"
      echo ""
      echo "  2. Queue multiple experiments:"
      echo "     $0 --queue --experiments \"method1:dataset1:gpu1,method2:dataset2:gpu2\""
      echo ""
      echo "  3. Batch experiments:"
      echo "     $0 --queue --methods \"method1,method2\" --datasets \"dataset1,dataset2\" --gpus \"gpu1,gpu2\""
      echo ""
      echo "  4. Load from config file:"
      echo "     $0 --queue --config <config_file>"
      echo ""
      echo "MONITORING MODES:"
      echo "  5. View experiment status:"
      echo "     $0 --status"
      echo ""
      echo "  6. Monitor specific experiment:"
      echo "     $0 --monitor <experiment_pattern>"
      echo ""
      echo "METHODS:"
      echo "  FORCE methods: soft_constraint, Newton_shulz, QR, muon"
      echo "  Combined methods: soft_constraint+muon, Newton_shulz+muon, QR+muon"
      echo "  Baseline methods: lora, ffa_lora"
      echo ""
      echo "DATASETS: sst2, qqp, qnli, mnli_matched, mnli_mismatched"
      echo ""
      echo "PARAMETERS:"
      echo "  --epochs <n>                    Training epochs per round (default: 3)"
      echo "  --rounds <n>                    Communication rounds (default: 5)"
      echo "  --clients <n>                   Number of clients (default: 3)"
      echo "  --lr <rate>                     Learning rate (default: 3e-4)"
      echo "  --lora-rank <n>                 LoRA rank (default: 4)"
      echo "  --lora-alpha <n>                LoRA alpha (default: 16)"
      echo "  --lora-dropout <rate>           LoRA dropout (default: 0.1)"
      echo "  --batch-size <n>                Batch size (default: 32)"
      echo "  --grad-accum <n>                Gradient accumulation steps (default: 1)"
      echo "  --lambda-ortho <value>          Orthogonality weight (default: 0.1)"
      echo "  --alpha <value>                 Dirichlet alpha (default: 0.5)"
      echo "  --model <name>                  Model name (default: roberta-base)"
      echo "  --exp-dir <path>                Experiment directory (default: experiments)"
      echo "  --seed <n>                      Random seed (default: 42)"
      echo "  --enable-federated-split        Enable federated non-IID split"
      echo ""
      echo "EXAMPLES:"
      echo "  # Single experiment"
      echo "  $0 --method QR --dataset sst2 --gpu 0"
      echo ""
      echo "  # Queue multiple experiments"
echo "  $0 --queue --experiments \"QR:sst2:0,lora:qqp:2,ffa_lora:qnli:1\""
echo ""
echo "  # Batch experiments across GPUs"
echo "  $0 --queue --methods \"QR,lora,ffa_lora\" --datasets \"sst2,qqp\" --gpus \"0,1,2,3\""
echo ""
echo "  # Monitor experiments"
echo "  $0 --status                    # View all running and recent experiments"
echo "  $0 --monitor QR_sst2          # Monitor specific experiment by pattern"
echo ""
echo "CONFIG FILE FORMAT:"
echo "  method:dataset:gpu_id"
echo "  QR:sst2:0"
echo "  lora:qqp:2"
echo "  ffa_lora:qnli:1"
      echo "==============================================================="
      exit 0
      ;;
    *)
      print_error "Unknown option: $key"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# ---- VALIDATE INPUTS ----
validate_method() {
    local method=$1
    local valid_methods=("soft_constraint" "Newton_shulz" "QR" "muon" 
                        "soft_constraint+muon" "Newton_shulz+muon" "QR+muon"
                        "lora" "ffa_lora")
    
    for valid in "${valid_methods[@]}"; do
        if [[ "$method" == "$valid" ]]; then
            return 0
        fi
    done
    return 1
}

validate_dataset() {
    local dataset=$1
    local valid_datasets=("sst2" "qqp" "qnli" "mnli_matched" "mnli_mismatched")
    
    for valid in "${valid_datasets[@]}"; do
        if [[ "$dataset" == "$valid" ]]; then
            return 0
        fi
    done
    return 1
}

# ---- BUILD EXPERIMENT QUEUE ----
build_experiment_queue() {
    if [[ -n "$CONFIG_FILE" ]]; then
        # Load from config file
        if [[ ! -f "$CONFIG_FILE" ]]; then
            print_error "Config file not found: $CONFIG_FILE"
            exit 1
        fi
        
        while IFS= read -r line; do
            # Skip empty lines and comments
            if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
                continue
            fi
            
            # Parse line: method:dataset:gpu
            IFS=':' read -r method dataset gpu <<< "$line"
            
            # Validate inputs
            if ! validate_method "$method"; then
                print_warning "Invalid method in config: $method (line: $line)"
                continue
            fi
            
            if ! validate_dataset "$dataset"; then
                print_warning "Invalid dataset in config: $dataset (line: $line)"
                continue
            fi
            
            if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
                print_warning "Invalid GPU ID in config: $gpu (line: $line)"
                continue
            fi
            
            EXPERIMENT_QUEUE+=("$method:$dataset:$gpu")
            
        done < "$CONFIG_FILE"
        
    elif [[ -n "$EXPERIMENTS_STRING" ]]; then
        # Parse experiments string
        IFS=',' read -r -a experiments <<< "$EXPERIMENTS_STRING"
        
        for exp in "${experiments[@]}"; do
            IFS=':' read -r method dataset gpu <<< "$exp"
            
            # Validate inputs
            if ! validate_method "$method"; then
                print_warning "Invalid method: $method"
                continue
            fi
            
            if ! validate_dataset "$dataset"; then
                print_warning "Invalid dataset: $dataset"
                continue
            fi
            
            if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
                print_warning "Invalid GPU ID: $gpu"
                continue
            fi
            
            EXPERIMENT_QUEUE+=("$method:$dataset:$gpu")
        done
        
    elif [[ -n "$METHODS_STRING" && -n "$DATASETS_STRING" && -n "$GPUS_STRING" ]]; then
        # Parse batch configuration
        IFS=',' read -r -a methods <<< "$METHODS_STRING"
        IFS=',' read -r -a datasets <<< "$DATASETS_STRING"
        IFS=',' read -r -a gpus <<< "$GPUS_STRING"
        
        local exp_id=0
        for method in "${methods[@]}"; do
            for dataset in "${datasets[@]}"; do
                # Validate method and dataset
                if ! validate_method "$method"; then
                    print_warning "Invalid method: $method"
                    continue
                fi
                
                if ! validate_dataset "$dataset"; then
                    print_warning "Invalid dataset: $dataset"
                    continue
                fi
                
                # Assign GPU (round-robin)
                local gpu_idx=$((exp_id % ${#gpus[@]}))
                local gpu=${gpus[$gpu_idx]}
                
                EXPERIMENT_QUEUE+=("$method:$dataset:$gpu")
                exp_id=$((exp_id + 1))
            done
        done
    fi
}

# ---- CREATE OUTPUT DIRECTORY ----
mkdir -p $BASE_EXP_DIR 2>/dev/null || { print_error "Failed to create output directory: $BASE_EXP_DIR"; exit 1; }

# ---- DISPLAY CONFIGURATION ----
echo "==============================================================="
echo "FEDERATED LEARNING EXPERIMENT CONFIGURATION"
echo "==============================================================="
echo "Epochs per Round: $NUM_EPOCHS"
echo "Communication Rounds: $NUM_ROUNDS"
echo "Number of Clients: $NUM_CLIENTS"
echo "Learning Rate: $LEARNING_RATE"
echo "LoRA Rank: $LORA_RANK"
echo "LoRA Alpha: $LORA_ALPHA"
echo "LoRA Dropout: $LORA_DROPOUT"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Lambda Ortho: $LAMBDA_ORTHO"
echo "Alpha: $ALPHA"
echo "Model: $MODEL_NAME"
echo "Base Experiment Directory: $BASE_EXP_DIR"
echo "Seed: $SEED"
if [[ -n "$ENABLE_FEDERATED_SPLIT" ]]; then
    echo "Federated Split: Enabled"
fi
echo "==============================================================="

# ---- EXPERIMENT RUNNER FUNCTION ----
run_experiment() {
    local method=$1
    local dataset=$2
    local cuda_device=$3
    local exp_id=$4

    echo "==============================================================="
    echo "Running $method on $dataset with GPU $cuda_device (Experiment ID: $exp_id)"
    echo "==============================================================="

    # For federated learning fairness:
    # - Use fixed seed for data distribution (all methods see same data split)
    # - Use method-specific seed for model initialization (different initializations)
    local data_seed=$SEED  # Fixed for all methods
    local method_hash=$(echo -n "$method" | cksum | cut -d' ' -f1)
    local model_seed=$((SEED + method_hash % 1000))
    
    echo "  - Data distribution seed: $data_seed (fixed for fair comparison)"
    echo "  - Model initialization seed: $model_seed (method-specific)"

    # Set up MUON environment if needed
    local port_param=""
    if [[ "$method" == *"muon"* ]]; then
        local port=$((BASE_PORT + exp_id*100 + 10#$cuda_device))
        # Set MASTER_PORT as environment variable instead of command line argument
        # since main.py has its own port allocation logic
        export MASTER_ADDR="localhost"
        export MASTER_PORT="$port"
        echo "  - MUON method detected: Using communication port $port"
    fi

    # Record start time
    START_TIME=$(date +%s)
    echo "  - Started at: $(date)"

    # Run the experiment directly without progress filtering
    CUDA_VISIBLE_DEVICES=$cuda_device python -u main.py \
        --method $method \
        --dataset $dataset \
        --model_name $MODEL_NAME \
        --num_clients $NUM_CLIENTS \
        --num_rounds $NUM_ROUNDS \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --lambda_ortho $LAMBDA_ORTHO \
        --alpha $ALPHA \
        --seed $SEED \
        --data_seed $data_seed \
        --model_seed $model_seed \
        --cuda_device 0 \
        --exp_dir "$BASE_EXP_DIR" \
        $ENABLE_FEDERATED_SPLIT

    RESULT=$?

    # Record end time and duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(( (DURATION % 3600) / 60 ))
    SECONDS=$((DURATION % 60))

    # Find the experiment directory created by Python
    # Replace + with _ in method name for directory matching
    local method_dir=$(echo "$method" | tr '+' '_')
    local exp_dir_pattern1="${BASE_EXP_DIR}/*${method_dir}_${dataset}*"
    local exp_dir_pattern2="${BASE_EXP_DIR}/alpha_*/*${method_dir}_${dataset}*"
    local exp_dirs=($(ls -dt $exp_dir_pattern1 $exp_dir_pattern2 2>/dev/null))
    local latest_exp_dir=""
    local python_log=""
    
    if [[ ${#exp_dirs[@]} -gt 0 ]]; then
        latest_exp_dir=${exp_dirs[0]}  # Get the most recent directory
        python_log="${latest_exp_dir}/logs/experiment.log"
        
        # Export the experiment directory so it can be accessed by the parent shell
        echo "EXPERIMENT_DIR:$latest_exp_dir" > /tmp/exp_${exp_id}_dir.txt
    fi

    # Report results
    echo ""
    echo "==============================================================="
    if [ $RESULT -eq 0 ]; then
        print_success "✓ Experiment completed successfully!"
        echo "  Method: $method"
        echo "  Dataset: $dataset"
        echo "  GPU: $cuda_device"
        echo "  Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
        if [[ -n "$latest_exp_dir" ]]; then
            echo "  Results: $latest_exp_dir"
        fi
        if [[ -n "$python_log" && -f "$python_log" ]]; then
            echo "  Log: $python_log"
        fi
    else
        print_error "✗ Experiment failed!"
        echo "  Method: $method"
        echo "  Dataset: $dataset"
        echo "  GPU: $cuda_device"
        echo "  Duration before error: ${HOURS}h ${MINUTES}m ${SECONDS}s"
        if [[ -n "$python_log" && -f "$python_log" ]]; then
            echo "  Error log: $python_log"
            echo "  Last error lines:"
            tail -n 10 "$python_log" | sed 's/^/    /'
        fi
    fi
    echo "==============================================================="

    return $RESULT
}

# ---- POST-EXPERIMENT PLOT GENERATION ----
generate_plots_for_experiment() {
    local exp_dir=$1
    local method=$2
    local dataset=$3
    local gpu=$4
    
    if [[ -z "$exp_dir" || ! -d "$exp_dir" ]]; then
        print_error "Cannot generate plots: experiment directory not found"
        return 1
    fi
    
    echo ""
    echo "Generating plots for $method on $dataset..."
    
    # Generate plots directly
    if python plotting.py --exp_dir "$exp_dir" --output_dir "$exp_dir/plots" 2>&1; then
        print_success "✓ Plots generated successfully for $method on $dataset"
        return 0
    else
        print_warning "⚠ Plot generation failed for $method on $dataset"
        return 1
    fi
}

# ---- MAIN EXECUTION ----
if $QUEUE_MODE; then
    # Build experiment queue
    build_experiment_queue
    
    if [[ ${#EXPERIMENT_QUEUE[@]} -eq 0 ]]; then
        print_error "No valid experiments in queue"
        exit 1
    fi
    
    echo "==============================================================="
    echo "EXPERIMENT QUEUE"
    echo "==============================================================="
    echo "Total experiments: ${#EXPERIMENT_QUEUE[@]}"
    echo ""
    for i in "${!EXPERIMENT_QUEUE[@]}"; do
        IFS=':' read -r method dataset gpu <<< "${EXPERIMENT_QUEUE[$i]}"
        echo "  $((i+1)). $method on $dataset (GPU $gpu)"
    done
    echo "==============================================================="
    
    # Track running experiments
    declare -A running_pids
    declare -A pid_to_exp
    declare -A exp_dirs
    
    # Start experiments in parallel
    exp_id=0
    for exp in "${EXPERIMENT_QUEUE[@]}"; do
        IFS=':' read -r method dataset gpu <<< "$exp"
        
        echo ""
        echo "▶ Starting experiment $((exp_id+1))/${#EXPERIMENT_QUEUE[@]}: $method on $dataset (GPU $gpu)"
        
        # Run experiment in background and capture its PID
        (
            run_experiment "$method" "$dataset" "$gpu" "$exp_id"
            exit_code=$?
            
            # Only generate plots if experiment succeeded
            if [[ $exit_code -eq 0 ]]; then
                # Read the experiment directory from the temp file
                if [[ -f "/tmp/exp_${exp_id}_dir.txt" ]]; then
                    exp_dir=$(grep "EXPERIMENT_DIR:" "/tmp/exp_${exp_id}_dir.txt" | cut -d: -f2)
                    rm -f "/tmp/exp_${exp_id}_dir.txt"
                    
                    if [[ -n "$exp_dir" && -d "$exp_dir" ]]; then
                        # Wait a bit to ensure all files are written
                        sleep 2
                        
                        # Check if results.json exists
                        if [[ -f "$exp_dir/results.json" ]]; then
                            echo ""
                            echo "Generating plots for: $exp_dir"
                            generate_plots_for_experiment "$exp_dir" "$method" "$dataset" "$gpu"
                        else
                            echo "Warning: results.json not found in $exp_dir"
                        fi
                    else
                        echo "Warning: Experiment directory not found or invalid: $exp_dir"
                    fi
                else
                    echo "Warning: Could not find experiment directory info for $method on $dataset"
                fi
            else
                echo "Skipping plot generation for failed experiment: $method on $dataset"
            fi
        ) &
        
        pid=$!
        running_pids[$gpu]=$pid
        pid_to_exp[$pid]="$method:$dataset:$gpu"
        
        exp_id=$((exp_id + 1))
        
        # Small delay to avoid race conditions
        sleep 2
    done
    
    # Wait for all experiments to complete
    echo ""
    echo "Waiting for all experiments to complete..."
    echo ""
    
    completed=0
    total=${#EXPERIMENT_QUEUE[@]}
    
    # Monitor running experiments
    while [[ $completed -lt $total ]]; do
        for pid in "${!pid_to_exp[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                # Process completed
                exp_info=${pid_to_exp[$pid]}
                IFS=':' read -r method dataset gpu <<< "$exp_info"
                
                wait $pid
                exit_code=$?
                
                if [[ $exit_code -eq 0 ]]; then
                    echo "✓ Completed: $method on $dataset (GPU $gpu)"
                else
                    echo "✗ Failed: $method on $dataset (GPU $gpu)"
                fi
                
                unset pid_to_exp[$pid]
                completed=$((completed + 1))
                echo "Progress: $completed/$total experiments completed"
            fi
        done
        
        # Brief sleep to avoid busy waiting
        if [[ $completed -lt $total ]]; then
            sleep 5
        fi
    done
    
    echo ""
    echo "==============================================================="
    print_success "All queued experiments completed!"
    echo "==============================================================="
    
else
    # Single experiment mode
    if [[ -z "$METHOD" || -z "$DATASET" || -z "$GPU_ID" ]]; then
        print_error "Single experiment mode requires --method, --dataset, and --gpu"
        echo "Use --help for usage information"
        exit 1
    fi
    
    # Validate inputs
    if ! validate_method "$METHOD"; then
        print_error "Invalid method: $METHOD"
        exit 1
    fi
    
    if ! validate_dataset "$DATASET"; then
        print_error "Invalid dataset: $DATASET"
        exit 1
    fi
    
    if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
        print_error "Invalid GPU ID: $GPU_ID"
        exit 1
    fi
    
    echo "==============================================================="
    echo "SINGLE EXPERIMENT MODE"
    echo "==============================================================="
    
    run_experiment "$METHOD" "$DATASET" "$GPU_ID" 0
    
    echo ""
    print_success "Single experiment completed!"
fi 