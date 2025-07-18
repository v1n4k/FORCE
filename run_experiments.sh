#!/bin/bash

# ===============================================================
# Intelligent Federated Learning Experiment Runner
# ===============================================================
# Features:
# - Smart GPU queue management - only runs when GPU is available
# - Automatic experiment status monitoring
# - Configuration file support for batch experiments
# - Automatic plot generation after completion
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

# ---- GPU STATUS FUNCTIONS ----
check_gpu_processes() {
    local gpu_id=$1
    # Check if there are any python processes using the GPU
    local processes=$(nvidia-smi -i $gpu_id --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -i python | wc -l)
    echo $processes
}

get_gpu_memory_usage() {
    local gpu_id=$1
    # Get memory usage percentage for specific GPU
    local mem_info=$(nvidia-smi -i $gpu_id --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    if [[ -z "$mem_info" ]]; then
        echo "100"  # Return 100% if can't get info (assume busy)
        return
    fi
    
    local mem_used=$(echo $mem_info | cut -d',' -f1 | xargs)
    local mem_total=$(echo $mem_info | cut -d',' -f2 | xargs)
    
    if [[ -n "$mem_used" && -n "$mem_total" && "$mem_total" -gt 0 ]]; then
        local mem_percent=$(( (mem_used * 100) / mem_total ))
        echo $mem_percent
    else
        echo "100"
    fi
}

get_gpu_free_memory() {
    local gpu_id=$1
    # Get free memory in MB
    local free_mem=$(nvidia-smi -i $gpu_id --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | xargs)
    if [[ -z "$free_mem" ]]; then
        echo "0"
    else
        echo $free_mem
    fi
}

is_gpu_available() {
    local gpu_id=$1
    
    # Check 1: Are there any python processes on this GPU?
    local num_processes=$(check_gpu_processes $gpu_id)
    if [[ $num_processes -gt 0 ]]; then
        return 1  # GPU is busy
    fi
    
    # Check 2: Is memory usage below threshold?
    local mem_usage=$(get_gpu_memory_usage $gpu_id)
    if [[ $mem_usage -gt $MAX_GPU_MEMORY_PERCENT ]]; then
        return 1  # GPU memory is too high
    fi
    
    # Check 3: Is there enough free memory?
    local free_mem=$(get_gpu_free_memory $gpu_id)
    if [[ $free_mem -lt $MIN_FREE_MEMORY_MB ]]; then
        return 1  # Not enough free memory
    fi
    
    return 0  # GPU is available
}

wait_for_gpu() {
    local gpu_id=$1
    local experiment_name=$2
    
    print_info "Checking GPU $gpu_id availability for: $experiment_name"
    
    while true; do
        if is_gpu_available $gpu_id; then
            print_success "GPU $gpu_id is available!"
            break
        else
            local processes=$(check_gpu_processes $gpu_id)
            local mem_usage=$(get_gpu_memory_usage $gpu_id)
            local free_mem=$(get_gpu_free_memory $gpu_id)
            
            echo "  GPU $gpu_id busy: ${processes} processes, ${mem_usage}% memory used, ${free_mem}MB free"
            echo "  Waiting ${GPU_CHECK_INTERVAL} seconds..."
            sleep $GPU_CHECK_INTERVAL
        fi
    done
}

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
BASE_PORT=12355

# GPU monitoring settings
GPU_CHECK_INTERVAL=30  # Check GPU status every 30 seconds
MAX_GPU_MEMORY_PERCENT=85  # Maximum GPU memory usage before considering it "busy"
MIN_FREE_MEMORY_MB=2000  # Minimum free memory required to start new experiment

# ---- PARSE COMMAND LINE ARGUMENTS ----
EXPERIMENT_QUEUE=()
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpu-check-interval)
            GPU_CHECK_INTERVAL="$2"
            shift 2
            ;;
        --max-gpu-memory)
            MAX_GPU_MEMORY_PERCENT="$2"
            shift 2
            ;;
        --min-free-memory)
            MIN_FREE_MEMORY_MB="$2"
            shift 2
            ;;
        --help)
            echo "==============================================================="
            echo "INTELLIGENT FEDERATED LEARNING EXPERIMENT RUNNER"
            echo "==============================================================="
            echo "Usage: $0 --config <config_file> [options]"
            echo ""
            echo "Options:"
            echo "  --config <file>              Configuration file with experiments"
            echo "  --gpu-check-interval <sec>   GPU check interval (default: 30)"
            echo "  --max-gpu-memory <percent>   Max GPU memory usage (default: 85)"
            echo "  --min-free-memory <MB>       Min free memory required (default: 2000)"
            echo ""
            echo "Config file format (experiments_config.txt):"
            echo "  method:dataset:gpu_id"
            echo "  soft_constraint:qnli:0"
            echo "  QR+muon:qqp:1"
            echo "  lora:sst2:2"
            echo ""
            echo "Features:"
            echo "  - Automatically queues experiments when GPU is busy"
            echo "  - Monitors GPU memory to prevent OOM"
            echo "  - Generates plots after each experiment"
            echo "==============================================================="
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ---- VALIDATE CONFIG FILE ----
if [[ -z "$CONFIG_FILE" ]]; then
    print_error "Config file required. Use --config <file>"
    echo "Run $0 --help for usage information"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# ---- BUILD EXPERIMENT QUEUE ----
while IFS= read -r line; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Parse line: method:dataset:gpu
    IFS=':' read -r method dataset gpu <<< "$line"
    
    # Basic validation
    if [[ -n "$method" && -n "$dataset" && "$gpu" =~ ^[0-9]+$ ]]; then
        EXPERIMENT_QUEUE+=("$method:$dataset:$gpu")
    else
        print_warning "Invalid line in config: $line"
    fi
done < "$CONFIG_FILE"

if [[ ${#EXPERIMENT_QUEUE[@]} -eq 0 ]]; then
    print_error "No valid experiments found in config file"
    exit 1
fi

# ---- CHECK NVIDIA-SMI ----
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# ---- DISPLAY CONFIGURATION ----
echo "==============================================================="
echo "INTELLIGENT EXPERIMENT RUNNER"
echo "==============================================================="
echo "Config file: $CONFIG_FILE"
echo "Total experiments: ${#EXPERIMENT_QUEUE[@]}"
echo ""
echo "GPU Management:"
echo "  Max memory usage: ${MAX_GPU_MEMORY_PERCENT}%"
echo "  Min free memory: ${MIN_FREE_MEMORY_MB}MB"
echo "  Check interval: ${GPU_CHECK_INTERVAL}s"
echo ""
echo "Experiments to run:"
for i in "${!EXPERIMENT_QUEUE[@]}"; do
    IFS=':' read -r method dataset gpu <<< "${EXPERIMENT_QUEUE[$i]}"
    echo "  $((i+1)). $method on $dataset (GPU $gpu)"
done
echo "==============================================================="

# ---- EXPERIMENT RUNNER FUNCTION ----
run_experiment_with_queue() {
    local method=$1
    local dataset=$2
    local gpu_id=$3
    local exp_id=$4
    
    local experiment_name="${method}_${dataset}"
    
    # Wait for GPU to be available
    wait_for_gpu $gpu_id "$experiment_name"
    
    echo ""
    echo "==============================================================="
    echo "Starting: $method on $dataset with GPU $gpu_id"
    echo "==============================================================="
    
    # Set up seeds for fair comparison
    local data_seed=$SEED
    local method_hash=$(echo -n "$method" | cksum | cut -d' ' -f1)
    local model_seed=$((SEED + method_hash % 1000))
    
    # Set up MUON environment if needed
    if [[ "$method" == *"muon"* ]]; then
        local port=$((BASE_PORT + exp_id*100 + gpu_id*10))
        export MASTER_ADDR="localhost"
        export MASTER_PORT="$port"
        echo "  - MUON port: $port"
    fi
    
    # Record start time
    START_TIME=$(date +%s)
    echo "  - Started at: $(date)"
    
    # Run the experiment
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main.py \
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
    
    # Record end time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(( (DURATION % 3600) / 60 ))
    SECONDS=$((DURATION % 60))
    
    if [ $RESULT -eq 0 ]; then
        print_success "✓ Completed: $method on $dataset (${HOURS}h ${MINUTES}m ${SECONDS}s)"
        
        # Generate plots
        local method_dir=$(echo "$method" | tr '+' '_')
        local exp_dirs=($(ls -dt ${BASE_EXP_DIR}/*${method_dir}_${dataset}* 2>/dev/null))
        if [[ ${#exp_dirs[@]} -gt 0 ]]; then
            echo "  Generating plots..."
            python plotting.py --exp_dir "${exp_dirs[0]}" --output_dir "${exp_dirs[0]}/plots" 2>&1
        fi
    else
        print_error "✗ Failed: $method on $dataset"
    fi
    
    return $RESULT
}

# ---- MAIN EXECUTION ----
mkdir -p $BASE_EXP_DIR 2>/dev/null

# Start all experiments (they will queue automatically)
declare -a pids
exp_id=0

for exp in "${EXPERIMENT_QUEUE[@]}"; do
    IFS=':' read -r method dataset gpu <<< "$exp"
    
    (
        run_experiment_with_queue "$method" "$dataset" "$gpu" "$exp_id"
    ) &
    
    pids+=($!)
    exp_id=$((exp_id + 1))
    
    # Small delay to avoid race conditions
    sleep 2
done

# Monitor progress
echo ""
echo "All experiments queued. Monitoring progress..."
echo ""

completed=0
total=${#EXPERIMENT_QUEUE[@]}

while [[ $completed -lt $total ]]; do
    for i in "${!pids[@]}"; do
        if [[ -z "${pids[$i]}" ]]; then
            continue
        fi
        
        if ! kill -0 ${pids[$i]} 2>/dev/null; then
            wait ${pids[$i]}
            completed=$((completed + 1))
            echo "Progress: $completed/$total experiments completed"
            unset pids[$i]
        fi
    done
    
    if [[ $completed -lt $total ]]; then
        sleep 5
    fi
done

echo ""
echo "==============================================================="
print_success "All experiments completed!"
echo "==============================================================="
echo ""
echo "To compare results, run:"
echo "  python compare_experiments.py"