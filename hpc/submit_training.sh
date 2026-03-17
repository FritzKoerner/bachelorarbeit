#!/bin/bash
# ==============================================================================
# Submit a training job to the HPC Leipzig SLURM scheduler
#
# Usage:
#   ./submit_training.sh global_coordinate
#   ./submit_training.sh global_coordinate --batch 4096 --iters 401 --hours 48
#   ./submit_training.sh obstacle_avoidance --partition clara --gpu-type v100 --hours 24 --batch 512
#
# Options:
#   --partition  PARTITION   Cluster partition (default: paula)
#   --gpu-type   TYPE        GPU type (default: a30)
#   --gpus       N           Number of GPUs (default: 1)
#   --hours      H           Time limit in hours (default: 48)
#   --mem        MEM         Memory (default: 64G)
#   --cpus       N           CPUs per task (default: 16)
#   --batch      B           Batch size / num envs (default: 4096)
#   --iters      N           Max iterations (default: 401)
#   --no-wandb               Use train_rl.py instead of train_rl_wb.py
# ==============================================================================

set -e

# --- Defaults ---
PARTITION="paula"
GPU_TYPE="a30"
GPU_COUNT=1
HOURS=48
MEM="64G"
CPUS=16
BATCH=4096
ITERS=401
USE_WANDB=true

# --- Parse prototype name ---
if [ -z "$1" ] || [[ "$1" == --* ]]; then
    echo "Usage: ./submit_training.sh <prototype> [options]"
    echo "  prototype: global_coordinate | obstacle_avoidance"
    exit 1
fi
PROTOTYPE="$1"
shift

# --- Parse options ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)  PARTITION="$2"; shift 2 ;;
        --gpu-type)   GPU_TYPE="$2"; shift 2 ;;
        --gpus)       GPU_COUNT="$2"; shift 2 ;;
        --hours)      HOURS="$2"; shift 2 ;;
        --mem)        MEM="$2"; shift 2 ;;
        --cpus)       CPUS="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --iters)      ITERS="$2"; shift 2 ;;
        --no-wandb)   USE_WANDB=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Resolve paths ---
GENESIS_DIR="$HOME/genesis"
PROTO_DIR="${GENESIS_DIR}/prototyp_${PROTOTYPE}"
LOG_DIR="${GENESIS_DIR}/logs"

# Pick training script
TRAIN_SCRIPT="train_rl.py"
if [ "$USE_WANDB" = true ] && [ -f "${PROTO_DIR}/train_rl_wb.py" ]; then
    TRAIN_SCRIPT="train_rl_wb.py"
fi

JOB_NAME="genesis-${PROTOTYPE}"

echo "=== Submitting training job ==="
echo "Prototype:  prototyp_${PROTOTYPE}"
echo "Script:     ${TRAIN_SCRIPT}"
echo "Partition:  ${PARTITION} (${GPU_TYPE} x${GPU_COUNT})"
echo "Time:       ${HOURS}h, ${MEM} RAM, ${CPUS} CPUs"
echo "Training:   -B ${BATCH} --max_iterations ${ITERS}"

# --- Ensure log directory exists on HPC ---
ssh hpc "mkdir -p ${LOG_DIR}"

# --- Generate and submit job script ---
JOBSCRIPT=$(mktemp /tmp/genesis_job_XXXXXX.sh)

cat > "$JOBSCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${GPU_TYPE}:${GPU_COUNT}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${HOURS}:00:00
#SBATCH --output=${LOG_DIR}/slurm-%j.out
#SBATCH --error=${LOG_DIR}/slurm-%j.err
#SBATCH --mail-type=END,FAIL

# --- Environment ---
module purge
module load Anaconda3
module load CUDA/12.6.0
eval "\$(conda shell.bash hook)"
conda activate ba

# --- Training ---
cd ${PROTO_DIR}
echo "=== Starting training: \$(date) ==="
echo "Host: \$(hostname), GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader)"

python ${TRAIN_SCRIPT} -B ${BATCH} --max_iterations ${ITERS}

echo "=== Training complete: \$(date) ==="
SLURM_EOF

# Copy job script to HPC and submit
scp "$JOBSCRIPT" "hpc:${LOG_DIR}/${JOB_NAME}.sh"
ssh hpc "sbatch ${LOG_DIR}/${JOB_NAME}.sh"

rm "$JOBSCRIPT"
echo "=== Job submitted. Monitor with: ssh hpc squeue -u fk67rahe ==="
