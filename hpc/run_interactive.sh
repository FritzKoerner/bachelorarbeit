#!/bin/bash
# ==============================================================================
# Request an interactive GPU session on HPC Leipzig
#
# Usage:
#   ./run_interactive.sh                    # default: paula, a30, 1h
#   ./run_interactive.sh paula a30 2        # partition, gpu_type, hours
#   ./run_interactive.sh clara v100 1
# ==============================================================================

PARTITION="${1:-paula}"
GPU_TYPE="${2:-a30}"
HOURS="${3:-1}"

echo "=== Requesting interactive session ==="
echo "Partition: $PARTITION"
echo "GPU:       $GPU_TYPE"
echo "Time:      ${HOURS}h"
echo ""
echo "After allocation, run:"
echo "  source ~/genesis/hpc/setup_env.sh --load"
echo "  cd ~/genesis/prototyp_global_coordinate"
echo "  python train_rl.py -B 4 --max_iterations 5"
echo ""

salloc \
    --partition="$PARTITION" \
    --gpus="${GPU_TYPE}:1" \
    --mem=32G \
    --cpus-per-task=8 \
    --time="${HOURS}:00:00"
