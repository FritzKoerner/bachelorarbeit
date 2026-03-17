#!/bin/bash
# ==============================================================================
# Sync local code to HPC Leipzig cluster
#
# Usage:
#   ./sync_to_hpc.sh              # sync everything
#   ./sync_to_hpc.sh --dry-run    # preview what would be synced
# ==============================================================================

set -e

LOCAL_DIR="/home/fritz-sfl/Bachelorarbeit/genesis/"
REMOTE_DIR="hpc:/home/sc.uni-leipzig.de/fk67rahe/genesis/"

EXTRA_ARGS=""
if [ "$1" = "--dry-run" ]; then
    EXTRA_ARGS="--dry-run"
    echo "=== DRY RUN — no files will be transferred ==="
fi

rsync -ahz --partial --info=progress2 $EXTRA_ARGS \
    --exclude='.git' \
    --exclude='wandb/' \
    --exclude='logs/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.firecrawl/' \
    --exclude='.claude/' \
    --exclude='artifacts/' \
    --exclude='eval_data/' \
    --exclude='hpc_results/' \
    --exclude='.vscode/' \
    "$LOCAL_DIR" "$REMOTE_DIR"

echo "=== Sync complete: local -> HPC ==="
