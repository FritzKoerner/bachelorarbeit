#!/bin/bash
# ==============================================================================
# Sync training results from HPC Leipzig cluster back to local machine
#
# Usage:
#   ./sync_from_hpc.sh                          # sync all prototype logs
#   ./sync_from_hpc.sh global_coordinate        # sync specific prototype
#   ./sync_from_hpc.sh obstacle_avoidance       # sync specific prototype
#   ./sync_from_hpc.sh --dry-run                # preview only
# ==============================================================================

set -e

LOCAL_DIR="/home/fritz-sfl/Bachelorarbeit/genesis/hpc_results/"
REMOTE_BASE="hpc:/home/sc.uni-leipzig.de/fk67rahe/genesis"

PROTOTYPE=""
DRY_RUN=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run"; echo "=== DRY RUN ===" ;;
        *) PROTOTYPE="$arg" ;;
    esac
done

mkdir -p "$LOCAL_DIR"

if [ -n "$PROTOTYPE" ]; then
    echo "=== Syncing logs from prototyp_${PROTOTYPE} ==="
    mkdir -p "${LOCAL_DIR}prototyp_${PROTOTYPE}/"
    rsync -ahz --partial --info=progress2 $DRY_RUN \
        "${REMOTE_BASE}/prototyp_${PROTOTYPE}/logs/" \
        "${LOCAL_DIR}prototyp_${PROTOTYPE}/"
else
    echo "=== Syncing logs from all prototypes ==="
    for proto in global_coordinate obstacle_avoidance; do
        echo "--- prototyp_${proto} ---"
        mkdir -p "${LOCAL_DIR}prototyp_${proto}/"
        rsync -ahz --partial --info=progress2 $DRY_RUN \
            "${REMOTE_BASE}/prototyp_${proto}/logs/" \
            "${LOCAL_DIR}prototyp_${proto}/" 2>/dev/null || echo "  (no logs yet)"
    done
fi

# Also sync SLURM output logs
echo "--- SLURM logs ---"
mkdir -p "${LOCAL_DIR}slurm/"
rsync -ahz --partial --info=progress2 $DRY_RUN \
    "${REMOTE_BASE}/logs/slurm-*.out" \
    "${REMOTE_BASE}/logs/slurm-*.err" \
    "${LOCAL_DIR}slurm/" 2>/dev/null || echo "  (no SLURM logs yet)"

echo "=== Sync complete: HPC -> local ($LOCAL_DIR) ==="
