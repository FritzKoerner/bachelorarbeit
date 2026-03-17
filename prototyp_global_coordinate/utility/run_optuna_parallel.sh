#!/usr/bin/env bash
# Parallel Optuna PID optimization launcher
# Usage: ./run_optuna_parallel.sh

# ── Adjustable variables ──────────────────────────
N_WORKERS=10          # number of parallel processes
N_TRIALS=100          # trials per worker (total = N_WORKERS * N_TRIALS)
N_STEPS=500           # simulation steps per trial evaluation
DB="sqlite:///optuna_pid.db"
# ──────────────────────────────────────────────────

echo "Launching $N_WORKERS workers × $N_TRIALS trials = $((N_WORKERS * N_TRIALS)) total trials"

pids=()
for i in $(seq 1 "$N_WORKERS"); do
    python optimize_pid.py \
        --n_trials "$N_TRIALS" \
        --n_steps "$N_STEPS" \
        --db "$DB" \
        > "logs/worker_${i}.log" 2>&1 &
    pids+=($!)
    echo "  Started worker $i (PID ${pids[-1]})"
done

echo "All workers launched. Waiting for completion..."
echo "  Logs: logs/worker_*.log"

# Wait for all workers and track failures
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "  Worker $((i + 1)) (PID ${pids[$i]}) FAILED"
        ((failed++))
    fi
done

if [ "$failed" -eq 0 ]; then
    echo "All $N_WORKERS workers completed successfully."
else
    echo "$failed/$N_WORKERS workers failed. Check logs for details."
fi

# Print best result
python optimize_pid.py --show_best --db "$DB"
