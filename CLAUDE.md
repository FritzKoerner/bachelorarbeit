# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bachelor thesis project: Training a drone to autonomously land in vineyard soil strips using reinforcement learning with the Genesis physics simulator (`genesis-world` v0.3.13).

## Repository Structure

- **prototyp_global_coordinate/** — **Active development.** Global coordinate-based landing with cascading PID controller. Supports PPO (rsl-rl) and SAC (Stable-Baselines3). All new work happens here.
- **prototyp_2/** — Reference only. Earlier velocity-command PPO pipeline.
- **prototyp_1/** — Reference only. Legacy initial prototype.

Each prototype is self-contained with its own `config/`, `envs/`, `controllers/`, etc.

## Running (prototyp_global_coordinate)

```bash
cd prototyp_global_coordinate

# PPO training (rsl-rl, headless, 4096 envs)
python train_rl.py -B 4096 --max_iterations 301

# PPO smoke test with viewer
python train_rl.py -B 4 -v --max_iterations 5

# SAC training (stable-baselines3, 16 envs)
python train_rl_ac.py -B 16 --total_steps 1_000_000

# PID test (no RL, manual target tracking)
python train.py

# Evaluation
python eval_rl.py      # PPO
python eval_rl_ac.py   # SAC
```

## Dependencies

Conda environment: `ba`. Key packages: `genesis-world==0.3.13`, `torch>=2.0.0`, `numpy`, `scipy`, `pyyaml`, `tensorboard`, `rsl-rl-lib==2.2.4`, `stable-baselines3`.

## Genesis API Quirks (v0.3.13)

- **Two-phase construction**: call `gs.init(backend=gs.gpu)` once, then `env.build()` — the env constructor does NOT build the scene
- **No `camera.start()`/`stop()`** — call `camera.render()` directly
- `camera.render(depth=True, segmentation=True)` returns a **tuple** `(rgb, depth, segmentation, normal)`, not a dict
- **Typo in API**: `drone.set_propellels_rpm(rpms)` (propellels, not propellers)
- `get_link("base")` for URDF root link reference
- Custom URDF drones need explicit `propellers_link_name` and `propellers_spin`
- `scene.build()` triggers JIT compilation; must be called before stepping
- Coordinates: right-handed, Z-up
- Quaternions: w-x-y-z format
- Euler angles: degrees, scipy extrinsic x-y-z
- All state tensors shaped `(n_envs, ...)` on `gs.device`

## Architecture

The RL agent outputs a target position; a cascading PID controller tracks it and produces motor RPMs.

```
prototyp_global_coordinate/
  train_rl.py          - PPO training via rsl-rl OnPolicyRunner
  train_rl_ac.py       - SAC training via Stable-Baselines3
  eval_rl.py           - PPO evaluation
  eval_rl_ac.py        - SAC evaluation
  train.py             - PID-only test (no RL)
  envs/
    coordinate_landing_env.py - rsl-rl compatible env (reset/step/reward)
  controllers/
    pid_controller.py         - CascadingPIDController (position→velocity→attitude→RPM)
    velocity_controller.py    - Legacy velocity-based controller (unused)
  utility/
    optimize_pid.py           - Optuna-based PID parameter optimization
  config/
    training_config.yaml      - PID gains and scene parameters
  assets/robots/draugas/      - Custom drone URDF + meshes
```

**Cascading PID** (three nested loops):
1. **Position PID** → desired velocity
2. **Velocity PID** (X/Y → desired roll/pitch, Z → thrust)
3. **Attitude PID** → corrections fed into motor mixer

## Observation & Action Spaces

- **Observation**: `(n_envs, 9)` — `rel_pos(3) + lin_vel(3) + ang_vel(3)`, each clipped and scaled
  - Observation scales: `rel_pos * 1/15`, `lin_vel * 1/5`, `ang_vel * 1/π`
  - Additional fields (pos, quat, last_actions) exist in code but are currently commented out
  - Note: `num_obs` in PPO config is set to 13 but `_compute_obs()` currently returns 9 — the config value is stale
- **Actions**: `(n_envs, 4)` float in `[-1, 1]` — `[ax, ay, az, ayaw]`
  - `target_xyz = current_pos + action[:3] * action_scales` (offset from current position)
  - `target_yaw = ayaw * 180.0` (degrees)

## Key Parameters

- Drone: custom "draugas" URDF, base hover RPM 1789.2, max RPM 5000
- Spawn: height 3-10m (SAC) or 10m fixed (PPO), drone offset ±5m
- Target: fixed at (3, 3, 1) during early training (PPO), randomized ±3m (SAC)
- Curriculum: first 15000 steps spawn target within 1m of drone, then full range
- Success: hover within 0.3m of target at <0.3 m/s for 30 consecutive steps (0.3s)
- Crash: height < 0.2m, tilt > 60°, or distance from target > 50m
- Rewards: distance penalty (-5.0), time penalty (-0.5), crash (-100), success (+200)

## Forbidden Actions

- **NEVER sync with the HPC cluster of Uni Leipzig** or execute `sync_with_hpc.sh`. Any rsync/scp/ssh operations targeting the HPC cluster are strictly off-limits. The user manages HPC sync manually.

## Genesis Plugin

Use the `genesis-world` plugin (installed at `~/.claude/plugins/genesis-world/`) as the primary API reference when writing Genesis simulation code. Consult its skill files for correct API usage before making changes.
