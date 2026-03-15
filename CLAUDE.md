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

# Visualization
python visualize_paths.py --ckpt 100 300                  # matplotlib + screenshots
python visualize_paths.py --ckpt 300 --video              # + landing GIF with trail
python visualize_paths.py --ckpt 100 300 --no_render      # matplotlib only
```

## Dependencies

Conda environment: `ba`. Key packages: `genesis-world==0.3.13`, `torch>=2.0.0`, `numpy`, `scipy`, `pyyaml`, `tensorboard`, `rsl-rl-lib==2.2.4`, `stable-baselines3`. Note: `moviepy`, `cv2`, and `ffmpeg` are NOT installed — video output falls back to PIL GIF.

## Genesis API Quirks (v0.3.13)

- **`scene.add_camera()` blocked post-build** — bypass via `scene._visualizer.add_camera(res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise, near, far, env_idx, debug)` + `cam.build()`
- **Debug drawing**: `scene.visualizer.context` has `draw_debug_line`, `draw_debug_sphere`, `draw_debug_mesh`, `clear_debug_objects()`. Rasterizer only (not RayTracer).
- **`gs.UID()` short format is 7 hex chars** — collides via birthday problem at ~15k nodes. Combine line segments into one trimesh via `trimesh.util.concatenate()` and use `draw_debug_mesh`.
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
- **Auto-reset contaminates state**: when `step()` returns `done=True`, the env has already reset — `base_pos` reflects the new spawn, not the terminal position. Record positions BEFORE `step()` or break before reading post-done state.

## Architecture

The RL agent outputs a target position; a cascading PID controller tracks it and produces motor RPMs.

```
prototyp_global_coordinate/
  train_rl.py          - PPO training via rsl-rl OnPolicyRunner
  train_rl_ac.py       - SAC training via Stable-Baselines3
  eval_rl.py           - PPO evaluation
  eval_rl_ac.py        - SAC evaluation
  train.py             - PID-only test (no RL)
  visualize_paths.py   - Flight path visualization, Genesis screenshots, and landing GIFs per checkpoint
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

## rsl-rl Gotchas

- `OnPolicyRunner.__init__` calls `.pop()` on config dicts, mutating them. Always pass `copy.deepcopy(train_cfg)` when creating multiple runners.
- `get_inference_policy()` returns a deterministic MLP wrapper — batch-size agnostic, works with any `num_envs`.

## rsl-rl Versions

- **Installed (`rsl-rl-lib==2.2.4`)**: Monolithic `ActorCritic` class, flat tensor obs, `eval()` class resolution, `"policy": {}` config format. Currently used by `train_rl.py`.
- **Development (`~/Repos/rsl_rl`)**: Separate `MLPModel`/`CNNModel` per actor+critic, `TensorDict` obs, `resolve_callable()`, `"actor":{}`+`"critic":{}` config, built-in CNN sharing via `share_cnn_encoders`. APIs are **incompatible** — env and config must be rewritten when upgrading.
- CNN ActorCritic implementation guides: `prototyp_global_coordinate/docs/custom_cnn_actor_critic_guide.md` (v2.2.4) and `custom_cnn_actor_critic_guide_new_rsl_rl.md` (new version).

## Observation & Action Spaces

- **Observation**: `(n_envs, 17)` — `rel_pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + last_actions(4)`, each clipped and scaled
  - Observation scales: `rel_pos * 1/15`, `lin_vel * 1/5`, `ang_vel * 1/π`
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

## Plugins

- **`genesis-world`** plugin (`~/.claude/plugins/genesis-world/`) — Genesis physics simulator API reference. Consult when writing simulation code.
- **`rsl-rl`** plugin (`~/.claude/plugins/rsl-rl/`) — rsl-rl reinforcement learning library API reference. Consult when working with PPO training, OnPolicyRunner, actor-critic models, rollout storage, or any rsl-rl integration code.
