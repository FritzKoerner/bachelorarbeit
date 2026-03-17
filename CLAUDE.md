# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bachelor thesis project: Training a drone to autonomously land in vineyard soil strips using reinforcement learning with the Genesis physics simulator (`genesis-world` v0.3.13).

## Repository Structure

- **prototyp_global_coordinate/** — **Active development.** Global coordinate-based landing with cascading PID controller. Supports PPO (rsl-rl) and SAC (Stable-Baselines3). All new work happens here.
- **prototyp_obstacle_avoidance/** — **Active development.** CNN depth-map obstacle avoidance prototype. Extends `prototyp_global_coordinate` with random obstacles, downward-facing depth camera, and rsl-rl v5.0.1 `CNNModel` + `share_cnn_encoders`.
- **prototyp_2/** — Reference only. Earlier velocity-command PPO pipeline.
- **prototyp_1/** — Reference only. Legacy initial prototype.

Each prototype is self-contained with its own `config/`, `envs/`, `controllers/`, etc.

## Running (prototyp_global_coordinate)

```bash
cd prototyp_global_coordinate

# PPO training (rsl-rl, headless, 4096 envs)
python train_rl.py -B 4096 --max_iterations 401

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

Conda environment: `ba`. Key packages: `genesis-world==0.3.13`, `torch>=2.0.0`, `numpy`, `scipy`, `pyyaml`, `tensorboard`, `rsl-rl-lib==5.0.1`, `stable-baselines3`, `tensordict`. Note: `moviepy`, `cv2`, and `ffmpeg` are NOT installed — video output falls back to PIL GIF.

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
- **Batched camera tracking**: `trans_quat_to_T(pos, quat)` from `genesis.utils.geom` converts to `(n_envs, 4, 4)` transforms. Multiply by a fixed offset matrix for attached cameras: `camera.set_pose(transform=torch.matmul(link_T, offset_T))`. Downward-facing offset: `Rotation.from_euler('zyx', [-90, -90, 0], degrees=True)` + `T[2,3] = -0.1`.
- **`gs.morphs.Box` with `fixed=True`**: supports per-env `set_pos(new_pos, envs_idx=idx, zero_velocity=True)` for obstacle randomization. Use `collision=False` when handling collisions via distance checks.

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
    training_config.yaml      - DEPRECATED, not used. All config is in train_rl.py
  assets/robots/draugas/      - Custom drone URDF + meshes
```

**Cascading PID** (three nested loops):
1. **Position PID** → desired velocity
2. **Velocity PID** (X/Y → desired roll/pitch, Z → thrust)
3. **Attitude PID** → corrections fed into motor mixer

PID features: derivative-on-measurement (no kick), EMA-filtered derivative (alpha=0.2), integral anti-windup (conditional integration), velocity clamping (5 m/s XY, 3 m/s Z).

## rsl-rl Gotchas

- `OnPolicyRunner.__init__` calls `.pop()` on config dicts, mutating them. Always pass `copy.deepcopy(train_cfg)` when creating multiple runners.
- `get_inference_policy()` returns the actor `MLPModel` directly — it accepts `TensorDict` input, not flat tensors.
- The runner accesses `cfg["algorithm"]["rnd_cfg"]` unconditionally — always include `"rnd_cfg": None` in algorithm config even when not using RND.
- **v5.0.1 uses default Kaiming init**, not orthogonal. For v2.2.4-like behavior, call `actor.mlp.init_weights()` / `critic.mlp.init_weights()` after runner construction.
- **GaussianDistribution defaults to `std_type="scalar"`** which can go negative. Use `"std_type": "log"` for stability, especially with low/zero `entropy_coef`.
- **`entropy_coef` must be small** relative to per-step rewards (which are dt-scaled to ~0.05/step). Values like 0.01 cause std explosion with log-space; 0.001 works.
- **`share_cnn_encoders: True`** goes in `"algorithm"` config, NOT in actor/critic. Critic config should omit `cnn_cfg` entirely — PPO injects `actor.cnns` automatically.
- **`get_inference_policy()`** returns the actor model directly — works with both `MLPModel` (flat TensorDict) and `CNNModel` (multi-key TensorDict including image obs).

## rsl-rl Version

- **Installed (`rsl-rl-lib==5.0.1`)**: Separate `MLPModel`/`CNNModel` per actor+critic, `TensorDict` obs, `resolve_callable()`, `"actor":{}`+`"critic":{}` config, built-in CNN sharing via `share_cnn_encoders`. Environment must return `TensorDict` from `get_observations()` and `step()`.
- CNN ActorCritic implementation guide: `prototyp_global_coordinate/docs/custom_cnn_actor_critic_guide_new_rsl_rl.md`.

## Observation & Action Spaces

- **Observation**: `(n_envs, 17)` — `rel_pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + last_actions(4)`, each clipped and scaled
  - Observation scales: `rel_pos * 1/15`, `lin_vel * 1/5`, `ang_vel * 1/π`
- **Actions**: `(n_envs, 4)` float in `[-1, 1]` — `[ax, ay, az, ayaw]`
  - `target_xyz = current_pos + action[:3] * action_scales` (offset from current position)
  - `target_yaw = ayaw * 180.0` (degrees)

## Key Parameters

- Drone: custom "draugas" URDF (mass 0.714kg, thrust2weight 2.25), base hover RPM 1789.2, max RPM 2700
- Spawn: height 3-10m (SAC) or 10m fixed (PPO), drone offset ±5m
- Target: fixed at (3, 3, 1) during early training (PPO), randomized ±3m (SAC)
- Curriculum: first 20000 steps spawn target within 1m of drone, then full range
- Success: hover within 0.3m of target at <0.3 m/s for 30 consecutive steps (0.3s)
- Crash: height < 0.2m, tilt > 60°, or distance from target > 50m
- Rewards: distance penalty (-5.0), time penalty (-0.5), crash (-100), success (+200)

## Running (prototyp_obstacle_avoidance)

```bash
cd prototyp_obstacle_avoidance

# PPO training (headless, 64 envs — depth rollouts use more memory)
python train_rl.py -B 64 --max_iterations 401

# Smoke test with viewer
python train_rl.py -B 4 -v --max_iterations 5

# Evaluation
python eval_rl.py --ckpt 300
python eval_rl.py --ckpt 300 --vis

# Visualization
python visualize_paths.py --ckpt 100 300 --no_render
python visualize_paths.py --ckpt 300 --video
```

## prototyp_obstacle_avoidance Architecture

```
prototyp_obstacle_avoidance/
  train_rl.py          - PPO training with CNNModel (shared CNN encoder)
  eval_rl.py           - Evaluation with obstacle collision tracking
  visualize_paths.py   - Flight paths with obstacle rendering
  envs/
    obstacle_avoidance_env.py - Env: obstacles + depth camera + TensorDict
  controllers/
    pid_controller.py         - CascadingPIDController (copied from prototyp_global_coordinate)
  assets/robots/draugas/      - Symlink → prototyp_global_coordinate
```

**Observations**: `TensorDict({"state": (n, 17), "depth": (n, 1, 64, 64)})`. State is identical 17-dim vector; depth is downward-facing, normalized `clamp(depth / 20, 0, 1)`.

**Model**: `CNNModel` (3 conv layers: [32,64,128], kernels [8,4,3], strides [4,2,1], batch norm, global avg pool → 128-dim latent) + MLP [256,256]. CNN shared between actor and critic via `share_cnn_encoders`.

**Obstacles**: 5 random `gs.morphs.Box(1x1x2m)`, distance-based collision (radius 0.8m), obstacle proximity penalty within safety radius (3.0m).

**Rewards**: distance (-5), time (-0.5), obstacle_proximity (-10), crash (-100), obstacle_collision (-150), success (+200). Per-step rewards scaled by dt.

## Forbidden Actions

- **NEVER sync with the HPC cluster of Uni Leipzig** or execute `sync_with_hpc.sh`. Any rsync/scp/ssh operations targeting the HPC cluster are strictly off-limits. The user manages HPC sync manually.

## Plugins

- **`genesis-world`** plugin (`~/.claude/plugins/genesis-world/`) — Genesis physics simulator API reference. Consult when writing simulation code.
- **`rsl-rl`** plugin (`~/.claude/plugins/rsl-rl/`) — rsl-rl reinforcement learning library API reference. Consult when working with PPO training, OnPolicyRunner, actor-critic models, rollout storage, or any rsl-rl integration code.
