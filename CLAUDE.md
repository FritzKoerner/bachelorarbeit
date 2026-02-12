# Drone Landing in Vineyards - RL Pipeline

## Project Overview

Bachelor thesis project: Training a drone to autonomously land in vineyard soil strips using reinforcement learning with the Genesis physics simulator (`genesis-world`).

The drone learns to descend from 6-10m altitude, identify safe landing zones (soil strips between vine rows) via a downward-facing depth+segmentation camera, and land softly while avoiding vine rows.

## Genesis Plugin

**Always use the `genesis-world` plugin** (installed at `~/.claude/plugins/genesis-world/`) as the primary API reference when writing or modifying Genesis simulation code. Consult its skill files (`SKILL.md`, `references/`) for correct API usage, patterns, and conventions before making changes.

Key Genesis conventions:
- Coordinates: right-handed, Z-up
- Quaternions: w-x-y-z format
- Euler angles: degrees, scipy extrinsic x-y-z
- All state tensors: `(n_envs, ...)` shape on `gs.device`
- `drone.set_propellels_rpm(rpms)` - note the typo in Genesis API (propellels)
- `scene.build()` triggers JIT compilation; must be called before stepping

## Architecture

```
train.py              - Main training loop (PPO rollout collection + update)
evaluate.py           - Evaluation & trajectory visualization
config/
  training_config.yaml - All hyperparameters
envs/
  vineyard_landing_env.py - Gym-like env wrapping Genesis (obs: depth+seg+state, act: vel cmds)
  scene_builder.py        - Genesis scene: ground plane + soil/vineyard strips + drone + camera
controllers/
  low_level_controller.py - PD velocity→RPM controller
networks/
  actor_critic.py         - Actor-Critic with vision+state fusion
  cnn_encoder.py          - CNN for depth+segmentation images (2ch, 64x64)
algorithms/
  ppo.py                  - PPO trainer with GAE, mini-batch updates, TensorBoard
utils/
  reward_functions.py     - Multi-component reward (descent, velocity, semantic, landing, crash, smoothness)
assets/
  robots/draugas/         - Custom drone URDF + meshes
  scene/vineyard-eltville-germany/ - Vineyard OBJ mesh + textures (not yet integrated)
prototyp/                 - Early prototype (scene_builder, train)
```

## Observation & Action Space

- **Visual obs**: `(2, 64, 64)` — depth (normalized 0-20m) + semantic segmentation (0=background, 1=soil, 2=vineyard)
- **State obs**: `(13,)` — pos(3) + vel(3) + quat(4) + ang_vel(3)
- **Actions**: `(4,)` — [vx, vy, vz, yaw_rate] in [-1, 1], scaled to m/s and rad/s

## Semantic Labels

| Label | Meaning | Color | Landing |
|-------|---------|-------|---------|
| 0 | Background/ground | Gray | Neutral |
| 1 | Soil strip | Brown | Safe |
| 2 | Vineyard strip | Green | Unsafe |

## Key Parameters

- Drone: custom "draugas" URDF, base hover RPM ~1475.8, max RPM 25000
- Scene: 5 alternating strips (soil/vineyard), 2m wide, 50m long
- Spawn: random XY within range, height 6-10m, random yaw
- Landing: height < 0.3m, speed < 0.5 m/s, over soil
- Crash: height < 0.3m AND (speed > 2.0 m/s OR tilt > 0.5 rad)

## Forbidden Actions

- **NEVER sync with the HPC cluster of Uni Leipzig** or execute `sync_with_hpc.sh`. This script and any rsync/scp/ssh operations targeting the HPC cluster are strictly off-limits. The user manages HPC sync manually.

## Dependencies

```
torch>=2.0.0, numpy>=1.24.0, scipy, genesis-world, pyyaml, tensorboard, matplotlib
```
