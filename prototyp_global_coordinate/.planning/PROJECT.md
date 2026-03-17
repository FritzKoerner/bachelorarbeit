# PROJECT — Global Coordinate Landing

## What This Is

A reinforcement learning pipeline that trains a quadrotor drone (Draugas) to navigate and land at a randomly specified (x, y) ground coordinate in the Genesis simulator. The closer to the target, the better the reward. Uses Stable-Baselines3 PPO with a state-based observation space (no camera).

## Core Value

Teach a drone precise coordinate-targeted landing via RL — the foundational navigation capability for autonomous drone operations.

## Reference

- **prototyp_2**: Sister prototype in `../prototyp_2/`. Trains the same Draugas drone to land on vineyard soil strips using custom PPO with visual observations. This prototype reuses the drone URDF, velocity controller, and Genesis setup patterns from prototyp_2 but differs in: SB3 instead of custom PPO, state-only observations, flat ground environment, coordinate-based reward.

## Requirements

### Validated

1. **Flat ground environment** — simple ground plane, no vineyard scene
2. **Random target (x, y)** — sampled within ±30m x/y each episode
3. **Random drone spawn** — random (x, y) offset from target, 10-20m altitude
4. **State-only observations** — relative position to target, drone velocity, orientation, angular velocity (no camera)
5. **Velocity command action space** — (vx, vy, vz, yaw_rate) with PD controller to RPMs, same as prototyp_2
6. **Distance-based reward** — closer to target = higher reward, with landing bonus and crash penalty
7. **Stable-Baselines3 PPO** — standard library implementation
8. **16 parallel environments** — batched Genesis simulation (same as prototyp_2)
9. **TensorBoard logging** — training metrics visualization
10. **Checkpoint saving** — periodic model saves

### Active

- All above requirements are active for milestone 1.

### Out of Scope

- Camera/visual observations (future milestone)
- Vineyard scene integration
- Real-world deployment
- Multi-agent training
- Dynamic obstacles

## Key Decisions

| # | Decision | Rationale | Date |
|---|----------|-----------|------|
| 1 | SB3 over custom PPO | User preference — well-tested, less code to maintain | 2026-02-13 |
| 2 | State-only (no camera) | Simpler, faster training, focus on navigation precision | 2026-02-13 |
| 3 | Flat ground | Isolate coordinate-landing capability from scene complexity | 2026-02-13 |
| 4 | Large area (±30m, 10-20m alt) | User preference — more challenging navigation | 2026-02-13 |
| 5 | Random drone offset | Drone must navigate laterally + descend, not just descend | 2026-02-13 |
| 6 | Keep velocity commands + PD controller | Proven action space from prototyp_2 | 2026-02-13 |

## Constraints

- **Genesis simulator** — must use genesis-world for physics
- **Draugas drone URDF** — reuse from prototyp_2 assets
- **GPU training** — PyTorch + CUDA
- **Python 3.13** — match prototyp_2 environment
- **Batched tensor operations** — all env logic on GPU tensors for parallel envs
