# REQUIREMENTS — Global Coordinate Landing

## Milestone 1: Core RL Pipeline

### R1: Genesis Environment (Flat Ground + Drone)

**Priority:** Must-have

- Flat ground plane scene in Genesis
- Draugas drone loaded from URDF (reuse from prototyp_2 assets)
- 16 parallel environments (batched on GPU)
- Random target (x, y) coordinate sampled within ±30m each episode
- Random drone spawn position: random (x, y) offset, 10-20m altitude
- Target visualized as a marker in the scene (optional but helpful for debugging)

### R2: State-Based Observation Space

**Priority:** Must-have

- Observation vector containing:
  - Relative position to target (dx, dy, dz) — 3 values
  - Drone velocity (vx, vy, vz) — 3 values
  - Drone orientation (quaternion) — 4 values
  - Drone angular velocity (wx, wy, wz) — 3 values
- Total: 13-dimensional Box observation space
- All values as float32 tensors on GPU

### R3: Velocity Command Action Space

**Priority:** Must-have

- 4D continuous action: (vx, vy, vz, yaw_rate)
- Clipped to [-1, 1], scaled to physical ranges
- PD velocity controller converts to 4 propeller RPMs
- Reuse VelocityController logic from prototyp_2

### R4: Distance-Based Reward Function

**Priority:** Must-have

- Primary: negative distance to target (encourage approaching)
- Landing bonus: large positive reward when landed close to target (height < threshold, speed < threshold)
- Crash penalty: large negative reward for hard impact or excessive tilt
- Smoothness penalty: penalize jerky control inputs
- Landing precision bonus: scale landing reward inversely with distance to target

### R5: Episode Termination

**Priority:** Must-have

- Successful landing: height < 0.3m AND speed < threshold AND near target
- Crash: height < 0.3m AND (speed > crash threshold OR tilt > tilt threshold)
- Timeout: max steps per episode (configurable, ~500)
- Out of bounds: exceeds spatial limits

### R6: SB3 PPO Integration

**Priority:** Must-have

- Wrap Genesis batched env as SB3-compatible VecEnv
- SB3 PPO with MLP policy (no CNN needed — state only)
- Configurable hyperparameters via YAML config
- GPU training support

### R7: Training Pipeline

**Priority:** Must-have

- `train.py` entry point
- YAML configuration file for all hyperparameters
- TensorBoard logging (rewards, episode lengths, losses)
- Periodic checkpoint saving
- Run directory with timestamp (`runs/<timestamp>/`)
- Config snapshot saved with each run

### R8: Evaluation

**Priority:** Should-have

- `evaluate.py` script to load checkpoint and run episodes
- Metrics: landing success rate, mean distance to target, mean episode length
- Optional trajectory visualization (matplotlib)

### R9: Code Organization

**Priority:** Should-have

- Modular structure mirroring prototyp_2:
  - `envs/` — environment definition
  - `controllers/` — velocity controller
  - `config/` — YAML configs
  - `assets/` — symlinked or copied drone URDF
  - `runs/` — training outputs
