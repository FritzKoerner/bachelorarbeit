# ROADMAP — Global Coordinate Landing

## Milestone 1: Core RL Pipeline

### Phase 1: Project Scaffolding & Environment

**Goal:** Working Genesis environment with flat ground, Draugas drone, random target spawning, and state-based observations. Drone can be reset and stepped with velocity commands. No training yet.

**Plans:** 2 plans

**Deliverables:**
- Project structure (config/, envs/, controllers/, assets/)
- Drone URDF linked from prototyp_2
- `CoordinateLandingEnv` — Gym-compatible env wrapping Genesis batched simulation
- Velocity controller (ported from prototyp_2)
- YAML config file
- Smoke test: env can reset, step, and return observations

**Depends on:** Nothing
**Risk:** SB3 VecEnv wrapper compatibility with Genesis batched envs

Plans:
- [ ] 01-01-PLAN.md — Project scaffold: directories, config YAML, assets symlink, VelocityController port
- [ ] 01-02-PLAN.md — CoordinateLandingEnv implementation and smoke test

---

### Phase 2: Reward Function & Termination Logic

**Goal:** Complete reward shaping for coordinate-targeted landing: distance-based guidance, landing precision bonus, crash penalty, smoothness penalty. All termination conditions working.

**Deliverables:**
- Distance-based reward (negative distance to target, or shaped potential)
- Landing bonus scaled by proximity to target
- Crash penalty
- Smoothness penalty
- Termination: successful landing, crash, timeout, out-of-bounds
- Unit tests or manual verification of reward components

**Depends on:** Phase 1

---

### Phase 3: SB3 PPO Integration & Training Pipeline

**Goal:** End-to-end training pipeline using SB3 PPO. Can launch training, see TensorBoard metrics, save checkpoints.

**Deliverables:**
- SB3-compatible VecEnv wrapper for Genesis batched env
- `train.py` using SB3 PPO with MlpPolicy
- TensorBoard logging (rewards, episode length, success rate)
- Checkpoint saving
- YAML config for SB3 hyperparameters
- Verified: training runs without errors for 10k+ steps

**Depends on:** Phase 2

---

### Phase 4: Evaluation & Tuning

**Goal:** Evaluate trained models, visualize trajectories, tune hyperparameters for reliable landing performance.

**Deliverables:**
- `evaluate.py` — load checkpoint, run episodes, report metrics
- Metrics: success rate, mean distance to target, mean episode length
- Trajectory visualization (matplotlib)
- Hyperparameter tuning notes
- At least one trained model that demonstrates coordinate-targeted landing

**Depends on:** Phase 3
