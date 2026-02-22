# Architecture

**Analysis Date:** 2026-02-12

## Pattern Overview

**Overall:** Actor-Critic PPO with Simulation Environment Wrapper

The codebase implements a reinforcement learning pipeline for training a quadrotor drone to land autonomously in vineyard soil strips using the Genesis physics simulator. The architecture separates concerns into three main layers:
1. **Simulation Layer** - Physics simulation with Genesis, scene building, camera rendering
2. **Control Layer** - Environment interface, low-level velocity controller
3. **Learning Layer** - Actor-Critic policy, PPO trainer, reward computation

**Key Characteristics:**
- Vectorized parallel environments (N instances running simultaneously)
- Multi-modal observations: visual (depth + semantic segmentation) + state (position, velocity, orientation)
- Two-stage control: high-level actions (velocity commands) → low-level controller (propeller RPM)
- GPU-accelerated tensors throughout for efficient batch processing
- Reward-driven learning with multi-component reward function

## Layers

**Simulation (Physics):**
- Purpose: Simulate drone dynamics, camera sensors, and environment
- Location: `envs/scene_builder.py`, `envs/vineyard_landing_env.py`
- Contains: Genesis scene construction, drone spawning, strip creation, camera attachment
- Depends on: Genesis (`genesis-world` plugin), numpy, scipy (rotation matrices)
- Used by: Environment wrapper

**Environment (Gym-like Interface):**
- Purpose: Provide standard RL interface (reset/step) with observation/action/reward abstraction
- Location: `envs/vineyard_landing_env.py`
- Contains: Observation composition (visual + state), action scaling, episode termination logic
- Depends on: Simulation layer, low-level controller, reward function
- Used by: PPO trainer, evaluation script

**Control (Velocity→RPM):**
- Purpose: Convert high-level velocity commands to motor RPM for realistic drone control
- Location: `controllers/low_level_controller.py`
- Contains: PD velocity control, attitude control, motor mixing for quadrotor
- Depends on: torch for batch computation
- Used by: Environment step function

**Policy (Actor-Critic Networks):**
- Purpose: Learn mapping from observations to actions
- Location: `networks/actor_critic.py`, `networks/cnn_encoder.py`
- Contains: Vision encoder (CNN for depth+semantic), state encoder, shared MLP, actor/critic heads
- Depends on: PyTorch, vision encoder
- Used by: PPO trainer, evaluation

**Training (PPO):**
- Purpose: Collect rollouts, compute advantages (GAE), perform policy/value updates
- Location: `algorithms/ppo.py`
- Contains: Mini-batch training loop, GAE computation, loss functions (policy+value+entropy), checkpointing
- Depends on: Policy network, torch optimizer
- Used by: Main training script

**Reward:**
- Purpose: Compute multi-component reward signal for task learning
- Location: `utils/reward_functions.py`
- Contains: Descent bonus, velocity smoothness, semantic segmentation rewards, landing bonus, crash penalty
- Depends on: torch, semantic segmentation from environment
- Used by: Environment step function

## Data Flow

**Training Loop:**

1. **Initialization** (`train.py:136-154`)
   - Initialize Genesis backend (GPU/CPU)
   - Create VineyardLandingEnv with N parallel environments
   - Create ActorCritic policy network
   - Create PPOTrainer with config

2. **Rollout Collection** (`train.py:25-127`, `vineyard_landing_env.py`)
   - Reset environments: random spawn positions (6-10m altitude), random yaw
   - For each step:
     - Get current observation: depth image + semantic mask + state vector
     - Policy forward pass: visual + state → action distribution
     - Sample action, compute log_prob and value estimate
     - Scale action: [-1, 1] → velocity commands (m/s) + yaw rate
     - VelocityController: convert velocity target to propeller RPM
     - Genesis step: apply RPM, simulate physics
     - Compute reward: multi-component (descent, velocity, semantic, landing, crash, smoothness)
     - Check termination: landing, crash, timeout, out-of-bounds
     - Auto-reset done environments (partial reset with env_ids)
   - Accumulate: observations, actions, rewards, log_probs, values

3. **Advantage Computation** (`algorithms/ppo.py:53-90`)
   - Compute last value estimate
   - Reverse-pass GAE: δₜ = rₜ + γVₜ₊₁ - Vₜ
   - Accumulate advantages with decay: Aₜ = δₜ + γλAₜ₊₁
   - Returns = Advantages + Values

4. **PPO Update** (`algorithms/ppo.py:92-178`)
   - Flatten batch: (n_steps, n_envs, ...) → (batch_size, ...)
   - Normalize advantages
   - For each epoch:
     - Shuffle mini-batch indices
     - For each mini-batch:
       - Forward pass: evaluate actions under current policy
       - Compute losses:
         - Policy loss: clipped surrogate objective
         - Value loss: MSE(value - return)
         - Entropy loss: maximize exploration
       - Backward pass with gradient clipping
       - Adam optimizer step

5. **Logging & Checkpointing** (`train.py:181-219`)
   - Log metrics to TensorBoard every log_interval
   - Save best model checkpoint when reward improves
   - Save periodic checkpoints every save_interval

**Evaluation Flow** (`evaluate.py`):

1. Load config and checkpoint
2. Create environment (single instance for eval)
3. For each episode:
   - Reset: random spawn
   - Until done:
     - Policy deterministic forward pass (no sampling)
     - Step environment
     - Record trajectory (position, velocity, action)
   - Track outcome: LANDED / CRASHED / TIMEOUT / OUT-OF-BOUNDS
4. Visualize trajectories: 3D plots, height/velocity/XY over time

**State Management:**

- **Episode state:** stored in `env.step_count`, `env.prev_pos`, `reward_fn.prev_actions`
- **Network state:** stored in policy weights (optimizer state in trainer)
- **Physics state:** managed entirely by Genesis (position, velocity, orientation for all N environments)
- **Partial resets:** When environments terminate, only those env_ids are reset while others continue

## Key Abstractions

**VineyardLandingEnv (Gym-like Environment):**
- Purpose: Unify Genesis physics, control, and RL interface
- Examples: `envs/vineyard_landing_env.py:14-279`
- Pattern: Standard reset() → obs, step(action) → (obs, reward, done, info) interface
- Vectorized: all methods handle (n_envs, ...) tensor shapes
- Auto-reset: internally resets done environments to support continuous training

**ActorCritic (Policy Network):**
- Purpose: Unified actor-critic network with shared trunk
- Examples: `networks/actor_critic.py:13-173`
- Pattern: Separate encoder branches (vision + state) → fusion → shared MLP → separate heads (actor/critic)
- Methods: `act()` for sampling, `evaluate()` for loss computation, `get_value()` for bootstrapping
- Output: continuous Gaussian actions (action_mean, action_std via learnable log_std)

**VelocityController (Low-Level Control):**
- Purpose: Realistic motor control without direct RPM access from policy
- Examples: `controllers/low_level_controller.py:10-155`
- Pattern: Hierarchical control
  - Layer 1: Velocity PD control → desired acceleration
  - Layer 2: Attitude estimation from acceleration → desired roll/pitch
  - Layer 3: Attitude error → differential RPM
  - Layer 4: Motor mixing (X-configuration) → 4 propeller RPMs
- Handles quaternion rotation representation and motor saturation

**VineyardSceneBuilder (Scene Construction):**
- Purpose: Programmatic scene creation with semantic labels
- Examples: `envs/scene_builder.py:12-172`
- Pattern:
  - Create ground plane (Label 0 - background)
  - Create alternating soil/vineyard strips (Labels 1/2)
  - Attach downward-facing camera to drone
  - Build parallel environments with Genesis
  - Map raw Genesis segmentation IDs to semantic labels
- Semantic mapping: entity_to_label dict tracks which entities → which semantic labels

**PPOTrainer (RL Algorithm):**
- Purpose: Execute PPO training loop with standard hyperparameters
- Examples: `algorithms/ppo.py:13-205`
- Pattern:
  - Stateful trainer object holds policy, optimizer, config, tensorboard writer
  - `compute_gae()`: offline advantage estimation
  - `update()`: mini-batch training with clipping, gradient norm clamping
  - Checkpoint save/load for resuming training

**LandingRewardFunction (Task Definition):**
- Purpose: Task-specific multi-component reward aggregation
- Examples: `utils/reward_functions.py:8-164`
- Pattern: Component-based with configurable weights
  - w_descent: height decrease encouragement
  - w_velocity: smooth descent at target speed
  - w_semantic: favor soil, penalize vineyard (via semantic segmentation)
  - w_landing: bonus for successful landing (low+slow+soil)
  - w_crash: penalty for unsafe contact (low+fast OR tilted)
  - w_smooth: penalize jerky action changes
- Extracts center region from semantic mask for "directly under drone" evaluation

## Entry Points

**Training** (`train.py:130-269`):
- Location: `train.py`
- Triggers: `python train.py --config config/training_config.yaml`
- Responsibilities:
  - Load config and command-line overrides
  - Initialize Genesis and environment
  - Create policy and trainer
  - Main loop: rollout collection → PPO update → logging
  - Save checkpoints

**Evaluation** (`evaluate.py:37-147`):
- Location: `evaluate.py`
- Triggers: `python evaluate.py --checkpoint path/to/model.pt --config config/training_config.yaml`
- Responsibilities:
  - Load trained policy from checkpoint
  - Run N episodes with deterministic policy
  - Collect success metrics (landing rate, crash rate, timeout rate)
  - Optionally visualize trajectories (3D plots, height/velocity over time)

**Environment Reset** (`vineyard_landing_env.py:70-124`):
- Location: `envs/vineyard_landing_env.py:reset()`
- Triggers: Initial `env.reset()` before training, or partial reset for done environments
- Responsibilities:
  - Random spawn: position (x, y, z) and orientation (yaw)
  - Reset drone state: zero velocity and angular velocity
  - Reset reward function and controller state
  - Return initial observation

## Error Handling

**Strategy:** Graceful degradation with torch.clamp and safety checks

**Patterns:**

1. **Velocity Bounds** (`low_level_controller.py:126`)
   ```python
   rpms = torch.clamp(rpms, 0, self.max_rpm)
   ```
   Saturation: motors limited to valid operating range

2. **Acceleration Bounds** (`low_level_controller.py:100`)
   ```python
   az = torch.clamp(desired_acc[:, 2], min=0.1)
   ```
   Prevents division by zero and unphysical negative upward acceleration

3. **Quaternion Stabilization** (`low_level_controller.py:152`)
   ```python
   pitch = torch.asin(torch.clamp(sinp, -1, 1))
   ```
   Clamps arcsin argument to valid domain [-1, 1]

4. **Advantage Normalization** (`ppo.py:120`)
   ```python
   advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
   ```
   Epsilon prevents division by zero during advantage normalization

5. **Environment Termination Cascading** (`vineyard_landing_env.py:188-207`)
   - Check multiple conditions: height, speed, tilt, bounds
   - Cumulative dones prevents re-checking done environments
   - Auto-reset only resets the done_ids subset

## Cross-Cutting Concerns

**Logging:**
- TensorBoard integration via `torch.utils.tensorboard.SummaryWriter`
- Metrics logged: train/reward, train/episode_length, train/landings, train/crashes, loss/* (policy, value, entropy)
- Location: `algorithms/ppo.py:180-183`, `train.py:193-203`

**Validation:**
- Genesis build validation: `scene.build()` must be called before stepping (JIT compilation)
- Observation validity: depth clamped [0, 20m], normalized to [0, 1]
- Semantic segmentation: labels must be 0/1/2 (background/soil/vineyard)
- Quaternion format: strictly w-x-y-z order throughout (Genesis convention)

**Configuration:**
- YAML-based config: `config/training_config.yaml` defines all hyperparameters
- Command-line overrides: `--n_envs`, `--max_iterations` in train.py
- Derived values computed at runtime: log_dir timestamp-stamped, save_dir created

**Device Management:**
- GPU/CPU detection: `torch.cuda.is_available()`
- Genesis backend: `gs.gpu` if CUDA available, `gs.cpu` otherwise
- All tensors explicitly moved: `.to(device)` after creation
- Mixed precision not used (full float32)

---

*Architecture analysis: 2026-02-12*
