# Codebase Structure

**Analysis Date:** 2026-02-12

## Directory Layout

```
genesis/
├── train.py                     # Main training loop (PPO)
├── evaluate.py                  # Evaluation and visualization
├── requirements.txt             # Python dependencies
├── CLAUDE.md                    # Project instructions
├── config/
│   └── training_config.yaml     # All hyperparameters
├── envs/
│   ├── __init__.py
│   ├── vineyard_landing_env.py  # Gym-like environment wrapper
│   └── scene_builder.py         # Genesis scene construction
├── networks/
│   ├── __init__.py
│   ├── actor_critic.py          # Actor-Critic policy network
│   └── cnn_encoder.py           # Vision encoder (depth+segmentation)
├── algorithms/
│   ├── __init__.py
│   └── ppo.py                   # PPO trainer with GAE
├── controllers/
│   ├── __init__.py
│   └── low_level_controller.py  # Velocity → RPM converter
├── utils/
│   ├── __init__.py
│   └── reward_functions.py      # Multi-component reward function
├── assets/
│   ├── robots/
│   │   └── draugas/
│   │       ├── draugas_genesis.urdf  # Drone URDF model
│   │       └── meshes/               # Drone mesh files
│   └── scene/
│       └── vineyard-eltville-germany/
│           ├── source/               # OBJ mesh (not yet integrated)
│           └── textures/             # Texture files
├── prototyp/                    # Early prototype (separate from main)
│   ├── train.py
│   ├── env.py
│   ├── network.py
│   ├── scene_builder.py
│   ├── observations.py
│   ├── actions.py
│   ├── rewards.py
│   └── utils.py
├── logs/                        # TensorBoard logs (created at runtime)
├── checkpoints/                 # Model checkpoints (created at runtime)
└── .planning/
    └── codebase/                # GSD analysis documents
```

## Directory Purposes

**Root:**
- Purpose: Training and evaluation entry points
- Contains: Main scripts for RL pipeline

**config/:**
- Purpose: Centralized configuration management
- Contains: YAML files with all hyperparameters (environment, network, PPO, rewards)
- Key files: `training_config.yaml`

**envs/:**
- Purpose: Environment abstraction and Genesis integration
- Contains: Gym-like wrapper, scene building, semantics mapping
- Key files:
  - `vineyard_landing_env.py`: Core environment with reset/step interface
  - `scene_builder.py`: Genesis scene construction and semantic label handling

**networks/:**
- Purpose: Policy network definitions
- Contains: Actor-Critic architecture, vision encoding
- Key files:
  - `actor_critic.py`: Main policy with separate encoder branches
  - `cnn_encoder.py`: CNN for visual observations

**algorithms/:**
- Purpose: RL algorithm implementation
- Contains: PPO trainer with advantage estimation and updates
- Key files: `ppo.py`

**controllers/:**
- Purpose: Low-level drone control
- Contains: Velocity → RPM conversion with PD control and attitude mixing
- Key files: `low_level_controller.py`

**utils/:**
- Purpose: Task-specific utilities
- Contains: Multi-component reward function
- Key files: `reward_functions.py`

**assets/:**
- Purpose: Robot and scene models
- Contains:
  - `robots/draugas/`: URDF drone model and meshes
  - `scene/vineyard-eltville-germany/`: Real vineyard geometry (OBJ + textures)

**prototyp/:**
- Purpose: Prototype development (separate from production)
- Contains: Early versions of environment, training, networks
- Status: Legacy; kept for reference but not actively used

**logs/ and checkpoints/:**
- Purpose: Training outputs
- Created at: Runtime by train.py
- Structure:
  - `logs/{YYYYMMDD_HHMMSS}/`: TensorBoard event files
  - `checkpoints/{YYYYMMDD_HHMMSS}/`: Model checkpoints (best_model.pt, checkpoint_N.pt, final_model.pt)

## Key File Locations

**Entry Points:**

- `train.py`: Main training script
  - Loads config → creates environment/policy/trainer → runs training loop
  - Saves checkpoints and logs metrics

- `evaluate.py`: Evaluation and visualization
  - Loads checkpoint → runs episodes with deterministic policy → visualizes trajectories

**Configuration:**

- `config/training_config.yaml`: Single source of truth for all hyperparameters
  - Environment: n_envs, spawn height, image resolution, episode steps
  - Scene: strip width/length, number of strips
  - Controller: hover RPM, max RPM, PD gains
  - Network: hidden dimensions, feature dimensions
  - PPO: learning rate, gamma, lambda, clip epsilon, epochs, batch size
  - Rewards: component weights, thresholds (landing height, velocity, crash angle)
  - Logging: log/save directories, intervals

**Core Logic:**

- `envs/vineyard_landing_env.py`: Central environment logic
  - Observation composition: depth + semantic segmentation + state vector
  - Action processing: scaling, controller integration
  - Episode termination: landing, crash, timeout, out-of-bounds
  - Auto-reset for vectorized training

- `envs/scene_builder.py`: Physics scene setup
  - Ground plane creation
  - Alternating soil/vineyard strips with semantic labels
  - Drone spawning with initial conditions
  - Camera attachment and segmentation mapping

- `networks/actor_critic.py`: Policy network
  - Vision encoder → feature extraction
  - State encoder → feature extraction
  - Fusion MLP → shared representation
  - Separate actor (mean/std) and critic heads

- `controllers/low_level_controller.py`: Control hierarchy
  - Velocity error → desired acceleration (PD)
  - Acceleration → attitude (roll/pitch estimation)
  - Attitude error → differential motor RPM
  - Motor mixing for quadrotor (X-configuration)

- `algorithms/ppo.py`: Training algorithm
  - GAE advantage estimation
  - Mini-batch PPO updates with clipping
  - Loss computation: policy + value + entropy
  - Checkpoint save/load

- `utils/reward_functions.py`: Task reward definition
  - 6 components: descent, velocity, semantic, landing, crash, smoothness
  - Configurable weights via config file
  - Semantic region extraction (center of image)

**Testing:**

- Not yet implemented (TODO)
- Would go in `tests/` with pytest

## Naming Conventions

**Files:**

- `snake_case.py` for module files (all files)
- `_private.py` for internal-only modules (none currently)
- `test_*.py` for test files (convention if tests added)
- Config files: `*_config.yaml` (e.g., training_config.yaml)

**Directories:**

- `lowercase_with_underscores` for package directories (envs, networks, algorithms, controllers, utils, assets, prototyp)
- Logical grouping: separate concerns into distinct directories

**Classes:**

- `PascalCase` for all classes (VineyardLandingEnv, ActorCritic, VelocityController, PPOTrainer, VineyardSceneBuilder, LandingRewardFunction)

**Functions:**

- `snake_case` for all functions (reset, step, compute_gae, update, etc.)
- Private functions: `_snake_case` (e.g., `_get_obs`, `_quat_to_rp`, `_init_weights`)

**Variables:**

- `snake_case` for local and instance variables (n_envs, obs_visual_shape, target_vel, current_quat, rewards)
- CONSTANTS: `UPPER_CASE` (SEMANTIC_LABELS, SOIL_COLOR, LABEL_SOIL)
- Config keys: `snake_case` in YAML (learning_rate, gamma, gae_lambda)

**Tensors:**

- Shape notation in comments: `(n_envs, H, W)` or `(batch, 3)` to indicate batch size first
- Device explicit: all tensors created with `device=self.device` or `.to(device)`

## Where to Add New Code

**New Feature:**

- **High-level policy component:** Add to `networks/` with integration in `ActorCritic`
  - Example: attention module for focusing on center region
  - Location: `networks/attention.py`
  - Integration: import in `actor_critic.py`, compose with existing encoders

- **New reward component:** Add method to `LandingRewardFunction` in `utils/reward_functions.py`
  - Example: penalize lateral drift
  - Pattern: compute component tensor (n_envs,), add to total reward with weight from config
  - Weight management: add w_newfeature to config/training_config.yaml

- **New environment condition:** Add to `VineyardLandingEnv.step()` termination logic
  - Example: obstacle collision detection
  - Location: `envs/vineyard_landing_env.py:188-190`
  - Pattern: compute boolean tensor (n_envs,), add to dones cumulative OR

**New Module/Package:**

- **New algorithm variant (e.g., SAC):** Create new file in `algorithms/`
  - Location: `algorithms/sac.py`
  - Pattern: class inheriting from base trainer, implementing update method
  - Integration: import in train.py, swap PPOTrainer for SACTrainer

- **New sensor (e.g., lidar):** Extend `VineyardSceneBuilder` and `VineyardLandingEnv`
  - Location: Add camera/sensor in scene_builder.py
  - Observation: extend obs dict in _get_obs() with new sensor data
  - Integration: update obs_visual_shape or add obs_lidar_shape

**Utilities:**

- **Shared helpers:** Place in `utils/`
  - Example: trajectory recording, visualization helpers
  - Current: only reward_functions.py; can expand for math utilities, logging, etc.

- **Training utilities:** Can expand `utils/` with:
  - Learning rate scheduling
  - Batch normalization helpers
  - Metric computation

## Special Directories

**assets/robots/draugas/:**
- Purpose: Custom drone definition
- Contains: URDF file and mesh geometry
- Generated: User-provided (custom design)
- Committed: Yes (critical for reproducibility)
- Key file: `draugas_genesis.urdf` referenced in `scene_builder.py:96`

**assets/scene/vineyard-eltville-germany/:**
- Purpose: Real vineyard geometry (future integration)
- Contains: OBJ mesh files and textures
- Generated: No (from external source)
- Committed: Yes (geometry data)
- Status: Not yet integrated into scene building (static strips used instead)

**prototyp/:**
- Purpose: Legacy prototype code
- Contains: Early versions of training loop, environment, networks
- Generated: No (manually created development versions)
- Committed: Yes (historical reference)
- Integration: Completely separate; not imported by main code

**logs/ and checkpoints/:**
- Purpose: Training artifacts
- Generated: Yes (created at runtime by train.py)
- Committed: No (too large, not code)
- Structure: Timestamped subdirectories for multiple runs
- Management: User responsibility for cleanup

**.planning/codebase/:**
- Purpose: GSD analysis documents
- Generated: Yes (by mapping commands)
- Committed: Yes (documentation)
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md

---

*Structure analysis: 2026-02-12*
