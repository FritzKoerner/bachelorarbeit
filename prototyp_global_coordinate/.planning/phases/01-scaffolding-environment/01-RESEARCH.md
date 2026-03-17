# Phase 1: Project Scaffolding & Environment - Research

**Researched:** 2026-02-13
**Domain:** Genesis physics simulator, Python RL environment scaffolding, SB3 VecEnv interface
**Confidence:** HIGH (primary source: direct inspection of installed packages and working sister project prototyp_2)

---

## Summary

Phase 1 requires creating a Genesis-based drone environment that mirrors the structure of prototyp_2 but is simpler: flat ground only, no camera, state-only observations, and a relative-position-to-target observation instead of absolute position. The sister project at `/home/fritz-sfl/Bachelorarbeit/genesis/prototyp_2/` is a near-complete implementation reference — most of the work is adapting existing code rather than writing from scratch.

The primary technical risk is the SB3 VecEnv wrapper interface. Genesis environments use GPU tensors natively; SB3 expects numpy arrays. The wrapper must convert torch tensors to numpy on every step/reset. Phase 1 does not need to implement the SB3 wrapper itself (that is Phase 3), but the `CoordinateLandingEnv` must be designed so Phase 3 can wrap it cleanly.

A critical API difference from prototyp_2: the current Genesis 0.3.13 installation does NOT have `drone.set_vel()` or `drone.get_ang_vel()`. The correct methods are `get_ang()` for angular velocity readout, and velocity is zeroed implicitly by `set_pos(zero_velocity=True)` (which is the default). New code must NOT call `set_vel` or `set_ang_vel`.

**Primary recommendation:** Adapt prototyp_2's `VineyardLandingEnv` into `CoordinateLandingEnv`. Strip camera, replace absolute pos observation with relative-to-target, add target state. The env is simpler than prototyp_2.

---

## User Constraints

No CONTEXT.md exists for this phase. The following decisions are recorded in `.planning/PROJECT.md` and are treated as locked:

### Locked Decisions
1. SB3 over custom PPO (user preference)
2. State-only observations, no camera
3. Flat ground environment
4. Large area: target ±30m x/y, drone spawn 10-20m altitude with random x/y offset
5. Random drone offset from target (drone must navigate laterally, not just descend)
6. Velocity command action space with PD controller (reuse from prototyp_2)

### Claude's Discretion
- YAML config structure and key names
- Internal env class design (method naming, state tracking details)
- How to represent the target in the scene (visual marker is optional per REQUIREMENTS.md)
- `env_spacing` value for `scene.build()` — use reasonable value for ±30m area
- Whether to include a smoke test script or put the test in the env file itself

### Deferred Ideas (OUT OF SCOPE for Phase 1)
- SB3 VecEnv wrapper implementation (Phase 3)
- Reward function (Phase 2)
- Training loop (Phase 3)
- Evaluation (Phase 4)
- Camera/visual observations (future milestone)

---

## Standard Stack

### Core
| Library | Version (installed) | Purpose | Why Standard |
|---------|---------------------|---------|--------------|
| genesis-world | 0.3.13 | Physics simulation, drone physics, batched envs | Required by project; proven in prototyp_2 |
| torch | 2.10.0+cu126 | GPU tensor operations, all env state is tensors | Genesis operates on torch tensors natively |
| numpy | 2.3.5 | Array conversion for SB3 interface, URDF parsing | SB3 requires numpy; Genesis accepts both |
| PyYAML | 6.0.3 | Load training_config.yaml | Pattern established in prototyp_2 |

### To Install (not yet present)
| Library | Version to use | Purpose | Note |
|---------|----------------|---------|------|
| stable-baselines3 | 2.7.1 | PPO training, VecEnv interface | Will pull in gymnasium 1.2.3 |
| gymnasium | 1.2.3 | Env spec (spaces), VecEnv ABC | Required by SB3 |

### Supporting
| Library | Version (installed) | Purpose | When to Use |
|---------|---------------------|---------|-------------|
| scipy | 1.17.0 | Rotation utilities (R.from_euler for URDF transforms) | Only if camera transforms needed — Phase 1 skips camera |
| tensorboard | 2.20.0 | Training metrics | Phase 3; not needed in Phase 1 |

**Installation (for Phase 3 dependency, install now to check compatibility):**
```bash
pip install stable-baselines3
# Will install: stable-baselines3==2.7.1, gymnasium==1.2.3
```

---

## Architecture Patterns

### Recommended Project Structure
```
prototyp_global_coordinate/
├── assets/
│   └── robots/
│       └── draugas/           # symlink or copy from prototyp_2
│           ├── draugas_genesis.urdf
│           └── meshes/
├── config/
│   └── training_config.yaml   # all hyperparameters
├── controllers/
│   ├── __init__.py
│   └── velocity_controller.py # port from prototyp_2 (unchanged)
├── envs/
│   ├── __init__.py
│   └── coordinate_landing_env.py  # CoordinateLandingEnv
├── runs/                      # created at runtime
├── train.py                   # Phase 3
├── evaluate.py                # Phase 4
├── requirements.txt
└── smoke_test.py              # Phase 1 deliverable
```

### Pattern 1: Two-Phase Construction (init + build)
**What:** `__init__()` stores config only. `build()` creates the Genesis scene and is called after `gs.init()`.
**When to use:** Always with Genesis. `gs.init()` must be called exactly once before any scene construction.
**Why:** Genesis initializes global Taichi state in `gs.init()`. Scene objects cannot be created before this.
**Example from prototyp_2:**
```python
# Source: prototyp_2/envs/vineyard_landing_env.py
class VineyardLandingEnv:
    def __init__(self, config, device=None):
        # Config extraction only — no Genesis calls
        self.n_envs = config['env']['n_envs']
        self._built = False

    def build(self):
        # Called by train.py after gs.init()
        scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=2), ...)
        self.drone = scene.add_entity(gs.morphs.Drone(...))
        scene.build(n_envs=self.n_envs, env_spacing=(70, 70))
        self._built = True
```

### Pattern 2: Batched Reset with envs_idx
**What:** Reset only the subset of environments that terminated, not all 16.
**When to use:** In the `step()` auto-reset loop and when environments terminate.
**Critical detail:** `envs_idx` can be a `torch.Tensor`, `np.ndarray`, or `list[int]` — all accepted by Genesis 0.3.13.

```python
# Source: direct inspection of genesis/engine/scene.py _sanitize_envs_idx
def reset(self, env_ids: torch.Tensor = None):
    if env_ids is None:
        env_ids = torch.arange(self.n_envs, device=self.device)

    n_reset = len(env_ids)

    # Random target spawn
    target_x = (torch.rand(n_reset, device=self.device) - 0.5) * 60  # ±30m
    target_y = (torch.rand(n_reset, device=self.device) - 0.5) * 60
    self.target_pos[env_ids, 0] = target_x
    self.target_pos[env_ids, 1] = target_y
    self.target_pos[env_ids, 2] = 0.0

    # Random drone spawn: random offset from target, 10-20m altitude
    offset_x = (torch.rand(n_reset, device=self.device) - 0.5) * 20  # ±10m offset
    offset_y = (torch.rand(n_reset, device=self.device) - 0.5) * 20
    spawn_z = torch.rand(n_reset, device=self.device) * 10 + 10  # 10-20m

    spawn_pos = torch.stack([
        target_x + offset_x,
        target_y + offset_y,
        spawn_z
    ], dim=-1)

    # set_pos has zero_velocity=True by default — zeros all dofs velocity
    self.drone.set_pos(spawn_pos, envs_idx=env_ids)
    self.drone.set_quat(self._yaw_to_quat(yaw), envs_idx=env_ids)
    # NOTE: do NOT call set_vel or set_ang_vel — these methods do not exist in Genesis 0.3.13
    # zero_velocity=True in set_pos already handles this
```

### Pattern 3: Drone Entity Setup
**What:** Adding drone via `gs.morphs.Drone` with URDF reference.
**Critical:** The URDF path is resolved relative to the Python working directory.
**Example from prototyp_2 (verified working):**
```python
# Source: prototyp_2/envs/scene_builder.py
drone = scene.add_entity(
    gs.morphs.Drone(
        file="assets/robots/draugas/draugas_genesis.urdf",
        pos=(0, 0, 10.0),
        euler=(0, 0, 90),
        propellers_link_name=["prop0_link", "prop1_link", "prop2_link", "prop3_link"],
        propellers_spin=[1, -1, 1, -1],
    )
)
```

### Pattern 4: State Observation (13D)
**What:** Gather relative position to target + velocity + quat + angular velocity.
**Critical API note:** Angular velocity is `drone.get_ang()`, NOT `drone.get_ang_vel()` (does not exist).

```python
# Source: adapted from prototyp_2/envs/vineyard_landing_env.py + genesis/engine/entities/rigid_entity.py
def _get_obs(self) -> torch.Tensor:
    pos = self.drone.get_pos()           # (n_envs, 3)
    vel = self.drone.get_vel()           # (n_envs, 3)
    quat = self.drone.get_quat()         # (n_envs, 4)
    ang_vel = self.drone.get_ang()       # (n_envs, 3) — NOTE: get_ang(), not get_ang_vel()

    # Relative position to target
    rel_pos = self.target_pos - pos      # (n_envs, 3)

    state = torch.cat([rel_pos, vel, quat, ang_vel], dim=-1)  # (n_envs, 13)
    return state
```

### Pattern 5: Scene Build Parameters
**What:** `scene.build(n_envs=N, env_spacing=(dx, dy))` allocates parallel environments with spacing.
**For flat ground with ±30m target area:** Use `env_spacing=(80, 80)` to give each env 80m×80m space.
```python
# Source: prototyp_2/envs/scene_builder.py + genesis/engine/scene.py
scene.build(n_envs=16, env_spacing=(80, 80))
```

### Anti-Patterns to Avoid
- **Calling `gs.init()` inside env `__init__` or `build()`:** `gs.init()` is called exactly once in `train.py` (or `smoke_test.py`) before `env.build()`. Never inside the env class.
- **Calling `drone.set_vel()` or `drone.set_ang_vel()`:** These methods do not exist in Genesis 0.3.13. Velocity is zeroed by `set_pos(zero_velocity=True)` (default).
- **Calling `drone.get_ang_vel()`:** Method does not exist. Use `drone.get_ang()`.
- **Using absolute drone position as observation:** The observation must be RELATIVE position to target (`target_pos - drone_pos`), not absolute drone position.
- **Camera in Phase 1:** Do not add a camera. No camera, no rendering, simpler code.
- **Storing target as a Genesis entity for reward computation:** Store target as a plain `torch.Tensor` of shape `(n_envs, 3)`. No need to make it a physics object for Phase 1.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PD velocity controller | Custom RPM computation | Port `VelocityController` from `prototyp_2/controllers/low_level_controller.py` verbatim | Already debugged, URDF-derived constants, tested |
| URDF loading | Custom URDF parser | `gs.morphs.Drone(file=...)` | Genesis handles link extraction, propeller config |
| Batched physics | Manual vectorization | `scene.build(n_envs=16)` + Genesis tensor API | Genesis handles batch dim in all tensor operations |
| Quaternion to roll/pitch | Custom conversion | Use `_quat_to_rp` from prototyp_2's VelocityController | Already correct, handles edge cases |
| Yaw to quaternion | Custom conversion | Port `_yaw_to_quat` from prototyp_2 | Already correct |

**Key insight:** 90% of Phase 1 is a subset of prototyp_2. The main additions are: (1) target position state storage, (2) relative position in observation, (3) removing camera. Do not rewrite what already works.

---

## Common Pitfalls

### Pitfall 1: Wrong Angular Velocity Method Name
**What goes wrong:** `drone.get_ang_vel()` raises `AttributeError` at runtime.
**Why it happens:** Prototyp_2 code calls `get_ang_vel()` — this API was either removed or never existed in Genesis 0.3.13. The correct method is `get_ang()`.
**How to avoid:** Use `drone.get_ang()` in all new code.
**Warning signs:** Any code copied from prototyp_2 that calls `get_ang_vel` or `set_vel` or `set_ang_vel` will fail.

### Pitfall 2: `set_vel`/`set_ang_vel` Do Not Exist
**What goes wrong:** Reset code calling `drone.set_vel(zeros)` or `drone.set_ang_vel(zeros)` raises `AttributeError`.
**Why it happens:** Genesis 0.3.13 has no `set_vel` or `set_ang_vel` on `RigidEntity` / `DroneEntity`. Velocity zeroing is handled implicitly by `set_pos(zero_velocity=True)` which calls `zero_all_dofs_velocity()` internally.
**How to avoid:** In reset, call only `drone.set_pos(...)` and `drone.set_quat(...)`. Both default to `zero_velocity=True` and will zero all velocities. No explicit velocity reset needed.
**Warning signs:** Copying prototyp_2's reset section that includes `set_vel` and `set_ang_vel` calls.

### Pitfall 3: URDF Path Resolution
**What goes wrong:** `FileNotFoundError` when loading URDF — Genesis cannot find `draugas_genesis.urdf`.
**Why it happens:** The `file=` path in `gs.morphs.Drone` is relative to the current working directory when `train.py` / `smoke_test.py` is run, not to the Python file.
**How to avoid:** Either (a) always run scripts from project root, (b) use absolute path via `os.path.dirname(__file__)`, or (c) symlink `assets/` in the project root exactly as in prototyp_2. Option (c) is the pattern from prototyp_2 and should be mirrored.
**Warning signs:** Works when run from one directory, fails from another.

### Pitfall 4: `gs.init()` Called Twice
**What goes wrong:** Genesis raises exception on second `gs.init()` call.
**Why it happens:** `gs.init()` initializes global Taichi state — it must be called exactly once per process.
**How to avoid:** Call `gs.init()` only in the top-level script (`smoke_test.py`), never inside env or controller classes.
**Warning signs:** Any class `__init__` or `build()` calling `gs.init()`.

### Pitfall 5: SB3 VecEnv Requires Numpy (for Phase 3 planning)
**What goes wrong:** SB3 PPO fails because observations are torch tensors, not numpy arrays.
**Why it happens:** SB3's internal buffers and policies expect numpy arrays. Genesis returns torch tensors.
**How to avoid (Phase 3):** The `CoordinateLandingEnv` should document that its `reset()` and `step()` return `torch.Tensor`. The SB3 wrapper (Phase 3) will call `.cpu().numpy()` on observations.
**Warning signs:** PPO update crashes with tensor/array type mismatch.
**Phase 1 action:** Design `CoordinateLandingEnv` to return torch tensors natively. Document this in the class docstring. Do NOT add `.cpu().numpy()` to Phase 1 env — that belongs in the Phase 3 wrapper.

### Pitfall 6: env_spacing Too Small for ±30m Target Area
**What goes wrong:** Parallel environment visual overlap, or drones from different envs collide in simulation (no physics boundary).
**Why it happens:** Genesis places parallel envs in a grid with `env_spacing` between them. If environments spawn drones and targets at ±30m, a spacing of e.g. `(10, 10)` causes overlap.
**How to avoid:** Use `env_spacing=(80, 80)` minimum to give each env an 80m×80m footprint.
**Warning signs:** Drone positions from different envs visually overlap in viewer.

### Pitfall 7: envs_idx Must Match Tensor Shape
**What goes wrong:** Shape mismatch error when calling `drone.set_pos(spawn_pos, envs_idx=env_ids)`.
**Why it happens:** `spawn_pos` has shape `(n_reset, 3)` and `env_ids` must have length `n_reset`.
**How to avoid:** Always compute `n_reset = len(env_ids)` and allocate position tensors with that first dimension.

---

## Code Examples

Verified patterns from direct source inspection:

### Genesis Scene Initialization (Flat Ground)
```python
# Source: adapted from prototyp_2/envs/scene_builder.py + genesis/engine/scene.py
import genesis as gs

# gs.init() called ONCE in smoke_test.py/train.py before this
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -50, 40),
        camera_lookat=(0, 0, 0),
        camera_fov=60,
    ),
    show_viewer=False,
)

# Flat ground — single plane, no strips
ground = scene.add_entity(gs.morphs.Plane())

# Drone from URDF (path relative to cwd)
drone = scene.add_entity(
    gs.morphs.Drone(
        file="assets/robots/draugas/draugas_genesis.urdf",
        pos=(0, 0, 15.0),
        euler=(0, 0, 90),
        propellers_link_name=["prop0_link", "prop1_link", "prop2_link", "prop3_link"],
        propellers_spin=[1, -1, 1, -1],
    )
)

# Build with 16 parallel envs, 80m spacing
scene.build(n_envs=16, env_spacing=(80.0, 80.0))
```

### Target Position Tracking (Pure Tensor, No Scene Object)
```python
# Source: design decision — no Genesis entity needed for coordinate target in Phase 1
import torch

class CoordinateLandingEnv:
    def build(self):
        # ... scene setup ...
        # Target positions as plain tensors — no physics entity needed
        self.target_pos = torch.zeros(self.n_envs, 3, device=self.device)
```

### Reset: Correct Velocity Handling
```python
# Source: genesis/engine/entities/rigid_entity/rigid_entity.py set_pos docstring
# set_pos(zero_velocity=True) is the default — zeros all dof velocities implicitly

def reset(self, env_ids=None):
    if env_ids is None:
        env_ids = torch.arange(self.n_envs, device=self.device)
    n_reset = len(env_ids)

    # Sample targets
    target_x = (torch.rand(n_reset, device=self.device) - 0.5) * 60.0
    target_y = (torch.rand(n_reset, device=self.device) - 0.5) * 60.0
    self.target_pos[env_ids] = torch.stack(
        [target_x, target_y, torch.zeros(n_reset, device=self.device)], dim=-1
    )

    # Sample drone spawn: random offset from target, 10-20m altitude
    offset_x = (torch.rand(n_reset, device=self.device) - 0.5) * 20.0
    offset_y = (torch.rand(n_reset, device=self.device) - 0.5) * 20.0
    spawn_z = torch.rand(n_reset, device=self.device) * 10.0 + 10.0

    spawn_pos = torch.stack([target_x + offset_x, target_y + offset_y, spawn_z], dim=-1)
    yaw = torch.rand(n_reset, device=self.device) * 2 * 3.14159
    spawn_quat = self._yaw_to_quat(yaw)

    # zero_velocity=True (default) zeroes all dof velocities — no separate set_vel call needed
    self.drone.set_pos(spawn_pos, envs_idx=env_ids)
    self.drone.set_quat(spawn_quat, envs_idx=env_ids)

    self.step_count[env_ids] = 0
    self.scene.step()  # one physics step to settle
    return self._get_obs()
```

### Observation: Relative Position to Target
```python
# Source: adapted from prototyp_2 + genesis API inspection
def _get_obs(self) -> torch.Tensor:
    """Returns (n_envs, 13) float tensor: [rel_pos(3), vel(3), quat(4), ang_vel(3)]"""
    pos = self.drone.get_pos()       # (n_envs, 3)
    vel = self.drone.get_vel()       # (n_envs, 3)
    quat = self.drone.get_quat()     # (n_envs, 4) — [w, x, y, z]
    ang_vel = self.drone.get_ang()   # (n_envs, 3) — NOTE: get_ang(), NOT get_ang_vel()

    rel_pos = self.target_pos - pos  # (n_envs, 3)
    state = torch.cat([rel_pos, vel, quat, ang_vel], dim=-1)  # (n_envs, 13)
    return state
```

### Step with Velocity Controller
```python
# Source: adapted from prototyp_2/envs/vineyard_landing_env.py
def step(self, actions: torch.Tensor):
    """
    Args:
        actions: (n_envs, 4) float in [-1, 1]: [vx, vy, vz, yaw_rate]
    Returns:
        obs: (n_envs, 13) torch.Tensor
        rewards: (n_envs,) torch.Tensor  -- zeros in Phase 1
        dones: (n_envs,) bool torch.Tensor
        info: dict
    """
    target_vel = actions[:, :3] * self.vel_scale   # scale to m/s
    target_yaw_rate = actions[:, 3] * self.yaw_scale

    current_vel = self.drone.get_vel()
    current_quat = self.drone.get_quat()

    rpms = self.controller.compute_rpm(
        target_vel, target_yaw_rate, current_vel, current_quat
    )
    self.drone.set_propellels_rpm(rpms)  # note: typo in Genesis API — "propellels"
    self.scene.step()

    self.step_count += 1
    obs = self._get_obs()

    # Termination (reward function in Phase 2 — return zeros for now)
    timeout = self.step_count >= self.max_steps
    pos = self.drone.get_pos()
    out_of_bounds = (pos[:, 2] < 0.1)  # crashed into ground
    dones = timeout | out_of_bounds

    rewards = torch.zeros(self.n_envs, device=self.device)

    # Auto-reset done envs
    done_ids = torch.where(dones)[0]
    if len(done_ids) > 0:
        self.reset(done_ids)

    return obs, rewards, dones, {}
```

### VelocityController (Port Verbatim from prototyp_2)
```python
# Source: prototyp_2/controllers/low_level_controller.py
# Port unchanged — same URDF, same physics constants
# File: controllers/velocity_controller.py
# Key constants (URDF-derived):
#   base_hover_rpm: 1789.2 = sqrt(0.714 * 9.81 / (4 * 5.47e-07))
#   max_rpm: 25000
#   Motor order: FR, FL, BL, BR (X configuration)
#   Propeller spins: [1, -1, 1, -1]
```

### YAML Config Structure
```yaml
# Source: adapted from prototyp_2/config/training_config.yaml
env:
  n_envs: 16
  max_episode_steps: 500

scene:
  show_viewer: false

spawn:
  height_min: 10.0        # differs from prototyp_2 (was 6.0)
  height_max: 20.0        # differs from prototyp_2 (was 10.0)
  target_range: 30.0      # ±30m for x and y

controller:
  base_hover_rpm: 1789.2
  max_rpm: 25000
  kp_vel: 5.0
  kd_vel: 1.0
  kp_att: 10.0
  vel_scale: 2.0
  yaw_scale: 1.0

# reward: {} — populated in Phase 2
# ppo: {} — populated in Phase 3
```

### Smoke Test Pattern
```python
# Source: design for Phase 1 deliverable
# File: smoke_test.py

import torch
import genesis as gs
import yaml
from envs.coordinate_landing_env import CoordinateLandingEnv

def main():
    with open("config/training_config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)

    env = CoordinateLandingEnv(config, device=device)
    env.build()

    obs = env.reset()
    assert obs.shape == (config['env']['n_envs'], 13), f"Bad obs shape: {obs.shape}"

    action = torch.zeros(config['env']['n_envs'], 4, device=device)
    obs, rewards, dones, info = env.step(action)
    assert obs.shape == (config['env']['n_envs'], 13)

    print("Smoke test PASSED")
    print(f"  obs shape: {obs.shape}")
    print(f"  obs dtype: {obs.dtype}")
    print(f"  dones: {dones.sum().item()} of {config['env']['n_envs']} done")

if __name__ == "__main__":
    main()
```

---

## State of the Art

| Old Approach (prototyp_2) | New Approach (Phase 1) | Change | Impact |
|---------------------------|------------------------|--------|--------|
| Observation: absolute drone pos | Observation: relative pos to target | Design change | 13D obs same size but semantically different; agent learns delta-navigation |
| Scene: vineyard strips + camera | Scene: flat plane only, no camera | Simplification | Faster build, no seg mapping, no camera pose tracking |
| `drone.get_ang_vel()` | `drone.get_ang()` | Genesis 0.3.13 API | Code in prototyp_2 would crash if run as-is; must adapt |
| `drone.set_vel()` + `drone.set_ang_vel()` in reset | Removed; `set_pos(zero_velocity=True)` handles it | Genesis 0.3.13 API | Simpler reset code; prototyp_2's extra calls were redundant |
| Custom PPO training loop | SB3 PPO (Phase 3) | User decision | Phase 1 env need not implement training-specific features |

**Deprecated/outdated from prototyp_2:**
- `drone.get_ang_vel()`: does not exist in Genesis 0.3.13, use `drone.get_ang()`
- `drone.set_vel()`: does not exist in Genesis 0.3.13
- `drone.set_ang_vel()`: does not exist in Genesis 0.3.13
- Camera/segmentation code: entirely removed in Phase 1

---

## Open Questions

1. **Target visual marker in scene**
   - What we know: REQUIREMENTS.md says "optional but helpful for debugging"
   - What's unclear: Phase 1 can implement it cheaply as a static sphere or skip it; affects scene complexity
   - Recommendation: Skip in Phase 1 (the marker is purely cosmetic and adds complexity). Add in a later phase if needed.

2. **SB3 VecEnv wrapper design interface between Phase 1 and Phase 3**
   - What we know: SB3 VecEnv.reset() must return numpy arrays; SB3 VecEnv.step() must return (obs_np, rewards_np, dones_np, list[dict])
   - What's unclear: Should `CoordinateLandingEnv` itself be the SB3 VecEnv, or be wrapped separately?
   - Recommendation: Keep `CoordinateLandingEnv` as a pure Genesis env returning torch tensors. Phase 3 creates a thin `GenesisSB3Wrapper(VecEnv)` that calls the env and converts to numpy. This follows the separation of concerns pattern and is easier to test.

3. **envs_idx type: torch tensor vs numpy array**
   - What we know: Genesis 0.3.13 `_sanitize_envs_idx` accepts both `torch.Tensor` and `np.ndarray`
   - What's unclear: prototyp_2 passes `.cpu().numpy()` — this was required in an older version
   - Recommendation: Pass torch tensors directly for `envs_idx` (e.g., `env_ids` from `torch.where(dones)[0]`). No need to call `.cpu().numpy()` on indices.

4. **Assets: symlink vs copy**
   - What we know: prototyp_2 has `assets/robots/draugas/` with URDF + meshes
   - What's unclear: Whether to symlink or copy
   - Recommendation: Create a symlink from the project root: `ln -s ../prototyp_2/assets/robots assets/robots`. This avoids duplication. Alternative: copy if portability is important. Either works; symlink is cleaner.

---

## Sources

### Primary (HIGH confidence)
- Direct file inspection: `/home/fritz-sfl/miniconda3/envs/ba/lib/python3.13/site-packages/genesis/engine/entities/rigid_entity/rigid_entity.py` — verified `get_ang()`, `set_pos(zero_velocity)`, no `set_vel`/`set_ang_vel`/`get_ang_vel`
- Direct file inspection: `/home/fritz-sfl/miniconda3/envs/ba/lib/python3.13/site-packages/genesis/engine/entities/drone_entity.py` — verified `set_propellels_rpm` API (note typo in method name)
- Direct file inspection: `/home/fritz-sfl/miniconda3/envs/ba/lib/python3.13/site-packages/genesis/engine/scene.py` — verified `build(n_envs, env_spacing)` and `_sanitize_envs_idx` accepting multiple types
- Direct file inspection: `/home/fritz-sfl/Bachelorarbeit/genesis/prototyp_2/envs/vineyard_landing_env.py` — verified working env pattern
- Direct file inspection: `/home/fritz-sfl/Bachelorarbeit/genesis/prototyp_2/controllers/low_level_controller.py` — verified VelocityController
- Direct file inspection: `/home/fritz-sfl/Bachelorarbeit/genesis/prototyp_2/envs/scene_builder.py` — verified Genesis scene construction
- `pip show genesis-world`: version 0.3.13 installed
- `pip show torch`: version 2.10.0+cu126 installed
- `python3 --version`: Python 3.13.11 in miniconda ba env

### Secondary (MEDIUM confidence)
- SB3 VecEnv docs (WebFetch): https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html — abstract method signatures confirmed
- `pip install stable-baselines3 --dry-run`: confirms SB3 2.7.1 + gymnasium 1.2.3 will install

### Tertiary (LOW confidence)
- Genesis drone hover tutorial (WebFetch): https://genesis-world.readthedocs.io/en/v0.3.3/user_guide/getting_started/hover_env.html — describes structure but no complete code listing; consistent with what we see in source

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — directly inspected installed packages with pip show
- Architecture: HIGH — based on working prototyp_2 code + Genesis 0.3.13 source inspection
- API pitfalls (ang_vel rename, set_vel removal): HIGH — verified by grep on installed source files
- SB3 interface: MEDIUM — verified from official docs; no Genesis+SB3 example found in local code
- Pitfalls: HIGH (API issues), MEDIUM (env_spacing sizing)

**Research date:** 2026-02-13
**Valid until:** 2026-03-15 (stable library; Genesis versioned at 0.3.13)
