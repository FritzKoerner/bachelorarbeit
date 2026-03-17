# Research: SB3 + Genesis Batched GPU Environment Integration

**Topic:** Integrating Stable-Baselines3 PPO with a custom batched GPU environment (Genesis, 16 parallel drone envs)
**Researched:** 2026-02-13
**Overall confidence:** HIGH (confirmed against official SB3 docs, Isaac Lab source code, and SB3 GitHub issues)

---

## Executive Summary

SB3 PPO can be used with Genesis batched environments, but it requires a thin custom `VecEnv` subclass that bridges the two APIs. The core challenge is that SB3's `VecEnv` contract is **numpy-first**: all return types (`VecEnvObs`, `VecEnvStepReturn`) are specified as `np.ndarray`. SB3 will never natively keep tensors on GPU — this is an explicit design decision by maintainers (Issue #314, confirmed closed as "won't fix" for main library).

The correct pattern — validated by Isaac Lab's production `Sb3VecEnvWrapper` — is:
1. Subclass `stable_baselines3.common.vec_env.VecEnv` directly (not `VecEnvWrapper`, since there is no underlying SB3 VecEnv to wrap).
2. Expose `num_envs`, `observation_space`, `action_space` as attributes.
3. In `step_async()`: store the numpy actions, convert to GPU tensor internally.
4. In `step_wait()`: run one physics step, then `.detach().cpu().numpy()` all outputs.
5. In `reset()`: reset all environments, return numpy observations.
6. Handle per-environment resets internally (Genesis already supports `reset_idx`-style partial resets).
7. Wrap with `VecMonitor` for SB3's episodic reward/length logging.

The CPU/GPU transfer overhead is real but acceptable for 16 environments. For 16 envs the transfer cost is negligible compared to physics simulation time. The overhead matters only at 1000+ parallel envs.

---

## Question 1: How to wrap a batched GPU tensor environment for SB3?

### Answer: Subclass `VecEnv` directly (not `VecEnvWrapper`)

`VecEnvWrapper` is for wrapping an existing SB3 `VecEnv` instance. For a Genesis environment that is itself the authoritative vectorized source, you subclass `VecEnv` directly.

**Minimum required abstract methods** (from `base_vec_env.py`):

```python
from stable_baselines3.common.vec_env import VecEnv
import numpy as np

class GenesisVecEnv(VecEnv):
    def __init__(self, genesis_env):
        self.genesis_env = genesis_env
        n_envs = genesis_env.n_envs

        # Define spaces — MUST be gymnasium.spaces objects
        obs_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        act_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,), dtype=np.float32
        )
        super().__init__(n_envs, obs_space, act_space)

    # --- 4 abstract methods that MUST be implemented ---

    def reset(self) -> np.ndarray:
        obs_tensor = self.genesis_env.reset()  # (n_envs, obs_dim) torch.Tensor on GPU
        return obs_tensor.detach().cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        # Store; will execute in step_wait
        self._actions = torch.as_tensor(actions, dtype=torch.float32, device=self.genesis_env.device)

    def step_wait(self):
        obs_t, rew_t, done_t, info = self.genesis_env.step(self._actions)
        obs  = obs_t.detach().cpu().numpy()           # (n_envs, obs_dim)
        rews = rew_t.detach().cpu().numpy()           # (n_envs,)
        dones = done_t.detach().cpu().numpy()         # (n_envs,) bool
        # SB3 expects a list of dicts, one per env
        infos = [{k: v[i].item() if hasattr(v, '__getitem__') else v
                  for k, v in info.items()} for i in range(self.num_envs)]
        # terminal_observation: SB3 expects obs at done step (before auto-reset)
        # Genesis auto-resets internally, so store obs before reset in genesis_env.step()
        for i, done in enumerate(dones):
            if done:
                infos[i]["terminal_observation"] = obs[i].copy()
        return obs, rews, dones, infos

    def close(self):
        self.genesis_env.close()

    # --- 4 more abstract methods: minimal stubs are fine ---
    def get_attr(self, attr_name, indices=None): ...
    def set_attr(self, attr_name, value, indices=None): ...
    def env_method(self, method_name, *args, indices=None, **kwargs): ...
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
```

**Critical: Genesis auto-resets internally.** prototyp_2's `step()` already calls `self.reset(done_ids)` when environments are done. This means by the time `step_wait()` returns, done environments already have their new initial observations. The `terminal_observation` trick is the standard SB3 mechanism for passing the true last observation of a done episode to the PPO rollout buffer. You must capture it **before** the auto-reset happens — so save it inside `genesis_env.step()` before calling the internal reset.

### Confirmed pattern from Isaac Lab

Isaac Lab's `Sb3VecEnvWrapper` follows exactly this structure:

```python
# From isaac-sim/IsaacLab source (isaaclab_rl/sb3.py)
def step_async(self, actions):
    if not isinstance(actions, torch.Tensor):
        actions = np.asarray(actions)
    actions = torch.from_numpy(actions).to(device=self.sim_device, dtype=torch.float32)
    self._async_actions = actions

def step_wait(self):
    obs, rewards, terminated, truncated, extras = self.env.step(self._async_actions)
    # Convert all tensors to numpy for SB3
    rewards = rewards.detach().cpu().numpy()
    terminated = terminated.detach().cpu().numpy()
    # ... build infos list with terminal_observation ...
    return self._process_obs(obs), rewards, dones, infos
```

Source: [Isaac Lab wrapping guide](https://isaac-sim.github.io/IsaacLab/main/source/how-to/wrap_rl_env.html)

---

## Question 2: Does SB3 support GPU-resident observations/actions natively?

### Answer: No. GPU support is explicitly out of scope for SB3 main.

**Confirmed from SB3 Issue #314** (GPU optimization discussion):

- SB3's internal type contract defines `VecEnvObs = Union[np.ndarray, dict[str, np.ndarray], tuple[np.ndarray, ...]]` — numpy only.
- Actions are converted from GPU tensors to numpy via `.cpu().numpy()` before being sent to the environment.
- A custom fork demonstrated 2–4x speedup from eliminating these transfers on an NVIDIA TITAN RTX, but maintainers declined to merge it.
- Maintainer quote: "GPU acceleration would be a messy addition to support."
- Recommendation from maintainers: use specialized GPU-native libraries (Isaac Lab's own RL stack, rsl-rl) for maximum throughput.

**For 16 envs this does not matter.** The transfer overhead for a (16, obs_dim) float32 tensor is microseconds. At 16 environments with state-only observations (~13-20 floats per env), the numpy conversion costs are negligible.

**Confirmed from Isaac Lab paper** (arxiv.org/abs/2301.04195, Section V-A): SB3 is measurably slower than rl_games for large-scale parallel training (100s–1000s of envs) specifically due to this GPU↔CPU overhead. At 16 envs, this gap is not material.

Source: [SB3 Issue #314](https://github.com/DLR-RM/stable-baselines3/issues/314), [Isaac Lab discussion #528](https://github.com/isaac-sim/IsaacLab/discussions/528)

---

## Question 3: Best VecEnv wrapper pattern for already-internally-vectorized envs

### Answer: Direct VecEnv subclass with synchronous step_async/step_wait

The canonical pattern for environments like Genesis, Isaac Lab, Brax, and EnvPool is documented in SB3 official docs:

> "Some massively parallel simulations such as EnvPool, Isaac Lab, Brax or ProcGen already produce a vectorized environment to speed up data collection. To use SB3 with these tools, you need to wrap the env with tool-specific VecEnvWrapper that pre-processes the data for SB3."

The "async" naming is misleading for synchronous environments — `step_async` simply stores the actions, and `step_wait` runs the actual step. Because Genesis is synchronous (all 16 envs step in lock-step), this is just a two-call dispatch pattern.

**Key design rules:**
1. `step_async` must be cheap (just store actions).
2. `step_wait` does the real work.
3. `reset()` resets all environments (not selective — SB3 calls `reset()` once at the start, then relies on per-episode auto-reset via `terminal_observation`).
4. Do NOT use `DummyVecEnv` or `SubprocVecEnv` wrapping around your Genesis env — they assume gym.Env interface and would serialize/parallelize incorrectly.

**Do not do this:**
```python
# WRONG: DummyVecEnv wraps gym.Env, not already-vectorized envs
env = DummyVecEnv([lambda: GenesisEnvSingleInstance()])
```

**Do this:**
```python
# CORRECT: Direct VecEnv subclass
env = GenesisVecEnv(genesis_env)
env = VecMonitor(env)  # adds episodic reward/length logging
model = PPO("MlpPolicy", env, n_steps=512, ...)
```

Source: [SB3 VecEnv docs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)

---

## Question 4: Existing examples of SB3 with Genesis, Isaac Gym, or similar

### Genesis official examples

Genesis's official hover_env tutorial uses **rsl-rl** (not SB3):
```bash
pip install tensorboard rsl_rl_lib==2.2.4
python hover_train.py -e drone-hovering -B 8192 --max_iterations 301
```

The Genesis environment returns PyTorch tensors directly and rsl-rl handles them natively on GPU. **There is no official Genesis + SB3 example.** This is a gap we fill by implementing the VecEnv wrapper.

Source: [Genesis hover_env docs](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hover_env.html)

### Isaac Lab (closest analogue — same GPU tensor pattern)

Isaac Lab provides `Sb3VecEnvWrapper` in `isaaclab_rl/sb3.py`. This is the gold standard reference implementation. Key behaviors confirmed from source:

1. `step_async`: converts numpy actions to GPU torch tensor, stores as `self._async_actions`.
2. `step_wait`: calls `env.step()`, converts all outputs via `.detach().cpu().numpy()`.
3. `reset`: calls `env.reset()`, converts obs via `_process_obs()` which handles numpy conversion.
4. Builds `infos` as a list of dicts with `terminal_observation` set when `dones[i]` is True.
5. Clamps unbounded action spaces to `[-100, 100]` for SB3 compatibility.

Source: [Isaac Lab RL wrapper API docs](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html)

### Brax (gymnasium.vector.VectorEnv route)

SB3 merged PR #2072 adding `gymnasium.vector.VectorEnv` adapter support. This is relevant if you want to use Brax-style environments. For Genesis (which does not use gymnasium's VectorEnv), the direct `VecEnv` subclass is still the right approach.

---

## Question 5: Observation space format for state-vector-only policy

### Answer: `gymnasium.spaces.Box(low=-inf, high=inf, shape=(obs_dim,), dtype=np.float32)`

For this project's state-vector observation (relative position, velocity, orientation, angular velocity):

```python
import gymnasium
import numpy as np

# Example for a 17-dim state vector:
# rel_pos(3) + drone_vel(3) + drone_quat(4) + ang_vel(3) + target_xy(2) + prev_actions(2)
# Adjust obs_dim to match your actual observation design

obs_space = gymnasium.spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(obs_dim,),  # e.g. 17 for this project
    dtype=np.float32
)
```

**SB3 PPO with `MlpPolicy` automatically selects a 2-layer MLP** for 1D Box observations. No custom policy architecture needed.

**Important:** SB3 does NOT support `Dict` observation spaces with PPO's default `MlpPolicy`. The state vector must be flat (1D Box). prototyp_2's dict-based `{'visual': ..., 'state': ...}` observation cannot be passed directly — you must flatten or select the state-only component.

For this project (state-only), return a flat numpy array of shape `(obs_dim,)` per environment, giving total shape `(n_envs, obs_dim)` from the VecEnv.

Source: [SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [SB3 custom env guide](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)

---

## Question 6: Gotchas with SB3 PPO + custom vectorized envs

### Gotcha 1: `terminal_observation` — CRITICAL

**Problem:** Genesis auto-resets done environments internally (same as prototyp_2). By the time `step_wait()` returns, the observation for done environments is already the new episode's initial observation, not the terminal observation.

**SB3's PPO rollout buffer** calls `self._last_obs = new_obs` and uses `terminal_observation` from the info dict to correctly bootstrap value for truncated episodes. If you omit it, value bootstrapping is silently wrong for done episodes — the policy learns from incorrect advantages.

**Fix:** Before calling the internal Genesis reset, save a copy of the observation for done envs:

```python
# Inside genesis_env.step() or wrapper step_wait():
# 1. Get obs BEFORE auto-reset
obs_t = self._get_obs()  # (n_envs, obs_dim)

# 2. Detect dones
dones = (landed | crashed | timeout | out_of_bounds)

# 3. Auto-reset internally
done_ids = torch.where(dones)[0]
if len(done_ids) > 0:
    self._reset_idx(done_ids)  # internal reset, updates obs for done envs

# 4. Get post-reset obs
new_obs_t = self._get_obs()  # (n_envs, obs_dim) — done envs now show episode 0 obs

# 5. In wrapper: set terminal_observation from obs_t (pre-reset) for done envs
obs_np = new_obs_t.detach().cpu().numpy()
terminal_obs_np = obs_t.detach().cpu().numpy()
for i in range(n_envs):
    if dones_np[i]:
        infos[i]["terminal_observation"] = terminal_obs_np[i]
```

Source: [SB3 VecEnv docs — terminal_observation section](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)

### Gotcha 2: `n_steps * n_envs` must be > 1

PPO normalizes advantages over the rollout buffer. With `n_envs=16`, any `n_steps >= 1` works. Typical recommendation: `n_steps=512` gives `512 * 16 = 8192` samples per update — a solid batch size.

### Gotcha 3: PPO `device` parameter is for the POLICY network, not the environment

```python
model = PPO("MlpPolicy", env, device="cuda", n_steps=512, ...)
```

Setting `device="cuda"` moves the policy network to GPU. The env still returns numpy. The forward pass (policy inference) happens on GPU, but observations come from CPU numpy. SB3 internally calls `obs_as_tensor(obs, device)` to convert numpy → CUDA tensor before the policy forward pass.

### Gotcha 4: SB3 wraps non-VecEnv in `DummyVecEnv` automatically

If you pass a plain `gym.Env` to PPO, SB3 wraps it in `DummyVecEnv([lambda: env])`. If you pass a proper `VecEnv` subclass (your wrapper), SB3 uses it as-is. Confirm by checking `isinstance(model.env, VecEnv)` after construction.

### Gotcha 5: VecMonitor is needed for episodic stats

SB3 PPO logs `ep_rew_mean` and `ep_len_mean` to TensorBoard automatically — but only if the environment is wrapped with `VecMonitor`. Without it, these metrics will be absent.

```python
from stable_baselines3.common.vec_env import VecMonitor
env = GenesisVecEnv(genesis_env)
env = VecMonitor(env)  # REQUIRED for ep_rew_mean/ep_len_mean in TensorBoard
```

`VecMonitor` reads `episode` key from `info` dicts (set by SB3's `DummyVecEnv`), or alternatively tracks rewards internally. When you provide your own `infos` list, ensure `terminal_observation` and not `episode` (VecMonitor injects that itself).

### Gotcha 6: Action space bounds must be finite

SB3 checks that action space bounds are finite. If your `action_space` is `Box(low=-np.inf, high=np.inf, ...)`, SB3 will raise a warning or error. Use explicit bounds matching your controller's expected range:

```python
# Correct for velocity commands in [-1, 1] normalized:
act_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
```

### Gotcha 7: `env_method`, `get_attr`, `set_attr` stubs

SB3 calls `env.get_attr("render_mode")` and similar during initialization and eval callbacks. These must exist and return sensible values. Minimal stubs:

```python
def get_attr(self, attr_name, indices=None):
    return [getattr(self.genesis_env, attr_name, None)] * self.num_envs

def set_attr(self, attr_name, value, indices=None):
    setattr(self.genesis_env, attr_name, value)

def env_method(self, method_name, *args, indices=None, **kwargs):
    method = getattr(self.genesis_env, method_name)
    return [method(*args, **kwargs)] * self.num_envs

def env_is_wrapped(self, wrapper_class, indices=None):
    return [False] * self.num_envs
```

### Gotcha 8: SB3 calls `reset()` with no arguments (no env_ids)

SB3's `VecEnv.reset()` signature is `reset() -> VecEnvObs`. It resets ALL environments at once and returns the initial observations. It does NOT call `reset()` per-episode — it relies on the auto-reset behavior (environments auto-reset when done, and `terminal_observation` carries the last obs). Your `reset()` must reset all `n_envs` and return shape `(n_envs, obs_dim)`.

### Gotcha 9: Genesis `gs.init()` must be called before building the scene

This is already handled correctly in prototyp_2's `train.py` pattern. Call `gs.init(backend=gs.gpu)` before constructing the environment, and call `env.build()` before passing to SB3. The wrapper should require a fully built genesis env.

---

## Recommended Integration Architecture

```
train.py
  │
  ├── gs.init(backend=gs.gpu)
  │
  ├── genesis_env = GlobalCoordinateLandingEnv(config)
  ├── genesis_env.build()
  │
  ├── sb3_env = GenesisVecEnv(genesis_env)      # VecEnv subclass
  ├── sb3_env = VecMonitor(sb3_env)             # episodic stats
  │
  └── model = PPO(
          "MlpPolicy",
          sb3_env,
          n_steps=512,          # steps per env per update; rollout = 512 * 16 = 8192
          n_epochs=10,
          batch_size=256,
          learning_rate=3e-4,
          gamma=0.99,
          gae_lambda=0.95,
          clip_range=0.2,
          ent_coef=0.01,
          device="cuda",
          tensorboard_log="./runs",
          verbose=1,
      )
  model.learn(total_timesteps=5_000_000)
  model.save("global_coordinate_ppo")
```

---

## Performance Expectations

| Aspect | Expected | Rationale |
|--------|----------|-----------|
| GPU↔CPU transfer overhead | Negligible at 16 envs | (16, 17) float32 = 1.1 KB per step; microseconds |
| Training speed vs custom PPO | Comparable | SB3 PPO forward pass on GPU; only data transfer differs |
| SB3 vs rsl-rl speed gap | Visible at 1000+ envs | At 16 envs, difference is not meaningful |
| MlpPolicy for state-only | Appropriate | Auto-selected 2-layer MLP for 1D Box observations |

---

## File Layout Recommendation

```
prototyp_global_coordinate/
├── envs/
│   ├── global_coordinate_env.py     # Genesis environment (VineyardLandingEnv analog)
│   └── sb3_vec_env.py               # GenesisVecEnv(VecEnv) wrapper
├── controllers/
│   └── low_level_controller.py      # reuse from prototyp_2
├── utils/
│   └── reward_functions.py          # new distance-based reward
└── train.py                         # gs.init → build env → wrap → PPO.learn()
```

---

## Sources

| Source | URL | Confidence |
|--------|-----|------------|
| SB3 VecEnv interface documentation | https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html | HIGH |
| SB3 PPO documentation | https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html | HIGH |
| SB3 custom env guide | https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html | HIGH |
| SB3 Issue #314: GPU optimization refused | https://github.com/DLR-RM/stable-baselines3/issues/314 | HIGH |
| Isaac Lab SB3 wrapper wrapping guide | https://isaac-sim.github.io/IsaacLab/main/source/how-to/wrap_rl_env.html | HIGH |
| Isaac Lab SB3 wrapper API | https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html | HIGH |
| Isaac Lab discussion: SB3 slower than rl_games | https://github.com/isaac-sim/IsaacLab/discussions/528 | HIGH |
| Genesis hover_env example (uses rsl-rl not SB3) | https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hover_env.html | HIGH |
| SB3 Issue #1745: gymnasium.vector.VectorEnv support | https://github.com/DLR-RM/stable-baselines3/issues/1745 | MEDIUM |
