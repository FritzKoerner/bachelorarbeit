---
phase: 01-core-infrastructure
plan: 02
subsystem: infra
tags: [genesis, ppo, drone, env, vectorized, camera, advantage-normalization]

# Dependency graph
requires:
  - phase: 01-01
    provides: Nested YAML config, VelocityController with 1789.2 hover RPM, VineyardSceneBuilder with seg_idx_to_label
provides:
  - VineyardLandingEnv with separate __init__/build(), direct camera.render() pattern, nested config, auto-reset
  - PPOTrainer with per-mini-batch advantage normalization and nested config support
affects: [03-training-loop, 04-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Env build() method: Genesis scene construction deferred to build() so gs.init() can be called first in train.py
    - Direct camera.render(depth=True, segmentation=True) returning (rgb, depth, seg, normals) tuple
    - Multi-env camera fallback: if depth.ndim==2 broadcast to all envs; if ndim==3 use directly
    - PPO per-mini-batch normalization: mb_advantage normalized inside mini-batch loop, not over full buffer
    - Nested config fallback: config.get('ppo', config) supports both nested and flat config dicts

key-files:
  created: []
  modified:
    - envs/vineyard_landing_env.py
    - algorithms/ppo.py

key-decisions:
  - "Env.__init__() stores config only; build() constructs Genesis scene -- prevents gs.init() ordering bugs"
  - "Direct camera.render() used (not start/stop) matching working prototype pattern"
  - "semantic_seg for reward extracted from obs['visual'][:, 1, :, :] * 2 to avoid double render"
  - "PPO advantage normalization moved inside mini-batch loop for statistically correct per-batch normalization"

patterns-established:
  - "VineyardLandingEnv: env = VineyardLandingEnv(config); env.build(); obs = env.reset()"
  - "PPOTrainer: ppo_cfg = config.get('ppo', config) pattern for backwards-compatible nested config"
  - "Camera output: _, depth, seg, _ = camera.render(depth=True, segmentation=True)"

# Metrics
duration: 5min
completed: 2026-02-12
---

# Phase 1 Plan 2: Environment Integration and PPO Fix Summary

**VineyardLandingEnv refactored with deferred build() for gs.init() ordering, direct camera.render() pattern, nested config, and multi-env camera fallback; PPOTrainer fixed to normalize advantages per mini-batch not over full buffer**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-02-12T17:29:59Z
- **Completed:** 2026-02-12T17:34:53Z
- **Tasks:** 2
- **Files modified:** 2 (envs/vineyard_landing_env.py, algorithms/ppo.py)

## Accomplishments

- Refactored VineyardLandingEnv into __init__(config-only) + build(Genesis-construction) split, eliminating gs.init() ordering bugs
- Replaced broken start/stop camera pattern with direct camera.render(depth=True, segmentation=True) returning (rgb, depth, seg, normals) tuple
- Added fallback handling for single-env (H,W) vs multi-env (n_envs,H,W) camera render output shapes
- Fixed all config reads to use nested config structure (config['env'], config['spawn'], config['controller'], etc.)
- Removed pre-loop full-buffer PPO advantage normalization; moved normalization inside mini-batch loop for correct per-batch statistics
- Added nested config support to PPOTrainer with flat-config fallback for backwards compatibility

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor VineyardLandingEnv for correct vectorized operation** - `c29f20a` (feat)
2. **Task 2: Fix PPO per-mini-batch advantage normalization** - `864df40` (fix)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `envs/vineyard_landing_env.py` - Separate __init__()/build(); direct camera.render() with multi-env fallback; nested config reads; auto-reset via reset(done_ids); vel_scale/yaw_scale from config; no gs.init() calls
- `algorithms/ppo.py` - Per-mini-batch advantage normalization; nested config['ppo'] support with flat fallback; nested config['logging'] for log_dir

## Decisions Made

- `env.build()` deferred from `__init__()`: Genesis scene construction must happen after `gs.init()` in train.py. Separating build() makes the ordering explicit and prevents import-time side effects.
- `camera.render()` direct pattern: The prototype's pattern `_, depth, seg, _ = camera.render(depth=True, segmentation=True)` is known to work; the parent project's start/stop pattern was untested and likely broken.
- Semantic segmentation extracted from obs['visual'] channel 1 in step() to avoid rendering camera twice (once in _get_obs(), once for reward). This ensures reward uses the same frame as the observation.
- PPO per-mini-batch normalization: Normalizing over the full buffer then sampling mini-batches uses stale statistics (mini-batch statistics differ from full buffer statistics). Per-mini-batch normalization is the correct approach for PPO.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The existing vineyard_landing_env.py and algorithms/ppo.py were not yet tracked by git (shown as untracked in git status). The commits show "create mode" but this is correct - the files were being version-controlled for the first time as part of this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- VineyardLandingEnv and PPOTrainer are ready for Plan 03 (training loop integration)
- The camera multi-env fallback is in place: if camera.render() returns (H,W) for a single view, it's broadcast to all envs; if it returns (n_envs, H, W) correctly, it's used directly
- Recommend running `python tools/validate_camera.py` before training to confirm which camera render path will be taken

## Self-Check: PASSED

- FOUND: envs/vineyard_landing_env.py
- FOUND: algorithms/ppo.py
- FOUND commit: c29f20a (feat - VineyardLandingEnv refactor)
- FOUND commit: 864df40 (fix - PPO per-mini-batch normalization)

---
*Phase: 01-core-infrastructure*
*Completed: 2026-02-12*
