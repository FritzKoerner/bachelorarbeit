---
phase: 01-scaffolding-environment
plan: 02
subsystem: infra
tags: [python, torch, genesis, drone, rl, velocity-controller, smoke-test]

# Dependency graph
requires:
  - phase: 01-01
    provides: "VelocityController, envs/__init__.py, config/training_config.yaml, assets/robots symlink"
provides:
  - "CoordinateLandingEnv class — Genesis-batched drone env returning (n_envs,13) float32 observations"
  - "smoke_test.py — Phase 1 end-to-end test (exits 0, prints 'Smoke test PASSED')"
  - "envs/__init__.py exporting CoordinateLandingEnv"
affects:
  - "02-reward-shaping"
  - "03-training"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-phase construction: __init__() stores config only, build() creates Genesis scene after gs.init()"
    - "Relative-position observation: target_pos - drone_pos as first 3 dims of (n_envs,13) state vector"
    - "VelocityController integration: step() -> compute_rpm() -> set_propellels_rpm() chain"
    - "Auto-reset pattern: done_ids = torch.where(dones)[0]; env.reset(done_ids) inside step()"

key-files:
  created:
    - envs/coordinate_landing_env.py
    - smoke_test.py
  modified:
    - envs/__init__.py

key-decisions:
  - "Observation is relative position (target - drone), not absolute position — directly exposes landing goal"
  - "Rewards are zeros in Phase 1; Phase 2 will add shaping without changing obs/action interface"
  - "get_ang() used for angular velocity — get_ang_vel() does not exist in Genesis 0.3.13"
  - "set_vel and set_ang_vel excluded — not in Genesis 0.3.13; set_pos(zero_velocity=True) default zeroes DOFs"
  - "gs.init() called once in smoke_test.py (caller), never inside CoordinateLandingEnv"

patterns-established:
  - "Env construction: CoordinateLandingEnv(config, device) then env.build() after gs.init()"
  - "Smoke test pattern: gs.init() -> env.build() -> env.reset() -> env.step(zeros) -> assert shapes"
  - "Partial reset: env.reset(env_ids) resets subset, returns full (n_envs, 13) obs"

# Metrics
duration: 2min
completed: 2026-02-16
---

# Phase 01 Plan 02: CoordinateLandingEnv Summary

**Genesis-batched 16-env drone environment with (n_envs,13) relative-position observations and smoke test proving reset/step/shape correctness against real GPU simulation**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-16T08:05:11Z
- **Completed:** 2026-02-16T08:07:13Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Implemented CoordinateLandingEnv (234 lines) with two-phase construction, random target+spawn sampling, VelocityController integration, and auto-reset on done
- smoke_test.py runs end-to-end against real GTX 1050 GPU simulation: obs (16,13) float32, rewards (16,), dones (16,), partial reset with env_ids=[0,3,7] all pass
- Relative-position observation (target_pos - drone_pos) confirmed non-zero across all 16 envs after reset

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement CoordinateLandingEnv** - `4de3e4b` (feat)
2. **Task 2: Write and run smoke_test.py** - `904cef9` (feat)

**Plan metadata:** `(pending docs commit)`

## Files Created/Modified
- `envs/coordinate_landing_env.py` - CoordinateLandingEnv class: __init__(), build(), reset(), step(), _get_obs() (get_ang()), _yaw_to_quat()
- `envs/__init__.py` - Exports CoordinateLandingEnv
- `smoke_test.py` - Phase 1 end-to-end smoke test; exits 0 with "Smoke test PASSED"

## Decisions Made
- Observation is relative position (target - drone), not absolute position — directly exposes landing goal to RL agent
- Phase 1 rewards are zeros; Phase 2 will add shaping without changing the obs/action interface
- `get_ang()` used for angular velocity — `get_ang_vel()` does not exist in Genesis 0.3.13
- `set_vel` and `set_ang_vel` excluded — not available in Genesis 0.3.13; `set_pos()` default zeroes all DOF velocities
- `gs.init()` always called once in the caller (smoke_test.py / train.py), never inside CoordinateLandingEnv

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Genesis warnings during URDF load (legacy parser, dubious propeller mass) — these are normal for this URDF and match prototyp_2 behaviour. Simulation runs correctly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 deliverable complete: CoordinateLandingEnv returns correct shapes, smoke test passes
- Phase 2 (reward shaping) can import CoordinateLandingEnv and extend `step()` to compute non-zero rewards
- Phase 3 (SB3 training) can wrap CoordinateLandingEnv in a VecEnv adapter

---
*Phase: 01-scaffolding-environment*
*Completed: 2026-02-16*

## Self-Check: PASSED

- FOUND: envs/coordinate_landing_env.py
- FOUND: envs/__init__.py
- FOUND: smoke_test.py
- FOUND: .planning/phases/01-scaffolding-environment/01-02-SUMMARY.md
- FOUND commit 4de3e4b (Task 1 - CoordinateLandingEnv)
- FOUND commit 904cef9 (Task 2 - smoke_test.py)
