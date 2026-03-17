# STATE — Global Coordinate Landing

## Project Reference

**Core value:** Teach a drone precise coordinate-targeted landing via RL
**Current focus:** Phase 1, Plan 1 complete — scaffold ready

## Current Position

Phase: 1 of 4 — Scaffolding & Environment
Plan: 2 of 2 complete (Phase 1 fully done)
Status: Phase 1 complete — ready to execute Phase 2 (reward shaping)

## Progress

[####░░░░░░] 40%

## Recent Decisions

- SB3 PPO over custom PPO (user preference)
- State-only observations, no camera
- Flat ground environment
- Large area (±30m, 10-20m altitude)
- Random drone offset from target
- Velocity command action space with PD controller
- Verbatim port of VelocityController from prototyp_2 (01-01)
- Absolute path symlink for assets/robots to avoid working-directory fragility (01-01)
- env_spacing=80m for ±30m target range (01-01)
- Runtime: ba conda env (torch 2.10.0+cu126) identified as project Python runtime (01-01)
- Relative-position observation (target - drone), not absolute position (01-02)
- Phase 1 rewards are zeros; Phase 2 adds shaping without changing obs/action interface (01-02)
- get_ang() for angular velocity — get_ang_vel() does not exist in Genesis 0.3.13 (01-02)
- gs.init() called once in caller (smoke_test.py / train.py), never inside CoordinateLandingEnv (01-02)

## Pending Todos

None.

## Blockers/Concerns

- SB3 VecEnv wrapper for Genesis batched envs needs research (Phase 3)

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 01    | 01   | 2min     | 3     | 7     |
| 01    | 02   | 2min     | 2     | 3     |

## Session Continuity

Last session: 2026-02-16
Stopped at: Completed 01-02-PLAN.md (CoordinateLandingEnv, smoke_test.py)
Resume file: .planning/phases/01-scaffolding-environment/01-02-SUMMARY.md
