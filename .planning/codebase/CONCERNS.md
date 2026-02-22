# Codebase Concerns

**Analysis Date:** 2026-02-12

## Tech Debt

**Prototype Code Not Integrated:**
- Issue: Complete prototype implementation exists in `prototyp/` directory (train.py, env.py, network.py, etc.) alongside production code, creating duplicate implementations and maintenance burden
- Files: `prototyp/train.py`, `prototyp/env.py`, `prototyp/network.py`, `prototyp/scene_builder.py`, `prototyp/rewards.py`, `prototyp/observations.py`, `prototyp/actions.py`, `prototyp/utils.py`
- Impact: Code duplication increases chance of bugs existing in one version but not the other; unclear which implementation should be used; future changes must be synchronized across both versions
- Fix approach: Migrate all useful patterns from `prototyp/` into production code in `envs/`, `networks/`, `utils/`, `controllers/`, remove prototype directory entirely

**No Error Handling in Critical Paths:**
- Issue: Camera rendering, scene stepping, and tensor operations have no try-catch blocks or error recovery
- Files: `envs/vineyard_landing_env.py` (lines 241-258 for `_get_depth()` and `_get_semantic_segmentation()`), `envs/scene_builder.py` (lines 52-123 for `build_scene()`)
- Impact: Any Genesis API failure or device/memory issue will crash entire training loop without graceful shutdown
- Fix approach: Add try-catch blocks around camera.render(), scene.step(), and device operations; implement fallback/skip behavior for rendering failures

**Unimplemented Reward Function Reset:**
- Issue: `LandingRewardFunction.reset()` in `utils/reward_functions.py` (lines 46-52) has no-op implementation with just `pass` statement
- Files: `utils/reward_functions.py` (lines 46-52)
- Impact: `prev_actions` state is not properly cleared for reset environments, causing action smoothness penalties to reference stale actions from different episodes; affects reward computation correctness
- Fix approach: Implement proper selective reset: `if env_ids is not None and self.prev_actions is not None: self.prev_actions[env_ids] = 0`

**Unused Return Type in Env.close():**
- Issue: `VineyardLandingEnv.close()` method body is just `pass` (line 278), provides no cleanup
- Files: `envs/vineyard_landing_env.py` (lines 276-278)
- Impact: No resource cleanup occurs; camera, scene, or physics objects may not be properly released
- Fix approach: Implement cleanup for scene resources, camera objects, and any open file handles

**Device Mismatch in Scene Builder:**
- Issue: `segmentation_to_semantic()` hardcodes device detection instead of using passed device parameter
- Files: `envs/scene_builder.py` (line 144): `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Impact: If environment was created on different device, segmentation conversion fails; breaks multi-GPU or CPU-only setups
- Fix approach: Add device parameter to method signature, pass from caller: `def segmentation_to_semantic(self, raw_segmentation, scene, entity_to_label, device)`

## Known Bugs

**Camera Start/Stop Not Idempotent:**
- Symptoms: Multiple calls to `camera.start()` and `camera.stop()` each step (lines 241-244, 255-258 in `vineyard_landing_env.py`)
- Files: `envs/vineyard_landing_env.py` lines 241-244 (`_get_depth()`), lines 255-258 (`_get_semantic_segmentation()`)
- Trigger: Normal operation - every environment step calls `_get_obs()` which calls both methods
- Workaround: Combine depth + segmentation into single render call: `self.camera.render(depth=True, segmentation=True)` returns both in one operation
- Root cause: Genesis API allows per-channel rendering but overhead of start/stop suggests batching is inefficient

**Quaternion-to-Roll-Pitch Conversion Duplicated:**
- Symptoms: Identical implementation exists in two files with no shared utility
- Files: `utils/reward_functions.py` (lines 149-163), `controllers/low_level_controller.py` (lines 130-154)
- Trigger: Both reward computation and controller use quaternion attitude estimation
- Workaround: Already working correctly, but represents code duplication risk
- Fix approach: Move to shared utility in `utils/` and import from both locations

**Segmentation Index Mapping Fragile:**
- Symptoms: Segmentation label conversion depends on `scene.segmentation_idx_dict` structure which is not documented
- Files: `envs/scene_builder.py` (lines 156-169)
- Trigger: If Genesis API changes segmentation index format (dict layout, tuple structure, key types), mapping fails silently (unmapped indices get label 0)
- Workaround: Verify segmentation_idx_dict structure in unit test at scene build time
- Root cause: Genesis Plugin API not fully documented for this internal structure

## Performance Bottlenecks

**Camera Rendering Per Step:**
- Problem: Semantic segmentation rendering occurs every step (line 172 in `vineyard_landing_env.py`) for reward computation, adding overhead
- Files: `envs/vineyard_landing_env.py` (lines 171-172), `utils/reward_functions.py` (lines 96-108)
- Cause: Center-region semantic analysis requires full camera render; cannot batch or cache between steps
- Current performance: 64 parallel environments × 32 steps per rollout = 2048 renders per training iteration
- Improvement path: (1) Cache segmentation if drone position/orientation unchanged between steps, (2) Use depth only + heuristic for soil detection (depth gradient analysis), (3) Render at lower frequency (every N steps)

**Tensor Device Movement in Rollout Collection:**
- Problem: Line 91-100 in `train.py` uses `.item()` to extract scalars from GPU tensors for statistics logging
- Files: `train.py` (lines 91-92, 98-100)
- Cause: Frequent GPU-to-CPU transfers in tight rollout loop; Python list appends have overhead
- Impact: Measurable when running large batch sizes (64+ envs) with small rollout steps (32)
- Improvement path: Accumulate statistics as tensors, reduce to CPU once per logging interval, not per episode

**PPO Mini-Batch Shuffling with numpy:**
- Problem: Line 127 in `algorithms/ppo.py` uses `np.random.shuffle(indices)` which is CPU-only and non-differentiable
- Files: `algorithms/ppo.py` (lines 123, 127-131)
- Impact: Not a bottleneck for small batches but prevents GPU-native data sampling in future optimizations
- Improvement path: Use `torch.randperm()` for consistent GPU acceleration

## Fragile Areas

**Coordinate System Assumptions:**
- Files: `envs/scene_builder.py` (lines 52-123), `controllers/low_level_controller.py` (lines 50-128)
- Why fragile: Code assumes Z-up coordinate system, quaternion format [w,x,y,z], Euler angles in degrees (scipy extrinsic x-y-z). These are Genesis conventions but not universally standard
- Safe modification: Any change to coordinate frames requires updates across: scene builder (line 98 euler angles), controller (lines 149-152 quat→rp), reward function (line 122 roll/pitch calculation), environment reset (line 96-97 yaw calculation)
- Test coverage: No unit tests verify coordinate transformations; only integration test is visual inspection in viewer
- Recommendation: Add dedicated test for known rotation cases (90° yaw, 45° pitch, etc.) with hard-coded expected values

**Landing Condition Boolean Logic:**
- Files: `utils/reward_functions.py` (lines 114-127)
- Why fragile: Landing and crash conditions use multiple intersecting boolean gates (height, velocity, angle, terrain)
- Safe modification: Changes to thresholds (landing_height 0.3m, safe_landing_vel 0.5 m/s, crash_vel 2.0 m/s, crash_angle 0.5 rad) affect reward signal shape and success rates unpredictably
- Test coverage: No unit tests isolate each condition; only empirical measurement during training
- Recommendation: Add tests for edge cases like: height=0.301m with speed=0.49 m/s, roll=0.49 rad - verify classification matches intent

**Semantic Segmentation Mapping Edge Cases:**
- Files: `envs/scene_builder.py` (lines 127-171)
- Why fragile: Mapping relies on iterating `scene.segmentation_idx_dict`, but undefined behavior if: (1) entity_info is neither tuple nor scalar, (2) empty tuple passed, (3) multiple seg_idx map to same entity
- Safe modification: Verify Genesis returns expected segmentation_idx_dict structure before using; log warnings if unmapped indices remain (suggest they get default label 0)
- Test coverage: No unit test for segmentation_to_semantic with known input
- Recommendation: Add test with manually constructed seg_dict and verify output labels

## Missing Critical Features

**No Velocity/State Normalization:**
- Problem: State observations include raw values (position, velocity, quaternion, angular velocity) with very different scales
- Files: `envs/vineyard_landing_env.py` (lines 226-232), `networks/actor_critic.py` (lines 45-49)
- Blocks: Poor neural network learning due to input feature imbalance; position in meters (0-50 range) dominates over quaternion components (±1 range)
- Recommendation: Normalize state vector before feeding to network - subtract mean, divide by std (compute from initial random rollouts or fixed ranges)

**No Validation of Genesis URDF Path:**
- Problem: Drone URDF path hardcoded as `"assets/robots/draugas/draugas_genesis.urdf"` with no existence check
- Files: `envs/scene_builder.py` (lines 94-99)
- Blocks: Training fails with cryptic Genesis error if file not found; no helpful error message
- Recommendation: Add path validation in `build_scene()` before passing to Genesis

**No Checkpoint Resume Logic:**
- Problem: `train.py` accepts `--checkpoint` argument but never uses it
- Files: `train.py` (lines 242-243, 247-253) - argument parsed but not applied
- Blocks: Cannot resume interrupted training runs; must restart from epoch 0
- Recommendation: Implement logic to load checkpoint, resume from iteration, continue optimizer state

**No Logging of Network Architecture:**
- Problem: Policy network complexity not logged; easy to silently change hidden_dim without documentation
- Files: `networks/actor_critic.py` (lines 26-50), `train.py` (lines 260-263) - config printed but not network details
- Blocks: Reproducibility - checkpoint files don't capture exact architecture (only weights)
- Recommendation: Log network summary (num parameters, layer types) at training start

## Test Coverage Gaps

**No Unit Tests:**
- What's not tested: Reward function components (descent/velocity/semantic rewards computed independently), controller RPM mixing (motor formulas), quaternion conversions, environment reset behavior
- Files: All core files lack corresponding `test_*.py` files
- Risk: Manual refactoring of equations (e.g., yaw-rate mixing in lines 120-123 of controller) can introduce bugs with no detection until training diverges
- Priority: HIGH - Add tests for: (1) reward function component isolation, (2) controller motor mixing with known inputs, (3) quaternion→rp→quat round-trip

**No Integration Test for Multi-Environment Reset:**
- What's not tested: Auto-reset of done environments during rollout collection
- Files: `train.py` (lines 89-94), `envs/vineyard_landing_env.py` (lines 204-207)
- Risk: Episode statistics tracking may count resets incorrectly; episode boundaries might misalign with advantage calculation
- Priority: MEDIUM - Add integration test: collect 1 rollout, verify all episodes reset exactly once, statistics match expected counts

**No Segmentation Rendering Validation:**
- What's not tested: Semantic labels actually match scene geometry
- Files: `envs/scene_builder.py` (lines 74-91 scene layout), `envs/vineyard_landing_env.py` (lines 253-265)
- Risk: If soil/vineyard strips renderedincorrectly, reward shaping is fundamentally broken and agent learns wrong strategy
- Priority: MEDIUM - Add test: render at known positions over known strip, verify correct label percentage

**No Device Consistency Tests:**
- What's not tested: Operations work identically on CPU and GPU
- Files: All CUDA device usage (scattered `.to(device)`, `.device` checks)
- Risk: Training works on GPU development machine but fails in cloud CPU environment
- Priority: LOW - Add parameterized tests for CPU/GPU with n_envs=1

## Scaling Limits

**Parallel Environment Scaling Unclear:**
- Current capacity: Default n_envs=64 in config; tested with 64 in training
- Limit: Unknown - likely hits either (1) GPU memory from 64× depth+seg renderings, (2) CPU physics bottleneck at scene.step()
- Scaling path: (1) Profile memory usage per environment, (2) Test with n_envs=128, 256 to find inflection point, (3) Consider rendering optimization (lower res cameras, skip frames)

**Episode Length Constraint:**
- Current: max_episode_steps=500 (configured in training_config.yaml)
- Limit: Longer episodes may exceed available GPT memory (values/advantages tensors pre-allocated), or exceed practical training time
- Scaling path: If 500 steps insufficient for good landing behavior, either (1) increase rollout buffer size, (2) implement recurrent policy (LSTM), (3) use trajectory sampling instead of full episode storage

## Dependencies at Risk

**Genesis World Plugin API Dependency:**
- Risk: Custom plugin at `~/.claude/plugins/genesis-world/` - not version controlled in main repo; undocumented API surface
- Impact: Breaking changes in plugin (segmentation_idx_dict structure, camera.render() signature) instantly break training
- Migration plan: (1) Document Genesis API contracts (quaternion format, coordinate conventions, rendering output shapes), (2) Add compatibility shim layer, (3) Pin Genesis version in requirements.txt with exact version check

**Hard Dependency on Genesis for Scene Building:**
- Risk: Cannot test environment logic without Genesis simulator running
- Impact: Unit tests impossible without Genesis installation; CI/CD requires GPU or CPU backend initialization
- Migration plan: (1) Extract scene configuration to data structure (JSON/YAML), (2) Mock Genesis entities for testing, (3) Separate "scene specification" from "Genesis physics execution"

## Security Considerations

**URDF File Loading Without Validation:**
- Risk: Drone URDF loaded from `assets/robots/draugas/` with no checksum/signature verification
- Files: `envs/scene_builder.py` (line 96)
- Current mitigation: File in local assets directory, not downloaded from network
- Recommendations: (1) Add SHA256 checksum verification, (2) Document origin/provenance of draugas.urdf, (3) If ever loading URDF from network, validate signature

**TensorBoard Event Files Unencrypted:**
- Risk: Training logs saved to `./logs/` include reward values, episode statistics that could leak proprietary info
- Files: `algorithms/ppo.py` (lines 48-50, 180-183)
- Current mitigation: Local filesystem only
- Recommendations: (1) Document sensitivity level, (2) If sharing logs with external parties, filter/anonymize, (3) Consider adding optional AES encryption for log directory

---

*Concerns audit: 2026-02-12*
