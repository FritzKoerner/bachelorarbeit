"""
ObstacleAvoidanceEnv: rsl-rl v5.x compatible drone environment with obstacle avoidance.

Extends CoordinateLandingEnv with:
  - Random obstacle boxes between drone spawn area and target
  - Downward-facing depth camera (CNN input)
  - Distance-based obstacle collision detection
  - Obstacle proximity penalty reward
  - TensorDict observations: {"state": (n, 17), "depth": (n, 1, 64, 64)}

Constructor signature (rsl-rl style):
    env = ObstacleAvoidanceEnv(num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer)
    gs.init(...)
    env.build()
"""

import math
import copy

import numpy as np
import torch
from tensordict import TensorDict
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
    trans_quat_to_T,
)

from controllers.pid_controller import CascadingPIDController


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class ObstacleAvoidanceEnv:
    def __init__(self, num_envs: int, env_cfg: dict, obs_cfg: dict,
                 reward_cfg: dict, show_viewer: bool = False):
        self.num_envs = num_envs
        self.num_state_obs = obs_cfg["num_state_obs"]   # 17
        self.num_obs = self.num_state_obs                # runner reads shapes from TensorDict
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]        # 4
        self.device = gs.device

        self.dt = 0.01  # 100 Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.cfg = env_cfg
        self.show_viewer = show_viewer

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # Obstacle params
        self.num_obstacles = env_cfg.get("num_obstacles", 5)
        self.obstacle_size = env_cfg.get("obstacle_size", [1.0, 1.0, 2.0])
        self.collision_radius = env_cfg.get("collision_radius", 0.8)
        self.safety_radius = env_cfg.get("safety_radius", 3.0)

        # Depth camera params
        self.depth_res = obs_cfg.get("depth_res", 64)
        self.render_interval = env_cfg.get("render_interval", 2)
        self.max_depth = env_cfg.get("max_depth", 20.0)

        self._built = False

    # ------------------------------------------------------------------
    # Two-phase construction
    # ------------------------------------------------------------------

    def build(self):
        """Create Genesis scene with obstacles, depth camera, drone, PID, and buffers."""
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(7.0, 0.0, 5.0),
                camera_lookat=(3.0, 3.0, 3.0),
                camera_fov=60,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(10, self.num_envs)))
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=self.show_viewer,
        )

        scene.add_entity(gs.morphs.Plane())

        # Target visualization sphere
        if self.env_cfg.get("visualize_target", False):
            self.target_vis = scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.15,
                    fixed=True,
                    collision=False,
                    batch_fixed_verts=True,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5))
                ),
            )
        else:
            self.target_vis = None

        # Obstacle boxes
        ox, oy, oz = self.obstacle_size
        self.obstacles = []
        for _ in range(self.num_obstacles):
            obs_entity = scene.add_entity(
                morph=gs.morphs.Box(
                    size=(ox, oy, oz),
                    pos=(0.0, 0.0, oz / 2.0),  # base on ground
                    fixed=True,
                    collision=False,  # collision handled via distance check
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(0.8, 0.3, 0.2))
                ),
            )
            self.obstacles.append(obs_entity)

        # Downward-facing depth camera (before scene.build)
        self.depth_camera = scene.add_camera(
            res=(self.depth_res, self.depth_res),
            pos=(0, 0, 5),
            lookat=(0, 0, 0),
            fov=90,
            GUI=False,
        )

        # Drone
        self.drone = scene.add_entity(
            gs.morphs.Drone(
                file="assets/robots/draugas/draugas_genesis.urdf",
                pos=(0, 0, 3.0),
                euler=(0, 0, 0),
                propellers_link_name=["prop0_link", "prop1_link", "prop2_link", "prop3_link"],
                propellers_spin=[1, -1, 1, -1],
            )
        )

        env_spacing = self.env_cfg.get("env_spacing", 40.0)
        scene.build(n_envs=self.num_envs, env_spacing=(env_spacing, env_spacing))
        self.scene = scene

        # Drone base link for camera tracking
        self.drone_base_link = self.drone.get_link("base")

        # Camera offset transform: looking straight down, 10cm below drone
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Rotation.from_euler("zyx", [-90, -90, 0], degrees=True).as_matrix()
        T[2, 3] = -0.1
        self.camera_offset_T = torch.as_tensor(T, dtype=gs.tc_float, device=gs.device)

        # Initial orientation
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # PID controller
        self.controller = CascadingPIDController(
            drone=self.drone,
            dt=self.dt,
            base_rpm=self.env_cfg["pid_params"]["base_rpm"],
            max_rpm=self.env_cfg["pid_params"]["max_rpm"],
            pid_params=self.env_cfg["pid_params"],
            n_envs=self.num_envs,
            device=gs.device,
        )

        # Reward scaling: per-step rewards *= dt
        per_step_rewards = {"distance", "time", "obstacle_proximity"}
        for name in self.reward_scales:
            if name in per_step_rewards:
                self.reward_scales[name] *= self.dt

        self.reward_functions = {
            name: getattr(self, "_reward_" + name)
            for name in self.reward_scales
        }
        self.episode_sums = {
            name: torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
            for name in self.reward_scales
        }

        # Observation buffers
        self.state_buf = torch.zeros(
            (self.num_envs, self.num_state_obs), device=gs.device, dtype=gs.tc_float
        )
        self.depth_buf = torch.zeros(
            (self.num_envs, 1, self.depth_res, self.depth_res), device=gs.device, dtype=gs.tc_float
        )

        # Reward / reset buffers
        self.rew_buf            = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf          = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        # Action buffers
        self.actions      = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # State buffers
        self.base_pos       = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos  = torch.zeros_like(self.base_pos)
        self.base_quat      = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel   = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel   = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        # Target
        self.target_pos   = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.rel_pos      = torch.zeros_like(self.base_pos)
        self.last_rel_pos = torch.zeros_like(self.base_pos)

        # Obstacle positions: (n_envs, n_obstacles, 3) — center of each box
        self.obstacle_positions = torch.zeros(
            (self.num_envs, self.num_obstacles, 3), device=gs.device, dtype=gs.tc_float
        )

        # Obstacle collision / proximity
        self.obstacle_collision = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        self.min_obstacle_dist  = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # Hover counter
        self.hover_counter = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.global_step = 0
        self.extras = {}
        self._built = True

        self.reset()

    # ------------------------------------------------------------------
    # Camera tracking
    # ------------------------------------------------------------------

    def _update_camera_pose(self):
        """Move downward-facing camera to follow drone position."""
        link_pos = self.drone_base_link.get_pos()    # (n_envs, 3)
        link_quat = self.drone_base_link.get_quat()  # (n_envs, 4)
        link_T = trans_quat_to_T(link_pos, link_quat)           # (n_envs, 4, 4)
        world_T = torch.matmul(link_T, self.camera_offset_T)    # (n_envs, 4, 4)
        self.depth_camera.set_pose(transform=world_T)

    # ------------------------------------------------------------------
    # rsl-rl interface
    # ------------------------------------------------------------------

    def step(self, actions):
        self.actions = torch.clip(actions, -1.0, 1.0)

        scales = self.env_cfg["action_scales"]
        target_x   = self.base_pos[:, 0] + self.actions[:, 0] * scales[0]
        target_y   = self.base_pos[:, 1] + self.actions[:, 1] * scales[1]
        target_z   = self.base_pos[:, 2] + self.actions[:, 2] * scales[2]
        target_yaw = self.actions[:, 3] * 180.0
        target_pos = torch.stack([target_x, target_y, target_z], dim=-1)

        rpms = self.controller.update(target_pos, target_yaw)
        self.drone.set_propellels_rpm(rpms)
        if self.target_vis is not None:
            self.target_vis.set_pos(self.target_pos, zero_velocity=True)
        self.scene.step()

        # Update state
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:]      = self.drone.get_pos()
        self.rel_pos          = self.target_pos - self.base_pos
        self.last_rel_pos     = self.target_pos - self.last_base_pos
        self.base_quat[:]     = self.drone.get_quat()

        base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True, degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # Obstacle distance check
        # base_pos: (n_envs, 3), obstacle_positions: (n_envs, n_obs, 3)
        obs_dists = torch.norm(
            self.base_pos.unsqueeze(1) - self.obstacle_positions, dim=-1
        )  # (n_envs, n_obs)
        self.min_obstacle_dist = obs_dists.min(dim=1).values
        self.obstacle_collision = self.min_obstacle_dist < self.collision_radius

        # Hover counter
        close_enough = torch.norm(self.rel_pos, dim=1) < self.env_cfg["hover_radius"]
        slow_enough  = torch.norm(self.base_lin_vel, dim=1) < self.env_cfg["success_vel_threshold"]
        near_target  = close_enough & slow_enough
        self.hover_counter[ near_target] += 1
        self.hover_counter[~near_target]  = 0

        # Termination
        self.crash_condition = (
            (self.base_pos[:, 2] < 0.2)
            | (torch.abs(base_euler[:, 0]) > 60.0)
            | (torch.abs(base_euler[:, 1]) > 60.0)
            | (torch.norm(self.rel_pos, dim=1) > 50.0)
            | self.obstacle_collision
        )
        self.success_condition = self.hover_counter >= self.env_cfg["hover_steps"]
        timeout_condition = self.episode_length_buf > self.max_episode_length

        self.reset_buf = timeout_condition | self.crash_condition | self.success_condition

        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, dtype=gs.tc_float)
        self.extras["time_outs"][timeout_condition] = 1.0

        self.global_step += 1

        # Rewards BEFORE reset
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Auto-reset AFTER rewards, BEFORE obs
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # Observations (post-reset)
        obs = self._compute_obs()
        self.last_actions[:] = self.actions[:]
        return obs, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        n = len(envs_idx)

        # Randomize drone spawn
        offset = self.env_cfg["spawn_offset"]
        sx = gs_rand_float(-offset, offset, (n,), gs.device)
        sy = gs_rand_float(-offset, offset, (n,), gs.device)
        sz = gs_rand_float(
            self.env_cfg["spawn_height_min"],
            self.env_cfg["spawn_height_max"],
            (n,), gs.device,
        )
        spawn_pos = torch.stack([sx, sy, sz], dim=-1)

        # Randomize target
        curriculum_steps = self.env_cfg.get("curriculum_steps", 0)
        if self.global_step < curriculum_steps:
            r = self.env_cfg.get("curriculum_radius", 1.0)
            tx = sx + gs_rand_float(-r, r, (n,), gs.device)
            ty = sy + gs_rand_float(-r, r, (n,), gs.device)
            tz = sz + gs_rand_float(-r, r, (n,), gs.device)
            tz = torch.clamp(tz, min=0.2)
        else:
            tx = gs_rand_float(*self.env_cfg["target_x_range"], (n,), gs.device)
            ty = gs_rand_float(*self.env_cfg["target_y_range"], (n,), gs.device)
            tz = gs_rand_float(*self.env_cfg["target_z_range"], (n,), gs.device)
        self.target_pos[envs_idx] = torch.stack([tx, ty, tz], dim=-1)

        # Randomize obstacles — scatter between spawn area and target
        ox_range = self.env_cfg.get("obstacle_x_range", [-8.0, 12.0])
        oy_range = self.env_cfg.get("obstacle_y_range", [-8.0, 12.0])
        oz_val = self.obstacle_size[2] / 2.0  # half-height so base sits on ground

        # Curriculum: fewer obstacles early
        curriculum_n = self.env_cfg.get("curriculum_n_obstacles", 0)
        active_obstacles = curriculum_n if self.global_step < curriculum_steps else self.num_obstacles

        for i, obs_entity in enumerate(self.obstacles):
            if i < active_obstacles:
                new_x = gs_rand_float(*ox_range, (n,), gs.device)
                new_y = gs_rand_float(*oy_range, (n,), gs.device)
                new_z = torch.full((n,), oz_val, device=gs.device)
            else:
                # Move inactive obstacles far away
                new_x = torch.zeros(n, device=gs.device)
                new_y = torch.zeros(n, device=gs.device)
                new_z = torch.full((n,), -100.0, device=gs.device)

            new_pos = torch.stack([new_x, new_y, new_z], dim=-1)
            obs_entity.set_pos(new_pos, envs_idx=envs_idx, zero_velocity=True)
            self.obstacle_positions[envs_idx, i] = new_pos

        # Set drone state
        self.base_pos[envs_idx]      = spawn_pos
        self.last_base_pos[envs_idx] = spawn_pos
        self.base_quat[envs_idx]     = self.base_init_quat.reshape(1, -1)

        self.drone.set_pos(spawn_pos, zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        self.rel_pos      = self.target_pos - self.base_pos
        self.last_rel_pos = self.target_pos - self.last_base_pos

        self.controller.reset_idx(envs_idx)

        self.last_actions[envs_idx]       = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True
        self.hover_counter[envs_idx]      = 0

        # Log episode stats
        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.get_observations()

    def get_observations(self) -> TensorDict:
        return self._compute_obs()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _compute_obs(self) -> TensorDict:
        s = self.obs_scales

        # State vector (17-dim, identical to CoordinateLandingEnv)
        self.state_buf[:] = torch.cat(
            [
                torch.clip(self.rel_pos     * s["rel_pos"],    -1, 1),   # 3
                self.base_quat,                                           # 4
                torch.clip(self.base_lin_vel * s["lin_vel"],   -1, 1),   # 3
                torch.clip(self.base_ang_vel * s["ang_vel"],   -1, 1),   # 3
                self.last_actions,                                        # 4
            ],
            dim=-1,
        )  # total: 17

        # Depth image — render every render_interval steps, reuse cached buffer otherwise
        if self.episode_length_buf[0] % self.render_interval == 0:
            self._update_camera_pose()
            _, depth, _, _ = self.depth_camera.render(depth=True, segmentation=False)
            # depth: (n_envs, H, W) for batched, or (H, W) for single env
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
            self.depth_buf[:, 0] = torch.clamp(depth / self.max_depth, 0.0, 1.0)

        return TensorDict({
            "state": self.state_buf.clone(),
            "depth": self.depth_buf.clone(),
        }, batch_size=[self.num_envs])

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _reward_distance(self):
        """Penalty proportional to distance from target."""
        return torch.norm(self.rel_pos, dim=1)

    def _reward_time(self):
        return torch.ones(self.num_envs, device=gs.device, dtype=gs.tc_float)

    def _reward_obstacle_proximity(self):
        """Penalty when within safety_radius of any obstacle.
        Linearly increases as drone gets closer: max(0, 1 - dist/safety_radius)."""
        proximity = torch.clamp(1.0 - self.min_obstacle_dist / self.safety_radius, min=0.0)
        return proximity

    def _reward_crash(self):
        """Penalty on crash (including obstacle collision)."""
        rew = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        # Crash but NOT obstacle collision — that has its own reward
        non_obs_crash = self.crash_condition & ~self.obstacle_collision
        rew[non_obs_crash] = 1.0
        return rew

    def _reward_obstacle_collision(self):
        """Penalty on obstacle collision."""
        rew = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        rew[self.obstacle_collision] = 1.0
        return rew

    def _reward_success(self):
        """Reward on success."""
        rew = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        rew[self.success_condition] = 1.0
        return rew
