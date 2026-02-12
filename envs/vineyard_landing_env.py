"""
Vineyard landing environment for RL training with Genesis simulator.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional

from envs.scene_builder import VineyardSceneBuilder
from controllers.low_level_controller import VelocityController
from utils.reward_functions import LandingRewardFunction


class VineyardLandingEnv:
    """
    Gym-like environment for drone landing in vineyard using Genesis.

    Observation Space:
        - visual: (2, 64, 64) depth + semantic segmentation
        - state: (13,) pos(3), vel(3), quat(4), ang_vel(3)

    Action Space:
        - (4,) continuous: [vx, vy, vz, yaw_rate] in m/s and rad/s

    Usage:
        env = VineyardLandingEnv(config, device)
        env.build()   # call after gs.init(), before reset()
        obs = env.reset()
    """

    def __init__(self, config: dict, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Environment parameters from nested config
        env_cfg = config['env']
        self.n_envs = env_cfg['n_envs']
        self.max_steps = env_cfg['max_episode_steps']
        self.image_width = env_cfg['image_width']
        self.image_height = env_cfg['image_height']

        # Spawn parameters
        spawn_cfg = config['spawn']
        self.spawn_height_min = spawn_cfg['height_min']
        self.spawn_height_max = spawn_cfg['height_max']
        self.spawn_x_range = spawn_cfg['x_range']
        self.spawn_y_range = spawn_cfg['y_range']

        # Action scaling from controller config
        ctrl_cfg = config['controller']
        self.vel_scale = ctrl_cfg.get('vel_scale', 2.0)
        self.yaw_scale = ctrl_cfg.get('yaw_scale', 1.0)

        # Observation and action dimensions (available before build())
        self.obs_visual_shape = (2, self.image_height, self.image_width)
        self.obs_state_dim = 13
        self.action_dim = 4

        self._built = False

    def build(self):
        """
        Build Genesis scene. Call after gs.init() and before reset().

        Constructs the scene, drone, camera, controller, and reward function.
        Must not be called before gs.init() in train.py.
        """
        self.scene_builder = VineyardSceneBuilder(self.config['scene'])
        self.scene, self.drone, self.camera, _ = self.scene_builder.build_scene(
            n_envs=self.n_envs,
            show_viewer=self.config['scene'].get('show_viewer', False),
        )

        ctrl_cfg = self.config['controller']
        self.controller = VelocityController(
            base_hover_rpm=ctrl_cfg['base_hover_rpm'],
            max_rpm=ctrl_cfg['max_rpm'],
            kp_vel=ctrl_cfg['kp_vel'],
            kd_vel=ctrl_cfg['kd_vel'],
            kp_att=ctrl_cfg['kp_att'],
            device=self.device,
        )

        self.reward_fn = LandingRewardFunction(self.config['reward'], device=self.device)

        # State tracking
        self.step_count = torch.zeros(self.n_envs, dtype=torch.int32, device=self.device)
        self.prev_pos = None

        self._built = True

    def reset(self, env_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Reset specified environments or all if env_ids is None.

        Args:
            env_ids: (n,) indices of environments to reset

        Returns:
            obs: dict with 'visual' and 'state' tensors
        """
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)

        n_reset = len(env_ids)

        # Random spawn positions
        spawn_x = (torch.rand(n_reset, device=self.device) - 0.5) * self.spawn_x_range
        spawn_y = (torch.rand(n_reset, device=self.device) - 0.5) * self.spawn_y_range
        spawn_z = (
            torch.rand(n_reset, device=self.device)
            * (self.spawn_height_max - self.spawn_height_min)
            + self.spawn_height_min
        )
        spawn_pos = torch.stack([spawn_x, spawn_y, spawn_z], dim=-1)

        # Random yaw orientation (level attitude)
        yaw = torch.rand(n_reset, device=self.device) * 2 * np.pi
        spawn_quat = self._yaw_to_quat(yaw)

        # Reset drone state for specified environments
        self.drone.set_pos(spawn_pos, envs_idx=env_ids.cpu().numpy())
        self.drone.set_quat(spawn_quat, envs_idx=env_ids.cpu().numpy())
        self.drone.set_vel(torch.zeros(n_reset, 3, device=self.device), envs_idx=env_ids.cpu().numpy())
        self.drone.set_ang_vel(torch.zeros(n_reset, 3, device=self.device), envs_idx=env_ids.cpu().numpy())

        # Reset step counter
        self.step_count[env_ids] = 0

        # Reset controller and reward function
        self.controller.reset()
        self.reward_fn.reset(env_ids)

        # Step physics once to settle
        self.scene.step()

        # Get initial observation
        obs = self._get_obs()

        # Store initial position for reward computation
        if self.prev_pos is None:
            self.prev_pos = self.drone.get_pos().clone()
        else:
            self.prev_pos[env_ids] = self.drone.get_pos()[env_ids].clone()

        return obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Step the environment.

        Args:
            actions: (n_envs, 4) actions in [-1, 1] range

        Returns:
            obs: dict with 'visual' and 'state'
            rewards: (n_envs,)
            dones: (n_envs,) bool
            info: dict with additional info
        """
        # Scale actions using config-driven scaling factors
        target_vel = torch.zeros(self.n_envs, 3, device=self.device)
        target_vel[:, 0] = actions[:, 0] * self.vel_scale
        target_vel[:, 1] = actions[:, 1] * self.vel_scale
        target_vel[:, 2] = actions[:, 2] * self.vel_scale
        target_yaw_rate = actions[:, 3] * self.yaw_scale

        # Get current state
        current_vel = self.drone.get_vel()
        current_quat = self.drone.get_quat()

        # Compute RPMs from velocity commands
        rpms = self.controller.compute_rpm(
            target_vel, target_yaw_rate, current_vel, current_quat
        )

        # Apply RPMs
        self.drone.set_propellels_rpm(rpms)

        # Step physics
        self.scene.step()

        # Increment step counter
        self.step_count += 1

        # Get new state
        pos = self.drone.get_pos()
        vel = self.drone.get_vel()
        quat = self.drone.get_quat()

        # Get observation (includes camera render)
        obs = self._get_obs()

        # Extract semantic segmentation from obs for reward computation
        semantic_seg = (obs['visual'][:, 1, :, :] * 2).long()  # reverse normalization

        # Compute rewards
        rewards, reward_info, landed, crashed = self.reward_fn.compute(
            pos=pos,
            vel=vel,
            quat=quat,
            semantic_seg=semantic_seg,
            actions=actions,
            prev_pos=self.prev_pos,
        )

        # Update prev_pos
        self.prev_pos = pos.clone()

        # Check termination
        timeout = self.step_count >= self.max_steps
        out_of_bounds = (torch.abs(pos[:, 0]) > 30) | (torch.abs(pos[:, 1]) > 10) | (pos[:, 2] < 0)
        dones = landed | crashed | timeout | out_of_bounds

        # Build info dict
        info = {
            'landed': landed,
            'crashed': crashed,
            'timeout': timeout,
            'out_of_bounds': out_of_bounds,
            **reward_info,
        }

        # Auto-reset done environments
        done_ids = torch.where(dones)[0]
        if len(done_ids) > 0:
            self.reset(done_ids)

        return obs, rewards, dones, info

    def _render_camera(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render depth + segmentation from drone camera.

        Uses the direct camera.render() pattern (not start/stop).

        Returns:
            depth: tensor, shape (H, W) for single env or (n_envs, H, W) for multi-env
            segmentation: tensor, shape (H, W) or (n_envs, H, W) raw segmentation indices
        """
        _, depth, segmentation, _ = self.camera.render(depth=True, segmentation=True)
        return depth, segmentation

    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """
        Get current observation.

        Returns:
            dict with:
              'visual': (n_envs, 2, H, W) float tensor -- channel 0 = depth, channel 1 = semantic
              'state': (n_envs, 13) float tensor -- pos, vel, quat, ang_vel
        """
        depth, seg_raw = self._render_camera()
        semantic = self.scene_builder.segmentation_to_semantic(seg_raw)

        # Convert to torch if numpy
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth.copy()).to(self.device)
        else:
            depth = depth.to(self.device)

        if isinstance(semantic, np.ndarray):
            semantic = torch.from_numpy(semantic.copy()).to(self.device)
        else:
            semantic = semantic.to(self.device)

        # Handle single-env vs multi-env render output
        # If camera returns (H, W), broadcast to all envs; if (n_envs, H, W), use directly
        if depth.ndim == 2:
            depth = depth.unsqueeze(0).expand(self.n_envs, -1, -1)
            semantic = semantic.unsqueeze(0).expand(self.n_envs, -1, -1)

        # Normalize depth to [0, 1] (clip to 20m range)
        depth = torch.clamp(depth.float(), 0, 20) / 20.0

        # Normalize semantic labels to [0, 1] (labels are 0/1/2, divide by 2)
        semantic_norm = semantic.float() / 2.0

        # Resize if camera resolution differs from env image size
        if depth.shape[-2] != self.image_height or depth.shape[-1] != self.image_width:
            import torch.nn.functional as F
            depth = F.interpolate(
                depth.unsqueeze(1), size=(self.image_height, self.image_width), mode='bilinear', align_corners=False
            ).squeeze(1)
            semantic_norm = F.interpolate(
                semantic_norm.unsqueeze(1), size=(self.image_height, self.image_width), mode='nearest'
            ).squeeze(1)

        # Stack channels: (n_envs, 2, H, W)
        visual = torch.stack([depth, semantic_norm], dim=1)

        # State observation
        pos = self.drone.get_pos()        # (n_envs, 3)
        vel = self.drone.get_vel()        # (n_envs, 3)
        quat = self.drone.get_quat()      # (n_envs, 4)
        ang_vel = self.drone.get_ang_vel()  # (n_envs, 3)
        state = torch.cat([pos, vel, quat, ang_vel], dim=-1)  # (n_envs, 13)

        return {'visual': visual, 'state': state}

    def _yaw_to_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        """Convert yaw angle to quaternion [w, x, y, z]."""
        half_yaw = yaw / 2
        w = torch.cos(half_yaw)
        x = torch.zeros_like(yaw)
        y = torch.zeros_like(yaw)
        z = torch.sin(half_yaw)
        return torch.stack([w, x, y, z], dim=-1)

    def close(self):
        """Clean up resources."""
        pass
