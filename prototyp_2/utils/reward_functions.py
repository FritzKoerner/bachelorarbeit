"""
Reward functions for vineyard landing task.
"""

import torch


class LandingRewardFunction:
    """
    Multi-component reward for vineyard landing task.

    Components:
    1. descent_reward: Encourage descending (negative height change)
    2. velocity_reward: Smooth descent velocity
    3. semantic_reward: Penalize hovering over vineyard (label 2), reward soil (label 1)
    4. landing_bonus: Big reward for safe landing on soil
    5. crash_penalty: Penalize crashes
    6. action_smoothness: Penalize jerky actions
    """

    # Semantic labels
    LABEL_BACKGROUND = 0
    LABEL_SOIL = 1
    LABEL_VINEYARD = 2

    def __init__(self, config: dict, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Reward weights
        self.w_descent = config.get('w_descent', 1.0)
        self.w_velocity = config.get('w_velocity', 0.5)
        self.w_semantic = config.get('w_semantic', 2.0)
        self.w_landing = config.get('w_landing', 100.0)
        self.w_crash = config.get('w_crash', -50.0)
        self.w_smooth = config.get('w_smooth', 0.1)

        # Thresholds
        self.landing_height = config.get('landing_height', 0.3)
        self.safe_landing_vel = config.get('safe_landing_vel', 0.5)
        self.crash_vel = config.get('crash_vel', 2.0)
        self.crash_angle = config.get('crash_angle', 0.5)  # radians (~28 deg)

        self.prev_actions = None

    def reset(self, env_ids: torch.Tensor = None):
        """Reset reward function state for specified environments."""
        if env_ids is None or self.prev_actions is None:
            self.prev_actions = None
        else:
            # Keep prev_actions for non-reset environments
            pass

    def compute(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        quat: torch.Tensor,
        semantic_seg: torch.Tensor,
        actions: torch.Tensor,
        prev_pos: torch.Tensor,
    ):
        """
        Compute total reward.

        Args:
            pos: (n_envs, 3) current position
            vel: (n_envs, 3) current velocity
            quat: (n_envs, 4) quaternion [w, x, y, z]
            semantic_seg: (n_envs, H, W) semantic labels (0/1/2)
            actions: (n_envs, 4) current actions
            prev_pos: (n_envs, 3) previous position

        Returns:
            rewards: (n_envs,) total reward per environment
            info: dict with reward component means
        """
        n_envs = pos.shape[0]
        rewards = torch.zeros(n_envs, device=self.device)

        # 1. Descent reward: encourage altitude decrease
        height_change = pos[:, 2] - prev_pos[:, 2]
        descent_reward = -height_change  # Positive when descending
        descent_reward = torch.clamp(descent_reward, -1, 1)
        rewards = rewards + self.w_descent * descent_reward

        # 2. Velocity reward: prefer smooth descent around -0.5 m/s
        vertical_vel = vel[:, 2]
        target_descent_vel = -0.5
        vel_error = torch.abs(vertical_vel - target_descent_vel)
        velocity_reward = torch.exp(-vel_error)
        rewards = rewards + self.w_velocity * velocity_reward

        # 3. Semantic reward: penalize hovering over vineyard, reward soil
        # Look at center region of image (where drone is roughly above)
        h, w = semantic_seg.shape[1], semantic_seg.shape[2]
        center_h, center_w = h // 4, w // 4
        center_region = semantic_seg[
            :,
            h // 2 - center_h : h // 2 + center_h,
            w // 2 - center_w : w // 2 + center_w,
        ]

        vineyard_ratio = (center_region == self.LABEL_VINEYARD).float().mean(dim=(1, 2))
        soil_ratio = (center_region == self.LABEL_SOIL).float().mean(dim=(1, 2))

        semantic_reward = soil_ratio - vineyard_ratio
        rewards = rewards + self.w_semantic * semantic_reward

        # 4. Landing bonus
        height = pos[:, 2]
        speed = torch.norm(vel, dim=-1)

        is_low = height < self.landing_height
        is_slow = speed < self.safe_landing_vel
        is_over_soil = soil_ratio > 0.5

        successful_landing = is_low & is_slow & is_over_soil
        rewards = rewards + self.w_landing * successful_landing.float()

        # 5. Crash penalty
        roll, pitch = self._quat_to_rp(quat)
        is_tilted = (torch.abs(roll) > self.crash_angle) | (torch.abs(pitch) > self.crash_angle)
        is_fast = speed > self.crash_vel

        crashed = is_low & (is_fast | is_tilted)
        rewards = rewards + self.w_crash * crashed.float()

        # 6. Action smoothness
        if self.prev_actions is not None and self.prev_actions.shape[0] == n_envs:
            action_diff = torch.norm(actions - self.prev_actions, dim=-1)
            rewards = rewards - self.w_smooth * action_diff

        self.prev_actions = actions.clone()

        # Compute info dict
        info = {
            'reward/descent': descent_reward.mean().item(),
            'reward/velocity': velocity_reward.mean().item(),
            'reward/semantic': semantic_reward.mean().item(),
            'reward/soil_ratio': soil_ratio.mean().item(),
            'reward/vineyard_ratio': vineyard_ratio.mean().item(),
            'episode/landing': successful_landing.sum().item(),
            'episode/crash': crashed.sum().item(),
        }

        return rewards, info, successful_landing, crashed

    def _quat_to_rp(self, quat: torch.Tensor):
        """Extract roll and pitch from quaternion."""
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1, 1))

        return roll, pitch
