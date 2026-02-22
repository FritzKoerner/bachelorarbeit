"""
Low-level velocity controller for drone.
Converts high-level velocity commands to propeller RPM.
"""

import torch
import numpy as np


class VelocityController:
    """
    PD controller converting velocity commands to propeller RPM.

    Based on draugas URDF physical constants:
    - Base hover RPM: 1789.2 (URDF-derived: sqrt(mass * g / (n_rotors * kf)))
    - Max RPM: 25000

    Control hierarchy:
    1. Velocity error → desired acceleration
    2. Desired acceleration → desired thrust + attitude
    3. Attitude error → differential RPM
    """

    def __init__(
        self,
        base_hover_rpm: float = 1789.2,  # URDF-derived: sqrt(mass * g / (n_rotors * kf)) = sqrt(0.714 * 9.81 / (4 * 5.47e-07))
        max_rpm: float = 25000,
        kp_vel: float = 5.0,
        kd_vel: float = 1.0,
        kp_att: float = 10.0,
        mass: float = 0.714,
        gravity: float = 9.81,
        device: torch.device = None,
    ):
        self.base_hover_rpm = base_hover_rpm
        self.max_rpm = max_rpm
        self.kp_vel = kp_vel
        self.kd_vel = kd_vel
        self.kp_att = kp_att
        self.mass = mass
        self.gravity = gravity
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.prev_vel_error = None

    def reset(self):
        """Reset controller state."""
        self.prev_vel_error = None

    def compute_rpm(
        self,
        target_vel: torch.Tensor,
        target_yaw_rate: torch.Tensor,
        current_vel: torch.Tensor,
        current_quat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 4 propeller RPMs from velocity commands.

        Args:
            target_vel: (n_envs, 3) desired velocity [vx, vy, vz] in m/s
            target_yaw_rate: (n_envs,) desired yaw rate in rad/s
            current_vel: (n_envs, 3) current velocity
            current_quat: (n_envs, 4) current quaternion [w, x, y, z]

        Returns:
            rpms: (n_envs, 4) propeller RPMs
        """
        n_envs = target_vel.shape[0]

        # Velocity PD control
        vel_error = target_vel - current_vel

        if self.prev_vel_error is not None and self.prev_vel_error.shape[0] == n_envs:
            vel_error_deriv = vel_error - self.prev_vel_error
        else:
            vel_error_deriv = torch.zeros_like(vel_error)

        self.prev_vel_error = vel_error.clone()

        # Desired acceleration from velocity error
        desired_acc = self.kp_vel * vel_error + self.kd_vel * vel_error_deriv

        # Add gravity compensation
        desired_acc[:, 2] = desired_acc[:, 2] + self.gravity

        # Compute thrust magnitude
        thrust = self.mass * torch.norm(desired_acc, dim=-1, keepdim=True)

        # Thrust ratio relative to hover
        hover_thrust = self.mass * self.gravity
        thrust_ratio = thrust.squeeze(-1) / hover_thrust

        # Base RPM adjusted by thrust (thrust ~ RPM^2)
        base_rpms = self.base_hover_rpm * torch.sqrt(torch.clamp(thrust_ratio, min=0.1))

        # Desired roll/pitch from horizontal acceleration
        ax = desired_acc[:, 0]
        ay = desired_acc[:, 1]
        az = torch.clamp(desired_acc[:, 2], min=0.1)

        desired_pitch = torch.atan2(ax, az)
        desired_roll = torch.atan2(-ay, az)

        # Get current roll/pitch from quaternion
        current_roll, current_pitch = self._quat_to_rp(current_quat)

        # Attitude error
        roll_error = desired_roll - current_roll
        pitch_error = desired_pitch - current_pitch

        # Differential RPM for attitude control
        roll_rpm = self.kp_att * roll_error * 100
        pitch_rpm = self.kp_att * pitch_error * 100
        yaw_rpm = target_yaw_rate * 50

        # Mix to individual motors (X configuration)
        # Motor order assumed: FR, FL, BL, BR
        rpms = torch.zeros(n_envs, 4, device=self.device)
        rpms[:, 0] = base_rpms + roll_rpm - pitch_rpm + yaw_rpm  # FR
        rpms[:, 1] = base_rpms - roll_rpm - pitch_rpm - yaw_rpm  # FL
        rpms[:, 2] = base_rpms - roll_rpm + pitch_rpm + yaw_rpm  # BL
        rpms[:, 3] = base_rpms + roll_rpm + pitch_rpm - yaw_rpm  # BR

        # Clip to valid range
        rpms = torch.clamp(rpms, 0, self.max_rpm)

        return rpms

    def _quat_to_rp(self, quat: torch.Tensor):
        """
        Extract roll and pitch from quaternion.

        Args:
            quat: (n_envs, 4) quaternion [w, x, y, z]

        Returns:
            roll, pitch: (n_envs,) each
        """
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1, 1))

        return roll, pitch
