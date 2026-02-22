"""
CNN encoder for visual observations (depth + semantic segmentation).
"""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """
    CNN encoder for depth + semantic segmentation input.

    Input: (batch, 2, 64, 64) - depth + semantic channels
    Output: (batch, feature_dim) - flattened visual features
    """

    def __init__(self, input_channels: int = 2, feature_dim: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute output size for 64x64 input:
        # 64 -> (64-8)/4+1 = 15 -> (15-4)/2+1 = 6 -> (6-3)/1+1 = 4
        # Output: 64 * 4 * 4 = 1024
        conv_output_dim = 64 * 4 * 4

        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim, feature_dim),
            nn.ReLU(),
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 2, H, W) visual input

        Returns:
            features: (batch, feature_dim)
        """
        features = self.conv(x)
        return self.fc(features)
