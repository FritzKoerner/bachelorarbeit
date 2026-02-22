"""
Actor-Critic network for PPO with vision + state fusion.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from .cnn_encoder import VisionEncoder


class ActorCritic(nn.Module):
    """
    Actor-Critic network with vision + state fusion.

    Inputs:
        visual: (batch, 2, 64, 64) depth + semantic
        state: (batch, 13) pos, vel, quat, ang_vel

    Outputs:
        action_mean: (batch, 4) velocity commands
        value: (batch,) state value
    """

    def __init__(
        self,
        state_dim: int = 13,
        action_dim: int = 4,
        hidden_dim: int = 256,
        visual_feature_dim: int = 256,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            input_channels=2,
            feature_dim=visual_feature_dim,
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion MLP
        fusion_dim = visual_feature_dim + 64
        self.shared = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

        # Small init for actor output (more stable initial policy)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, visual: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute shared features.

        Args:
            visual: (batch, 2, H, W) visual observations
            state: (batch, state_dim) state vector

        Returns:
            shared_features: (batch, hidden_dim)
        """
        visual_features = self.vision_encoder(visual)
        state_features = self.state_encoder(state)

        combined = torch.cat([visual_features, state_features], dim=-1)
        return self.shared(combined)

    def act(
        self,
        visual: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Sample action from policy.

        Args:
            visual: (batch, 2, H, W) visual observations
            state: (batch, state_dim) state vector
            deterministic: if True, return mean action

        Returns:
            action: (batch, action_dim)
            log_prob: (batch,) or None if deterministic
        """
        features = self.forward(visual, state)
        action_mean = self.actor_mean(features)

        if deterministic:
            return action_mean, None

        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate(
        self,
        visual: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        """
        Evaluate actions for PPO update.

        Args:
            visual: (batch, 2, H, W) visual observations
            state: (batch, state_dim) state vector
            action: (batch, action_dim) actions to evaluate

        Returns:
            log_prob: (batch,)
            value: (batch,)
            entropy: (batch,)
        """
        features = self.forward(visual, state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        value = self.critic(features).squeeze(-1)

        return log_prob, value, entropy

    def get_value(self, visual: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate.

        Args:
            visual: (batch, 2, H, W) visual observations
            state: (batch, state_dim) state vector

        Returns:
            value: (batch,)
        """
        features = self.forward(visual, state)
        return self.critic(features).squeeze(-1)
