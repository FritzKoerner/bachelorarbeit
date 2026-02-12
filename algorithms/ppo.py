"""
PPO (Proximal Policy Optimization) trainer for parallel Genesis environments.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict
from torch.utils.tensorboard import SummaryWriter


class PPOTrainer:
    """
    PPO trainer for parallel Genesis environments.

    Features:
    - Batched rollout collection
    - GAE advantage estimation
    - Multi-modal observations (visual + state)
    - GPU-accelerated training
    - Per-mini-batch advantage normalization (correct statistical behavior)
    """

    def __init__(
        self,
        policy: nn.Module,
        config: dict,
        device: torch.device = None,
    ):
        self.policy = policy
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Support both nested config (config['ppo']) and flat config (backwards compat)
        ppo_cfg = config.get('ppo', config)

        # PPO hyperparameters
        self.lr = ppo_cfg.get('learning_rate', 3e-4)
        self.gamma = ppo_cfg.get('gamma', 0.99)
        self.lambda_ = ppo_cfg.get('gae_lambda', 0.95)
        self.clip_epsilon = ppo_cfg.get('clip_epsilon', 0.2)
        self.value_coef = ppo_cfg.get('value_coef', 0.5)
        self.entropy_coef = ppo_cfg.get('entropy_coef', 0.01)
        self.max_grad_norm = ppo_cfg.get('max_grad_norm', 0.5)
        self.n_epochs = ppo_cfg.get('ppo_epochs', 10)
        self.batch_size = ppo_cfg.get('batch_size', 64)

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)

        # Logging -- support nested config['logging'] or flat config
        log_cfg = config.get('logging', config)
        log_dir = log_cfg.get('log_dir', './runs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ):
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: (n_steps, n_envs)
            values: (n_steps, n_envs)
            dones: (n_steps, n_envs)
            last_value: (n_envs,)

        Returns:
            returns: (n_steps, n_envs)
            advantages: (n_steps, n_envs)
        """
        n_steps, n_envs = rewards.shape

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(n_envs, device=self.device)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            next_non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.lambda_ * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return returns, advantages

    def update(self, rollouts: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        PPO update step.

        Args:
            rollouts: dict containing:
                - visuals: (n_steps, n_envs, C, H, W)
                - states: (n_steps, n_envs, state_dim)
                - actions: (n_steps, n_envs, action_dim)
                - log_probs: (n_steps, n_envs)
                - returns: (n_steps, n_envs)
                - advantages: (n_steps, n_envs)

        Returns:
            losses: dict with loss values
        """
        n_steps, n_envs = rollouts['rewards'].shape
        batch_size_total = n_steps * n_envs

        # Flatten all tensors
        visuals = rollouts['visuals'].view(batch_size_total, *rollouts['visuals'].shape[2:])
        states = rollouts['states'].view(batch_size_total, -1)
        actions = rollouts['actions'].view(batch_size_total, -1)
        old_log_probs = rollouts['log_probs'].view(batch_size_total)
        returns = rollouts['returns'].view(batch_size_total)
        advantages = rollouts['advantages'].view(batch_size_total)

        # NOTE: Do NOT normalize advantages here over the full buffer.
        # Normalization happens per mini-batch inside the loop below,
        # ensuring each mini-batch uses its own mean/std statistics.

        # Mini-batch training
        indices = np.arange(batch_size_total)
        losses = {'policy': [], 'value': [], 'entropy': [], 'total': []}

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size_total, self.batch_size):
                end = min(start + self.batch_size, batch_size_total)
                mb_idx = indices[start:end]

                # Get mini-batch
                mb_visual = visuals[mb_idx]
                mb_state = states[mb_idx]
                mb_action = actions[mb_idx]
                mb_old_log_prob = old_log_probs[mb_idx]
                mb_return = returns[mb_idx]
                mb_advantage = advantages[mb_idx]

                # Normalize advantages per mini-batch (correct: uses this batch's statistics)
                mb_advantage = (mb_advantage - mb_advantage.mean()) / (mb_advantage.std() + 1e-8)

                # Forward pass
                log_prob, value, entropy = self.policy.evaluate(
                    mb_visual, mb_state, mb_action
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_prob - mb_old_log_prob)
                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * mb_advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((value - mb_return) ** 2).mean()

                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses['policy'].append(policy_loss.item())
                losses['value'].append(value_loss.item())
                losses['entropy'].append(-entropy_loss.item())  # Log positive entropy
                losses['total'].append(loss.item())

        return {k: np.mean(v) for k, v in losses.items()}

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to tensorboard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def save_checkpoint(self, path: str, iteration: int, extra_info: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        if extra_info:
            checkpoint.update(extra_info)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('iteration', 0)
