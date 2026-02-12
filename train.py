"""
PPO training loop for vineyard landing drone.
Loads all config from config/training_config.yaml.
"""

import os
import yaml
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import genesis as gs

from envs.vineyard_landing_env import VineyardLandingEnv
from networks.actor_critic import ActorCritic
from algorithms.ppo import PPOTrainer


def load_config(path='config/training_config.yaml'):
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_run_dir(config):
    """Create timestamped run directory, save full config snapshot."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config['logging']['log_dir'], timestamp)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save full resolved config (self-contained snapshot per user decision)
    with open(os.path.join(run_dir, 'config_snapshot.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return run_dir, ckpt_dir


def collect_rollouts(env, policy, obs, n_steps, device):
    """
    Collect (n_steps, n_envs, ...) rollout tensors.

    Args:
        env: VineyardLandingEnv (already built)
        policy: ActorCritic network
        obs: dict with 'visual' and 'state' tensors from previous rollout or reset()
        n_steps: number of steps to collect per environment
        device: torch device

    Returns:
        rollouts: dict with tensors of shape (n_steps, n_envs, ...)
        stats: episode statistics dict
        obs: updated obs for next collect_rollouts call
    """
    n_envs = env.n_envs

    # Pre-allocate storage on device
    visuals = torch.zeros(n_steps, n_envs, *env.obs_visual_shape, device=device)
    states = torch.zeros(n_steps, n_envs, env.obs_state_dim, device=device)
    actions = torch.zeros(n_steps, n_envs, env.action_dim, device=device)
    log_probs = torch.zeros(n_steps, n_envs, device=device)
    rewards = torch.zeros(n_steps, n_envs, device=device)
    dones = torch.zeros(n_steps, n_envs, device=device)
    values = torch.zeros(n_steps, n_envs, device=device)

    # Episode tracking per environment
    current_ep_reward = torch.zeros(n_envs, device=device)
    current_ep_length = torch.zeros(n_envs, dtype=torch.int32, device=device)
    episode_rewards = []
    episode_lengths = []
    episode_landings = 0
    episode_crashes = 0

    # Rollout loop — obs stored BEFORE step (critical for correct alignment)
    for step in range(n_steps):
        with torch.no_grad():
            action, log_prob = policy.act(obs['visual'], obs['state'])
            value = policy.get_value(obs['visual'], obs['state'])

        visuals[step] = obs['visual']
        states[step] = obs['state']
        actions[step] = action
        log_probs[step] = log_prob
        values[step] = value

        obs, reward, done, info = env.step(action)

        rewards[step] = reward
        dones[step] = done.float()

        # Track episodes
        current_ep_reward += reward
        current_ep_length += 1

        done_indices = torch.where(done)[0]
        for idx in done_indices:
            episode_rewards.append(current_ep_reward[idx].item())
            episode_lengths.append(current_ep_length[idx].item())
            current_ep_reward[idx] = 0
            current_ep_length[idx] = 0

        # Count landing and crash events (info contains boolean tensors)
        if 'landed' in info:
            episode_landings += int(info['landed'].sum().item())
        if 'crashed' in info:
            episode_crashes += int(info['crashed'].sum().item())

    # Bootstrap last value for GAE
    with torch.no_grad():
        last_value = policy.get_value(obs['visual'], obs['state'])

    rollouts = {
        'visuals': visuals,
        'states': states,
        'actions': actions,
        'log_probs': log_probs,
        'rewards': rewards,
        'dones': dones,
        'values': values,
        'last_value': last_value,
    }

    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'landings': episode_landings,
        'crashes': episode_crashes,
    }

    return rollouts, stats, obs


def train(config):
    """Main training loop."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Seed for reproducibility (from config, not hardcoded)
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize Genesis ONCE — never called inside env, builder, or network classes
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)

    # Create and build environment (build() deferred from __init__ per Plan 02)
    print("Creating and building environment...")
    env = VineyardLandingEnv(config, device=device)
    env.build()

    # Create policy network
    print("Creating policy network...")
    net_cfg = config['network']
    policy = ActorCritic(
        state_dim=env.obs_state_dim,
        action_dim=env.action_dim,
        hidden_dim=net_cfg['hidden_dim'],
        visual_feature_dim=net_cfg['visual_feature_dim'],
    ).to(device)

    # Create PPO trainer
    print("Creating PPO trainer...")
    trainer = PPOTrainer(policy, config, device=device)

    # Setup run directory and TensorBoard writer
    run_dir, ckpt_dir = setup_run_dir(config)
    writer = SummaryWriter(run_dir)
    print(f"Run directory: {run_dir}")

    # Training hyperparameters (all from YAML, no hardcoded constants)
    ppo_cfg = config['ppo']
    rollout_steps = ppo_cfg['rollout_steps']
    total_timesteps = ppo_cfg.get('total_timesteps', 1_000_000)
    log_cfg = config['logging']
    log_interval = log_cfg['log_interval']
    save_interval = log_cfg['save_interval']

    print(f"\nStarting training:")
    print(f"  Environments: {env.n_envs}")
    print(f"  Rollout steps: {rollout_steps}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Batch size: {ppo_cfg['batch_size']}")
    print()

    total_steps = 0
    update_step = 0

    # Initial reset to get first observation
    obs = env.reset()

    while total_steps < total_timesteps:
        # Collect rollouts — pass obs in, get updated obs back
        rollouts, stats, obs = collect_rollouts(env, policy, obs, rollout_steps, device)

        # Compute GAE returns and advantages
        returns, advantages = trainer.compute_gae(
            rollouts['rewards'],
            rollouts['values'],
            rollouts['dones'],
            rollouts['last_value'],
        )
        rollouts['returns'] = returns
        rollouts['advantages'] = advantages

        # PPO update
        losses = trainer.update(rollouts)

        total_steps += rollout_steps * env.n_envs
        update_step += 1

        # Logging
        if update_step % log_interval == 0:
            mean_reward = np.mean(stats['episode_rewards']) if stats['episode_rewards'] else 0.0
            mean_length = np.mean(stats['episode_lengths']) if stats['episode_lengths'] else 0.0

            print(
                f"Update {update_step:4d} | "
                f"Steps: {total_steps:8d} | "
                f"Reward: {mean_reward:7.2f} | "
                f"Length: {mean_length:6.1f} | "
                f"Land: {stats['landings']:3d} | "
                f"Crash: {stats['crashes']:3d} | "
                f"Loss: {losses['total']:.4f}"
            )

            writer.add_scalar('train/mean_reward', mean_reward, total_steps)
            writer.add_scalar('train/mean_episode_length', mean_length, total_steps)
            writer.add_scalar('train/landings', stats['landings'], total_steps)
            writer.add_scalar('train/crashes', stats['crashes'], total_steps)
            writer.add_scalar('loss/policy', losses['policy'], total_steps)
            writer.add_scalar('loss/value', losses['value'], total_steps)
            writer.add_scalar('loss/entropy', losses['entropy'], total_steps)
            writer.add_scalar('loss/total', losses['total'], total_steps)

        # Periodic checkpoint
        if update_step % save_interval == 0:
            trainer.save_checkpoint(
                os.path.join(ckpt_dir, f'checkpoint_{update_step}.pt'),
                update_step,
            )

    # Final checkpoint and cleanup
    trainer.save_checkpoint(os.path.join(ckpt_dir, 'final_model.pt'), update_step)
    writer.close()
    env.close()

    print(f"\nTraining complete! Total steps: {total_steps}")
    print(f"Run directory: {run_dir}")


def main():
    config = load_config()  # NO argparse, NO CLI overrides per user decision
    train(config)


if __name__ == '__main__':
    main()
