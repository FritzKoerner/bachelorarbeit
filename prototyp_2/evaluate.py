"""
Evaluation and visualization script for trained drone landing policy.
"""

import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import genesis as gs

from envs.vineyard_landing_env import VineyardLandingEnv
from networks.actor_critic import ActorCritic


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str, policy: ActorCritic, device: torch.device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    iteration = checkpoint.get('iteration', 0)
    best_reward = checkpoint.get('best_reward', None)
    print(f"Loaded checkpoint from iteration {iteration}")
    if best_reward is not None:
        print(f"  Best reward: {best_reward:.2f}")
    return iteration


def evaluate(
    env: VineyardLandingEnv,
    policy: ActorCritic,
    n_episodes: int,
    device: torch.device,
    render: bool = False,
    save_dir: str = None,
):
    """
    Evaluate policy on environment.

    Args:
        env: Environment
        policy: Trained policy
        n_episodes: Number of episodes to evaluate
        device: Torch device
        render: Whether to render and save visualizations
        save_dir: Directory to save visualizations

    Returns:
        stats: Evaluation statistics
    """
    policy.eval()

    episode_rewards = []
    episode_lengths = []
    landings = 0
    crashes = 0
    timeouts = 0

    trajectories = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = torch.zeros(env.n_envs, dtype=torch.bool, device=device)
        episode_reward = 0
        episode_length = 0

        trajectory = {
            'positions': [],
            'velocities': [],
            'actions': [],
        }

        while not done.all():
            with torch.no_grad():
                action, _ = policy.act(obs['visual'], obs['state'], deterministic=True)

            obs, reward, done, info = env.step(action)

            episode_reward += reward[0].item()
            episode_length += 1

            # Record trajectory
            pos = obs['state'][0, :3].cpu().numpy()
            vel = obs['state'][0, 3:6].cpu().numpy()
            trajectory['positions'].append(pos.copy())
            trajectory['velocities'].append(vel.copy())
            trajectory['actions'].append(action[0].cpu().numpy().copy())

            if done[0]:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if info['landed'][0]:
            landings += 1
            outcome = "LANDED"
        elif info['crashed'][0]:
            crashes += 1
            outcome = "CRASHED"
        elif info['timeout'][0]:
            timeouts += 1
            outcome = "TIMEOUT"
        else:
            outcome = "OOB"

        final_pos = trajectory['positions'][-1]
        print(f"Episode {ep + 1:3d}: {outcome:8s} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Length: {episode_length:4d} | "
              f"Final pos: ({final_pos[0]:5.1f}, {final_pos[1]:5.1f}, {final_pos[2]:5.2f})")

        trajectories.append(trajectory)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f}")
    print(f"Landings: {landings} ({100 * landings / n_episodes:.1f}%)")
    print(f"Crashes: {crashes} ({100 * crashes / n_episodes:.1f}%)")
    print(f"Timeouts: {timeouts} ({100 * timeouts / n_episodes:.1f}%)")

    # Visualize if requested
    if render and save_dir:
        visualize_trajectories(trajectories, save_dir)

    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'landings': landings,
        'crashes': crashes,
        'timeouts': timeouts,
        'success_rate': landings / n_episodes,
    }

    return stats


def visualize_trajectories(trajectories: list, save_dir: str):
    """Create visualization plots."""
    os.makedirs(save_dir, exist_ok=True)

    # 3D trajectory plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, traj in enumerate(trajectories):
        positions = np.array(traj['positions'])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                alpha=0.7, linewidth=1, label=f'Ep {i + 1}' if i < 5 else None)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                   marker='x', s=50)

    # Draw landing strips
    strip_y = [-4, 0, 4]
    for y in strip_y:
        xs = np.array([-25, 25])
        ys = np.array([y - 1, y - 1, y + 1, y + 1, y - 1])
        zs = np.zeros(5)
        ax.plot([-25, 25, 25, -25, -25], [y - 1, y - 1, y + 1, y + 1, y - 1],
                [0, 0, 0, 0, 0], 'g--', alpha=0.3)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Drone Landing Trajectories')
    if len(trajectories) <= 5:
        ax.legend()

    plt.savefig(os.path.join(save_dir, 'trajectories_3d.png'), dpi=150)
    plt.close()

    # Height over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot height
    ax = axes[0, 0]
    for traj in trajectories[:10]:
        positions = np.array(traj['positions'])
        ax.plot(positions[:, 2], alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Height (m)')
    ax.set_title('Height vs Time')
    ax.axhline(y=0.3, color='r', linestyle='--', label='Landing threshold')
    ax.legend()

    # Plot vertical velocity
    ax = axes[0, 1]
    for traj in trajectories[:10]:
        velocities = np.array(traj['velocities'])
        ax.plot(velocities[:, 2], alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Vertical velocity (m/s)')
    ax.set_title('Vertical Velocity vs Time')
    ax.axhline(y=-0.5, color='g', linestyle='--', label='Target descent')
    ax.legend()

    # Plot XY position
    ax = axes[1, 0]
    for traj in trajectories[:10]:
        positions = np.array(traj['positions'])
        ax.plot(positions[:, 0], positions[:, 1], alpha=0.7)
        ax.scatter(positions[-1, 0], positions[-1, 1], marker='x', s=50)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Trajectory (top view)')
    ax.axhspan(-3, -1, alpha=0.2, color='green', label='Soil strips')
    ax.axhspan(1, 3, alpha=0.2, color='green')
    ax.axhspan(-5, -3, alpha=0.2, color='red', label='Vineyard strips')
    ax.axhspan(3, 5, alpha=0.2, color='red')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-6, 6)
    ax.legend()

    # Plot actions
    ax = axes[1, 1]
    if trajectories:
        actions = np.array(trajectories[0]['actions'])
        ax.plot(actions[:, 0], label='vx')
        ax.plot(actions[:, 1], label='vy')
        ax.plot(actions[:, 2], label='vz')
        ax.plot(actions[:, 3], label='yaw_rate')
    ax.set_xlabel('Step')
    ax.set_ylabel('Action')
    ax.set_title('Actions (Episode 1)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'analysis.png'), dpi=150)
    plt.close()

    print(f"\nVisualizations saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained drone landing policy')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Path to config file')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Save trajectory visualizations')
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                        help='Directory to save results')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Genesis
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)

    # Load config
    config = load_config(args.config)
    config['n_envs'] = 1  # Single environment for evaluation

    # Create environment
    print("Creating environment...")
    env = VineyardLandingEnv(config, device=device)

    # Create and load policy
    print("Loading policy...")
    policy = ActorCritic(
        state_dim=env.obs_state_dim,
        action_dim=env.action_dim,
        hidden_dim=config.get('hidden_dim', 256),
        visual_feature_dim=config.get('visual_feature_dim', 256),
    ).to(device)

    load_checkpoint(args.checkpoint, policy, device)

    # Add timestamp to save dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)

    # Evaluate
    print(f"\nRunning evaluation ({args.n_episodes} episodes)...")
    stats = evaluate(
        env=env,
        policy=policy,
        n_episodes=args.n_episodes,
        device=device,
        render=args.render,
        save_dir=save_dir if args.render else None,
    )

    env.close()


if __name__ == '__main__':
    main()
