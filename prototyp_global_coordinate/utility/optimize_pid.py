"""
PID parameter optimization using Optuna (Bayesian optimization).

Runs headless Genesis simulation episodes with candidate PID parameters
and minimizes a cost function measuring tracking performance.

Usage:
    python optimize_pid.py                    # run 100 trials
    python optimize_pid.py --n_trials 200     # run 200 trials
    python optimize_pid.py --show_best        # print best params found so far
"""

import argparse
import torch
import genesis as gs
import yaml
import optuna
from envs.coordinate_landing_env import CoordinateLandingEnv
from genesis.utils.geom import quat_to_xyz

# Penalty returned when a trial causes simulation instability (NaN forces)
NAN_PENALTY = 100.0


def build_env(config: dict, device: torch.device) -> CoordinateLandingEnv:
    """Build a headless environment (reused across trials)."""
    config["scene"]["show_viewer"] = False
    config["scene"]["visualize_target"] = False
    env = CoordinateLandingEnv(config, device=device)
    env.build()
    return env


def evaluate(env: CoordinateLandingEnv, pid_params: dict, n_steps: int = 500) -> float:
    """
    Run one episode with the given PID params and return a cost.

    The test scenario:
        - Drone spawns at (0, 0, 3), target at (5, 0, 0)
        - Action: constant forward velocity command toward target
        - Measure how well the controller tracks the command

    Cost function components (lower is better):
        - tracking_error: mean distance to target over the episode
        - crash_penalty:  large penalty if drone hits the ground
        - oscillation:    penalize rapid velocity changes (jitter)

    TODO: This is where YOUR design choices matter most.
          Adjust the weights and components to match what "good" means
          for your landing task.
    """
    # Inject PID params into the controller
    ctrl = env.controller
    ctrl._DronePIDController__pid_vel_x.kp = pid_params["pid_params_vel_x"][0]
    ctrl._DronePIDController__pid_vel_x.ki = pid_params["pid_params_vel_x"][1]
    ctrl._DronePIDController__pid_vel_x.kd = pid_params["pid_params_vel_x"][2]
    ctrl._DronePIDController__pid_vel_y.kp = pid_params["pid_params_vel_y"][0]
    ctrl._DronePIDController__pid_vel_y.ki = pid_params["pid_params_vel_y"][1]
    ctrl._DronePIDController__pid_vel_y.kd = pid_params["pid_params_vel_y"][2]
    ctrl._DronePIDController__pid_vel_z.kp = pid_params["pid_params_vel_z"][0]
    ctrl._DronePIDController__pid_vel_z.ki = pid_params["pid_params_vel_z"][1]
    ctrl._DronePIDController__pid_vel_z.kd = pid_params["pid_params_vel_z"][2]
    ctrl._DronePIDController__pid_att_roll.kp = pid_params["pid_params_roll"][0]
    ctrl._DronePIDController__pid_att_roll.ki = pid_params["pid_params_roll"][1]
    ctrl._DronePIDController__pid_att_roll.kd = pid_params["pid_params_roll"][2]
    ctrl._DronePIDController__pid_att_pitch.kp = pid_params["pid_params_pitch"][0]
    ctrl._DronePIDController__pid_att_pitch.ki = pid_params["pid_params_pitch"][1]
    ctrl._DronePIDController__pid_att_pitch.kd = pid_params["pid_params_pitch"][2]
    ctrl._DronePIDController__pid_att_yaw.kp = pid_params["pid_params_yaw"][0]
    ctrl._DronePIDController__pid_att_yaw.ki = pid_params["pid_params_yaw"][1]
    ctrl._DronePIDController__pid_att_yaw.kd = pid_params["pid_params_yaw"][2]

    env.reset()
    device = env.device

    total_dist = 0.0
    crashed = False
    prev_vel = None
    jitter_sum = 0.0
    vel_tracking_sum = 0.0
    att_tracking_sum = 0.0

    # Constant action: move toward target (+X direction)
    action = torch.zeros(env.n_envs, 6, device=device)
    action[:, 0] = 1.0  # forward velocity command

    # Commanded targets (action layout: vx, vy, vz, roll, pitch, yaw)
    cmd_vel = action[0, :3]   # (3,) commanded velocity
    cmd_att = action[0, 3:]   # (3,) commanded body rates (compared vs attitude in degrees)

    for step in range(n_steps):
        obs, rewards = env.step(action)

        dist = torch.norm(env.rel_pos[0]).item()
        total_dist += dist

        # Velocity tracking error: how far actual vel is from commanded
        actual_vel = env.drone.get_vel()[0]
        vel_tracking_sum += torch.norm(actual_vel - cmd_vel).item()

        # Attitude tracking error: how far actual attitude is from commanded
        actual_att = quat_to_xyz(env.drone.get_quat(), rpy=True, degrees=True)[0]
        att_tracking_sum += torch.norm(actual_att - cmd_att).item()

        # Track velocity jitter (oscillation penalty)
        if prev_vel is not None:
            jitter_sum += torch.norm(actual_vel - prev_vel).item()
        prev_vel = actual_vel.clone()

        # Early termination if crashed
        if env.pos[0, 2].item() < 0.2:
            crashed = True
            break

    # --- Cost function ---
    n = max(step + 1, 1)
    avg_dist = total_dist / n
    avg_jitter = jitter_sum / max(step, 1)
    avg_vel_tracking = vel_tracking_sum / n
    avg_att_tracking = att_tracking_sum / n
    crash_penalty = 10000.0 if crashed else 0.0

    cost = (avg_dist
            + avg_jitter
            + 2.0 * avg_vel_tracking
            + 1.0 * avg_att_tracking
            + crash_penalty)

    return cost


def objective(trial: optuna.Trial, env: CoordinateLandingEnv) -> float:
    """Optuna objective: sample PID params and evaluate."""

    # Velocity controllers — typically need higher Kp
    vel_x_kp = trial.suggest_float("vel_x_kp", 0.5, 20.0)
    vel_x_ki = trial.suggest_float("vel_x_ki", 0.0, 1.0)
    vel_x_kd = trial.suggest_float("vel_x_kd", 0.0, 5.0)

    vel_y_kp = trial.suggest_float("vel_y_kp", 0.5, 20.0)
    vel_y_ki = trial.suggest_float("vel_y_ki", 0.0, 1.0)
    vel_y_kd = trial.suggest_float("vel_y_kd", 0.0, 5.0)

    vel_z_kp = trial.suggest_float("vel_z_kp", 0.5, 30.0)
    vel_z_ki = trial.suggest_float("vel_z_ki", 0.0, 1.0)
    vel_z_kd = trial.suggest_float("vel_z_kd", 0.0, 5.0)

    # Attitude controllers — typically need lower Kp
    roll_kp = trial.suggest_float("roll_kp", 0.1, 10.0)
    roll_ki = trial.suggest_float("roll_ki", 0.0, 0.5)
    roll_kd = trial.suggest_float("roll_kd", 0.0, 3.0)

    pitch_kp = trial.suggest_float("pitch_kp", 0.1, 10.0)
    pitch_ki = trial.suggest_float("pitch_ki", 0.0, 0.5)
    pitch_kd = trial.suggest_float("pitch_kd", 0.0, 3.0)

    yaw_kp = trial.suggest_float("yaw_kp", 0.1, 5.0)
    yaw_ki = trial.suggest_float("yaw_ki", 0.0, 0.5)
    yaw_kd = trial.suggest_float("yaw_kd", 0.0, 2.0)

    pid_params = {
        "pid_params_vel_x": [vel_x_kp, vel_x_ki, vel_x_kd],
        "pid_params_vel_y": [vel_y_kp, vel_y_ki, vel_y_kd],
        "pid_params_vel_z": [vel_z_kp, vel_z_ki, vel_z_kd],
        "pid_params_roll":  [roll_kp, roll_ki, roll_kd],
        "pid_params_pitch": [pitch_kp, pitch_ki, pitch_kd],
        "pid_params_yaw":   [yaw_kp, yaw_ki, yaw_kd],
    }

    try:
        return evaluate(env, pid_params)
    except gs.GenesisException:
        # Aggressive PID gains can blow up the sim — penalize and move on
        env.reset()
        return NAN_PENALTY


def print_best(study: optuna.Study):
    """Print best parameters in YAML-friendly format."""
    best = study.best_trial
    print(f"\nBest cost: {best.value:.4f}")
    print(f"Best parameters (copy to training_config.yaml):\n")
    print(f"  pid_params_vel_x: [{best.params['vel_x_kp']:.4f}, {best.params['vel_x_ki']:.4f}, {best.params['vel_x_kd']:.4f}]")
    print(f"  pid_params_vel_y: [{best.params['vel_y_kp']:.4f}, {best.params['vel_y_ki']:.4f}, {best.params['vel_y_kd']:.4f}]")
    print(f"  pid_params_vel_z: [{best.params['vel_z_kp']:.4f}, {best.params['vel_z_ki']:.4f}, {best.params['vel_z_kd']:.4f}]")
    print(f"  pid_params_roll:  [{best.params['roll_kp']:.4f}, {best.params['roll_ki']:.4f}, {best.params['roll_kd']:.4f}]")
    print(f"  pid_params_pitch: [{best.params['pitch_kp']:.4f}, {best.params['pitch_ki']:.4f}, {best.params['pitch_kd']:.4f}]")
    print(f"  pid_params_yaw:   [{best.params['yaw_kp']:.4f}, {best.params['yaw_ki']:.4f}, {best.params['yaw_kd']:.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=500, help="Sim steps per evaluation")
    parser.add_argument("--show_best", action="store_true", help="Show best params from existing study")
    parser.add_argument("--db", type=str, default="sqlite:///optuna_pid.db", help="Optuna storage for resumable studies")
    args = parser.parse_args()

    study_name = "pid_optimization"

    if args.show_best:
        study = optuna.load_study(study_name=study_name, storage=args.db)
        print_best(study)
        return

    # Build environment once — reused for all trials
    with open("config/training_config.yaml") as f:
        config = yaml.safe_load(f)
    config["env"]["n_envs"] = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")

    env = build_env(config, device)

    # Create or resume study (persisted to SQLite)
    study = optuna.create_study(
        study_name=study_name,
        storage=args.db,
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, env),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print_best(study)


if __name__ == "__main__":
    main()
