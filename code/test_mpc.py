"""
test_mpc.py
-----------
Test script for MPC controller with MuJoCo visualization.
Compares MPC tracking performance against baseline KP controller.

Usage:
    python test_mpc.py [--render] [--steps 500]
"""

import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from env.manipulator_env import ManipulatorEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--render", action="store_true", help="Enable MuJoCo viewer")
    p.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
    p.add_argument("--horizon", type=int, default=10, help="MPC prediction horizon")
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    _venv_data = os.path.join(_here, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                              "/share/example-robot-data/robots/panda_description")
    _default_urdf = os.path.join(_venv_data, "urdf/panda.urdf")
    _default_xml = os.path.join(_root, "models/panda_scene.xml")

    p.add_argument("--urdf", type=str, default=_default_urdf)
    p.add_argument("--xml", type=str, default=_default_xml)
    return p.parse_args()


def test_controller(env, args, controller_name):
    """Run test for a single controller."""
    print(f"\n{'='*60}")
    print(f"Testing {controller_name}")
    print(f"{'='*60}")

    obs = env.reset()

    tracking_errors = []
    obstacle_distances = []
    rewards = []

    print(f"\n{'Step':>6} {'Reward':>10} {'Track_err':>10} {'d_obs':>8}")
    print("-" * 50)

    for step in range(args.steps):
        # Zero action (pure tracking for MPC, null-space for RL)
        action = np.zeros(env.act_dim)

        # Step environment
        obs, reward, done, info = env.step(action)

        # Compute tracking error
        x_ee, _ = env.kin.forward_kinematics(env.q)
        track_err = np.linalg.norm(x_ee - env.x_d)

        # Record statistics
        tracking_errors.append(track_err)
        obstacle_distances.append(info["d_obs"])
        rewards.append(reward)

        # Render
        # env.render()
        # time.sleep(0.01)
        if args.render:
            env.render()
            time.sleep(0.01)

        # Print progress
        if step % 50 == 0:
            print(f"{step:>6d} {reward:>10.3f} {track_err:>10.4f} {info['d_obs']:>8.3f}")

        if done:
            print(f"\nEpisode terminated at step {step}")
            break

    # Summary
    print(f"\n{'-'*60}")
    print(f"{controller_name} Summary:")
    print(f"{'-'*60}")
    print(f"Mean tracking error:   {np.mean(tracking_errors):.4f} m")
    print(f"Max tracking error:    {np.max(tracking_errors):.4f} m")
    print(f"Min tracking error:    {np.min(tracking_errors):.4f} m")
    print(f"Std tracking error:    {np.std(tracking_errors):.4f} m")
    print(f"Mean reward:           {np.mean(rewards):.3f}")
    print(f"Total reward:          {np.sum(rewards):.3f}")

    return {
        'tracking_errors': tracking_errors,
        'mean_error': np.mean(tracking_errors),
        'max_error': np.max(tracking_errors),
        'rewards': rewards,
        'total_reward': np.sum(rewards)
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("MPC Controller Test")
    print("=" * 60)

    # Test 1: Baseline KP controller
    print("\n[1/2] Testing baseline KP controller...")
    env_kp = ManipulatorEnv(
        urdf_path=args.urdf,
        xml_path=args.xml,
        n_obstacles=0,
        obs_radius=0.03,
        episode_len=args.steps,
        use_mpc=False
    )
    results_kp = test_controller(env_kp, args, "KP Controller")

    # Test 2: MPC controller
    print("\n[2/2] Testing MPC controller...")
    env_mpc = ManipulatorEnv(
        urdf_path=args.urdf,
        xml_path=args.xml,
        n_obstacles=0,
        obs_radius=0.03,
        episode_len=args.steps,
        use_mpc=True,
        mpc_horizon=args.horizon
    )
    results_mpc = test_controller(env_mpc, args, "MPC Controller")

    # Comparison
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"{'Metric':<30} {'KP':>12} {'MPC':>12} {'Improvement':>12}")
    print("-" * 60)

    mean_err_kp = results_kp['mean_error']
    mean_err_mpc = results_mpc['mean_error']
    improvement = (mean_err_kp - mean_err_mpc) / mean_err_kp * 100
    print(f"{'Mean tracking error (m)':<30} {mean_err_kp:>12.4f} {mean_err_mpc:>12.4f} {improvement:>11.1f}%")

    max_err_kp = results_kp['max_error']
    max_err_mpc = results_mpc['max_error']
    improvement = (max_err_kp - max_err_mpc) / max_err_kp * 100
    print(f"{'Max tracking error (m)':<30} {max_err_kp:>12.4f} {max_err_mpc:>12.4f} {improvement:>11.1f}%")

    reward_kp = results_kp['total_reward']
    reward_mpc = results_mpc['total_reward']
    improvement = (reward_mpc - reward_kp) / abs(reward_kp) * 100
    print(f"{'Total reward':<30} {reward_kp:>12.1f} {reward_mpc:>12.1f} {improvement:>11.1f}%")

    print("\nTest completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
