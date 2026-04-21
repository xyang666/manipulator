"""
test.py
-------
Test script for KP controller with MuJoCo visualization.
Validates the fixed MuJoCo configuration and demonstrates end-effector tracking.

Usage:
    python test.py [--render]
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
    p.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    _venv_data = os.path.join(_here, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                              "/share/example-robot-data/robots/panda_description")
    _default_urdf = os.path.join(_venv_data, "urdf/panda.urdf")
    _default_xml = os.path.join(_root, "models/panda_scene.xml")

    p.add_argument("--urdf", type=str, default=_default_urdf)
    p.add_argument("--xml", type=str, default=_default_xml)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("KP Controller Test with MuJoCo Visualization")
    print("=" * 60)

    # Create environment (no obstacles for pure tracking test)
    env = ManipulatorEnv(
        urdf_path=args.urdf,
        xml_path=args.xml,
        n_obstacles=0,  # Disable obstacles
        obs_radius=0.1,
        episode_len=args.steps
    )

    print(f"\nEnvironment initialized:")
    print(f"  State dim: {env.obs_dim}")
    print(f"  Action dim: {env.act_dim}")
    print(f"  MuJoCo model: {'Loaded' if env.mj_model else 'Not available'}")

    # Reset environment
    obs = env.reset()
    print(f"\nInitial state:")
    print(f"  q:   {env.q}")
    print(f"  x_d: {env.x_d}")

    # Statistics
    tracking_errors = []
    obstacle_distances = []
    rewards = []
    commanded_vels = []
    actual_vels = []

    print(f"\n{'Step':>6} {'Reward':>10} {'Track_err':>10} {'d_obs':>8} {'cmd_vel':>10} {'act_vel':>10}")
    print("-" * 70)

    for step in range(args.steps):
        # Pure null-space action (zero self-motion)
        # This tests if task-space tracking works without RL interference
        action = np.zeros(env.act_dim)
        # Get commanded velocity before step
        dx_desired = env._compute_task_velocity()
        dq_cmd = env.kin.combine_velocities(env.q, dx_desired, action)

        # Step environment
        obs, reward, done, info = env.step(action)

        # Compute tracking error
        x_ee, _ = env.kin.forward_kinematics(env.q)
        track_err = np.linalg.norm(x_ee - env.x_d)

        # Record statistics
        tracking_errors.append(track_err)
        obstacle_distances.append(info["d_obs"])
        rewards.append(reward)
        commanded_vels.append(np.linalg.norm(dq_cmd))
        actual_vels.append(np.linalg.norm(env.dq))

        # Render with target trajectory visualization
        if args.render:
            env.render()
            time.sleep(0.2)
        # env.render()

        # Print progress
        if step % 10 == 0:
            cmd_vel = np.linalg.norm(dq_cmd)
            act_vel = np.linalg.norm(env.dq)
            print(f"{step:>6d} {reward:>10.3f} {track_err:>10.4f} "
                  f"{info['d_obs']:>8.3f} {cmd_vel:>10.3f} {act_vel:>10.3f}")

        if done:
            print(f"\nEpisode terminated at step {step}")
            break

    # Summary statistics
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total steps:           {len(tracking_errors)}")
    print(f"Mean tracking error:   {np.mean(tracking_errors):.4f} m")
    print(f"Max tracking error:    {np.max(tracking_errors):.4f} m")
    print(f"Min obstacle distance: {np.min(obstacle_distances):.4f} m")
    print(f"Mean reward:           {np.mean(rewards):.3f}")
    print(f"Total reward:          {np.sum(rewards):.3f}")
    print(f"Mean cmd velocity:     {np.mean(commanded_vels):.3f} rad/s")
    print(f"Mean actual velocity:  {np.mean(actual_vels):.3f} rad/s")
    print(f"Velocity tracking:     {np.mean(actual_vels)/np.mean(commanded_vels)*100:.1f}%")

    # Check stability
    if np.any(np.isnan(env.q)) or np.any(np.isinf(env.q)):
        print("\n❌ FAILED: NaN/Inf detected in joint positions")
        return 1

    if np.max(np.abs(env.dq)) > 10.0:
        print(f"\n⚠️  WARNING: High joint velocity detected: {np.max(np.abs(env.dq)):.2f} rad/s")

    if np.mean(tracking_errors) < 0.1:
        print("\n✅ PASSED: Tracking error within acceptable range")
    else:
        print(f"\n⚠️  WARNING: High tracking error: {np.mean(tracking_errors):.4f} m")

    print("\nTest completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
