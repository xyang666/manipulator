"""
tune_baseline.py
----------------
Automatic tuning of baseline KP controller parameters.
Uses grid search to optimize tracking performance.

Usage:
    python tune_baseline.py [--steps 500] [--trials 20]

Author: xie yang
Date:   2025-06

"""

import numpy as np
import argparse
import sys
import os
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from env.manipulator_env import ManipulatorEnv
from robot_config import DEFAULT_URDF, DEFAULT_XML


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500, help="Steps per trial")
    p.add_argument("--trials", type=int, default=20, help="Number of tuning trials")
    p.add_argument("--urdf", type=str, default=DEFAULT_URDF)
    p.add_argument("--xml", type=str, default=DEFAULT_XML)
    return p.parse_args()


def evaluate_params(env, steps: int, kp_base: float, kp_joint: float, kd_joint: float) -> Dict:
    """Evaluate controller with given parameters."""
    # Patch the environment methods
    def patched_compute_task_velocity(self):
        x_ee, _ = self.kin.forward_kinematics(self.q)
        pos_err = self.x_d - x_ee
        err_norm = np.linalg.norm(pos_err)
        Kp = kp_base * (1.0 + 2.0 * np.tanh(err_norm / 0.1))
        dx_cmd = np.zeros(6)
        dx_cmd[:3] = self.dx_d[:3] + Kp * pos_err
        dx_cmd[:3] = np.clip(dx_cmd[:3], -0.5, 0.5)
        return dx_cmd

    def patched_mujoco_step(self, dq_cmd):
        q_desired = self.q + dq_cmd * self.dt
        self.mj_data.ctrl[:self.n] = dq_cmd
        self.mj_data.qpos[self.n:self.n + 2] = 0.0
        self.mj_data.qvel[self.n:self.n + 2] = 0.0
        import mujoco
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.q = self.mj_data.qpos[:self.n].copy()
        self.dq = self.mj_data.qvel[:self.n].copy()

    # Monkey patch
    import types
    env._compute_task_velocity = types.MethodType(patched_compute_task_velocity, env)
    env._mujoco_step = types.MethodType(patched_mujoco_step, env)

    # Run evaluation
    obs = env.reset()
    tracking_errors = []
    rewards = []

    for step in range(steps):
        action = np.zeros(env.act_dim)
        obs, reward, done, info = env.step(action)
        x_ee, _ = env.kin.forward_kinematics(env.q)
        track_err = np.linalg.norm(x_ee - env.x_d)
        tracking_errors.append(track_err)
        rewards.append(reward)
        if done:
            break

    return {
        'mean_error': np.mean(tracking_errors),
        'max_error': np.max(tracking_errors),
        'std_error': np.std(tracking_errors),
        'total_reward': np.sum(rewards),
        'mean_reward': np.mean(rewards)
    }


def grid_search(env, args) -> Tuple[Dict, Dict]:
    """Grid search over controller parameters."""
    # Parameter ranges - focus on task-space gain
    kp_base_range = [5, 10, 15, 20, 25, 30, 40, 50]
    kp_joint_range = [100]  # Fixed
    kd_joint_range = [20]   # Fixed

    best_params = None
    best_score = float('inf')
    best_results = None

    print("\n" + "="*80)
    print("Grid Search for Baseline Controller Tuning")
    print("="*80)
    print(f"{'Trial':>5} {'Kp_base':>10} {'Kp_joint':>10} {'Kd_joint':>10} {'Mean_err':>12} {'Max_err':>12} {'Score':>12}")
    print("-"*80)

    trial = 0
    for kp_base in kp_base_range:
        for kp_joint in kp_joint_range:
            for kd_joint in kd_joint_range:
                if trial >= args.trials:
                    break

                results = evaluate_params(env, args.steps, kp_base, kp_joint, kd_joint)

                # Score: weighted combination of mean and max error
                score = results['mean_error'] + 0.2 * results['max_error']

                print(f"{trial:>5d} {kp_base:>10.1f} {kp_joint:>10.1f} {kd_joint:>10.1f} "
                      f"{results['mean_error']:>12.4f} {results['max_error']:>12.4f} {score:>12.4f}")

                if score < best_score:
                    best_score = score
                    best_params = {'kp_base': kp_base, 'kp_joint': kp_joint, 'kd_joint': kd_joint}
                    best_results = results

                trial += 1

    return best_params, best_results


def main():
    args = parse_args()

    print("="*80)
    print("Automatic Baseline Controller Tuning")
    print("="*80)

    # Create environment
    env = ManipulatorEnv(
        urdf_path=args.urdf,
        xml_path=args.xml,
        n_obstacles=0,
        obs_radius=0.03,
        episode_len=args.steps,
        use_mpc=False
    )

    # Run grid search
    best_params, best_results = grid_search(env, args)

    # Print results
    print("\n" + "="*80)
    print("Tuning Results")
    print("="*80)
    print(f"Best parameters:")
    print(f"  Kp_base  = {best_params['kp_base']:.1f}")
    print(f"  Kp_joint = {best_params['kp_joint']:.1f}")
    print(f"  Kd_joint = {best_params['kd_joint']:.1f}")
    print(f"\nPerformance:")
    print(f"  Mean tracking error: {best_results['mean_error']:.4f} m")
    print(f"  Max tracking error:  {best_results['max_error']:.4f} m")
    print(f"  Std tracking error:  {best_results['std_error']:.4f} m")
    print(f"  Total reward:        {best_results['total_reward']:.1f}")

    print("\nTo apply these parameters, update manipulator_env.py:")
    print(f"  Line ~415: Kp_base = {best_params['kp_base']}")
    print(f"  Line ~432: Kp = {best_params['kp_joint']}")
    print(f"  Line ~433: Kd = {best_params['kd_joint']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
