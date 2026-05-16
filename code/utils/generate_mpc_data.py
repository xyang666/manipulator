"""
generate_mpc_data.py
--------------------
Run MPC on a scene and collect (obs, action_rl) pairs for BC pretraining.
Maps MPC's dq_cmd to the RL action format [delta_x_rl(3), z(4)].
"""
import numpy as np
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.manipulator_env import ManipulatorEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_json", type=str, default=None, required=True)
    p.add_argument("--scene_id", type=int, default=9)
    p.add_argument("--n_trajectories", type=int, default=10,
                   help="Number of MPC rollouts (with different start/goal if available)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--output", type=str, default="/root/manipulator/data/mpc_scene9.npz")
    p.add_argument("--obs_scene_embed", type=int, default=5)
    p.add_argument("--obs_waypoint_steps", type=str, default="10,20,50")
    return p.parse_args()


def run_mpc_on_scene(args):
    """Run MPC and collect RL-equivalent actions."""
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    urdf = os.path.join(_ROOT, "code/.venv/lib/python3.12/site-packages/cmeel.prefix"
                        "/share/example-robot-data/robots/panda_description/urdf/panda.urdf")
    xml = os.path.join(_ROOT, "models/panda_scene.xml")

    # Load scenes
    import json
    with open(args.scene_json) as f:
        scenes = json.load(f)

    obs_list = []
    action_list = []

    for traj_idx in range(args.n_trajectories):
        # Use specified scene (or cycle through available ones)
        sid = args.scene_id
        scene = scenes[sid % len(scenes)]

        waypoint_steps = [int(x) for x in args.obs_waypoint_steps.split(",")] if args.obs_waypoint_steps else []
        env = ManipulatorEnv(
            urdf_path=urdf, xml_path=xml,
            episode_len=args.steps, n_obstacles=len(scene["obstacles"]),
            obs_radius=0.03,
            controller="mpc", mpc_horizon=10,
            obs_scene_embed=args.obs_scene_embed,
            obs_waypoint_steps=waypoint_steps,
        )
        env.reset()

        # Apply scene
        from test import apply_scene
        apply_scene(env, scene)

        for step in range(args.steps):
            obs = env._get_obs()

            # Run MPC step
            _, _, done, info = env.step(np.zeros(env.act_dim))

            # Extract MPC's dq_cmd from the env's last step
            # MPC dq_cmd = env.dq (the applied joint velocity)
            dq_mpc = env.dq.copy()

            # Convert MPC dq_cmd to RL action [delta_x_rl(3), z(4)]
            J_pos = env.kin.jacobian_position(env.q)  # (3, 7)
            dx_ee_mpc = J_pos @ dq_mpc

            # dx_nom = Kp * (x_d - x_ee) + dx_d + Ki * integral
            # This matches env._compute_task_velocity()
            dx_nom = env._compute_task_velocity()

            # RL task relaxation component
            delta_x_rl = dx_ee_mpc - dx_nom

            # Null-space component: project dq_mpc onto null-space
            B = env.kin.null_space_basis_position(env.q)  # (7, 4)
            # Pseudo-inverse of B to get coefficients
            z = np.linalg.pinv(B) @ dq_mpc  # (4,)
            z = np.clip(z, -2.0, 2.0)  # clip for stability

            # RL action
            action_rl = np.concatenate([delta_x_rl, z])  # (7,)

            obs_list.append(obs.astype(np.float32))
            action_list.append(action_rl.astype(np.float32))

            if done:
                break

        print(f"[MPC data] Trajectory {traj_idx+1}/{args.n_trajectories}: "
              f"scene_id={sid}, steps={step+1}")

    obs_data = np.stack(obs_list, axis=0)
    act_data = np.stack(action_list, axis=0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, obs=obs_data, act=act_data)
    print(f"[MPC data] Saved {len(obs_data)} pairs to {args.output}")
    print(f"  obs shape: {obs_data.shape}, act shape: {act_data.shape}")


if __name__ == "__main__":
    args = parse_args()
    run_mpc_on_scene(args)