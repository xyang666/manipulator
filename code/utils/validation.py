"""
validation.py
-------------
Validation utilities for evaluating trained agents on fixed test scenes.

Loads scenes from trajectories.json and evaluates agent performance.

Author: xie yang
Date:   2025-06

"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional


class ValidationSet:
    """
    Load and manage validation scenes from JSON file.
    """

    def __init__(self, json_path: str):
        """
        Parameters
        ----------
        json_path : path to trajectories JSON file
        """
        self.json_path = json_path
        self.scenes = self._load_scenes()
        print(f"[Validation] Loaded {len(self.scenes)} validation scenes from {json_path}")

    def _load_scenes(self) -> List[Dict]:
        """Load scenes from JSON file."""
        with open(self.json_path, 'r') as f:
            scenes = json.load(f)
        return scenes

    def get_scene(self, scene_id: int | str) -> Dict:
        """Get scene by ID (string) or index (int)."""
        for scene in self.scenes:
            if scene["scene_id"] == scene_id:
                return scene
        # Fallback: treat as numeric index
        try:
            idx = int(scene_id)
            if 0 <= idx < len(self.scenes):
                return self.scenes[idx]
        except (ValueError, TypeError):
            pass
        raise ValueError(f"Scene {scene_id} not found in {self.json_path}")

    def apply_scene_to_env(self, env, scene: Dict):
        """
        Apply a validation scene to the environment.

        Parameters
        ----------
        env   : ManipulatorEnv instance
        scene : scene dictionary with start, goal, obstacles
        """
        # Extract scene data
        env._current_scene_id = scene.get("scene_id", -1)
        env._current_scenario = (scene.get("scenario") or scene.get("kind") or
                                 str(env._current_scene_id).split("-validation-")[0])
        env.x_start = np.array(scene["start"])
        env.x_goal = np.array(scene["goal"])

        obstacles = scene["obstacles"]
        obstacle_centers = [np.array(obs[:3]) for obs in obstacles]
        obstacle_radii = [obs[3] for obs in obstacles]

        # Update SDF
        env.sdf.set_static_obstacles(obstacle_centers, obstacle_radii)

        # Dynamic obstacles: entries may carry optional velocity [x,y,z,r,vx,vy,vz]
        velocities = np.zeros((len(obstacles), 3), dtype=float)
        for i, obs in enumerate(obstacles):
            if len(obs) >= 7:
                velocities[i] = obs[4:7]
        bounds = None
        if "obstacle_bounds" in scene:
            bounds = np.array(scene["obstacle_bounds"], dtype=float)
        # Always reset motion state.  Otherwise a static scene evaluated after
        # a dynamic one inherits stale obstacle velocities from the prior
        # episode and silently becomes dynamic.
        env.set_dynamic_obstacles(velocities, bounds)

        trajectory = scene.get("trajectory", {"type": "linear"})
        if trajectory.get("type") == "figure_eight":
            center = np.asarray(trajectory["center"], dtype=float)
            y_amplitude = float(trajectory["y_amplitude"])
            z_amplitude = float(trajectory["z_amplitude"])
            period = float(trajectory["period"])
            omega = 2.0 * np.pi / period

            def position(t):
                phase = omega * t
                return center + np.array([
                    0.0, y_amplitude * np.sin(phase),
                    z_amplitude * np.sin(2.0 * phase),
                ])

            def velocity(t):
                phase = omega * t
                return np.array([
                    0.0, y_amplitude * omega * np.cos(phase),
                    2.0 * z_amplitude * omega * np.cos(2.0 * phase),
                ])

            env.set_parametric_trajectory(position, velocity)
            env.x_d = position(0.0)
            env.dx_d[:3] = velocity(0.0)
        else:
            env.use_parametric_traj = False
            env._parametric_pos_func = None
            env._parametric_vel_func = None
            env.x_d = env.x_start.copy()
            env.dx_d = np.zeros(3)

        # Set initial configuration: use start_q if available, otherwise IK
        if "start_q" in scene:
            env.q = np.array(scene["start_q"])
        else:
            q_init = env.kin.inverse_kinematics(
                np.concatenate([env.x_start, np.array([0, 0, 0, 1])])
            )
            if q_init is not None:
                env.q = q_init
            else:
                # Fallback to home pose
                env.q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])

        env.dq = np.zeros(env.n)

        # Sync to MuJoCo
        env._sync_obstacles_to_mujoco()
        if env.mj_data is not None:
            env.mj_data.qpos[:env.n] = env.q
            env.mj_data.qvel[:env.n] = env.dq
            env.mj_data.qpos[env.n:] = 0.0
            env.mj_data.qvel[env.n:] = 0.0
            import mujoco
            mujoco.mj_forward(env.mj_model, env.mj_data)

        # Reset episode state
        env._reset_episode_progress()
        env._integral_err = np.zeros(3)
        env._ever_collided = False
        env.reward_fn._prev_dist_to_goal = None
        env.ee_trajectory.clear()
        env._lag_lambda = 0.0
        if hasattr(env, "_gradient_prior_z"):
            env._gradient_prior_z.fill(0.0)


def evaluate_on_validation_set(agent, env, val_set: ValidationSet,
                               num_scenes: Optional[int] = None,
                               max_steps: int = 200) -> Dict:
    """
    Evaluate agent on validation scenes.

    Parameters
    ----------
    agent      : trained agent
    env        : environment instance
    val_set    : ValidationSet instance
    num_scenes : number of scenes to evaluate (None = all)
    max_steps  : maximum steps per episode

    Returns
    -------
    results : dict with validation metrics
    """
    if num_scenes is None:
        num_scenes = len(val_set.scenes)
    else:
        num_scenes = min(num_scenes, len(val_set.scenes))

    results = {
        "success_rate": 0.0,
        "avg_reward": 0.0,
        "avg_tracking_error": 0.0,
        "avg_min_distance": 0.0,
        "collision_rate": 0.0,
        "obstacle_collision_rate": 0.0,
        "self_collision_rate": 0.0,
        "scenario_metrics": {},
        "scene_results": []
    }

    successes = 0
    total_reward = 0.0
    total_tracking_error = 0.0
    total_min_distance = 0.0
    collisions = 0
    obstacle_collisions = 0
    self_collisions = 0
    scenario_totals = defaultdict(
        lambda: {"episodes": 0, "successes": 0, "collisions": 0,
                 "obstacle_collisions": 0, "self_collisions": 0}
    )

    for i in range(num_scenes):
        scene = val_set.scenes[i]

        # Apply scene to environment
        val_set.apply_scene_to_env(env, scene)
        obs = env._get_obs()

        # Run episode
        ep_reward = 0.0
        ep_tracking_errors = []
        ep_min_distances = []
        ep_ever_collided_mj = False  # MuJoCo collision flag
        ep_obstacle_collision = False
        ep_self_collision = False
        ep_success = False
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_reward += reward
            ep_tracking_errors.append(info.get("tracking_error", 0.0))
            ep_min_distances.append(info.get("d_obs", 0.0))
            ep_ever_collided_mj = ep_ever_collided_mj or info.get("collision", False)
            ep_obstacle_collision = (
                ep_obstacle_collision
                or info.get("obstacle_collision", False)
            )
            ep_self_collision = (
                ep_self_collision or info.get("self_collision", False)
            )
            ep_success = ep_success or info.get("success", False)

            steps += 1

        # Check success: reached goal with no MuJoCo collision
        x_ee, _ = env.kin.forward_kinematics(env.q)
        final_distance = np.linalg.norm(x_ee - env.x_goal)
        min_obs_dist = min(ep_min_distances) if ep_min_distances else 0.0
        success = ep_success and final_distance < env.success_tolerance \
            and not ep_ever_collided_mj

        if success:
            successes += 1
        if ep_ever_collided_mj:
            collisions += 1
        obstacle_collisions += int(ep_obstacle_collision)
        self_collisions += int(ep_self_collision)
        scene_id = str(scene["scene_id"])
        scenario = scene.get("scenario") or scene.get("kind")
        if not scenario:
            scenario = scene_id.split("-validation-")[0].split("-")[0]
        group = scenario_totals[str(scenario)]
        group["episodes"] += 1
        group["successes"] += int(success)
        group["collisions"] += int(ep_ever_collided_mj)
        group["obstacle_collisions"] += int(ep_obstacle_collision)
        group["self_collisions"] += int(ep_self_collision)

        total_reward += ep_reward
        total_tracking_error += np.mean(ep_tracking_errors) if ep_tracking_errors else 0.0
        total_min_distance += min_obs_dist

        # Store per-scene result
        results["scene_results"].append({
            "scene_id": scene["scene_id"],
            "success": success,
            "scenario": str(scenario),
            "collision": ep_ever_collided_mj,
            "obstacle_collision": ep_obstacle_collision,
            "self_collision": ep_self_collision,
            "reward": ep_reward,
            "tracking_error": np.mean(ep_tracking_errors) if ep_tracking_errors else 0.0,
            "min_distance": min_obs_dist,
            "steps": steps
        })

    # Aggregate metrics
    results["success_rate"] = successes / num_scenes
    results["avg_reward"] = total_reward / num_scenes
    results["avg_tracking_error"] = total_tracking_error / num_scenes
    results["avg_min_distance"] = total_min_distance / num_scenes
    results["collision_rate"] = collisions / num_scenes
    results["obstacle_collision_rate"] = obstacle_collisions / num_scenes
    results["self_collision_rate"] = self_collisions / num_scenes
    for scenario, counts in sorted(scenario_totals.items()):
        episodes = counts["episodes"]
        results["scenario_metrics"][scenario] = {
            "episodes": episodes,
            "success_rate": counts["successes"] / episodes,
            "collision_rate": counts["collisions"] / episodes,
            "obstacle_collision_rate": (
                counts["obstacle_collisions"] / episodes
            ),
            "self_collision_rate": counts["self_collisions"] / episodes,
        }

    return results
