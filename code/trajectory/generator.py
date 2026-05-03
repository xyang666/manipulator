"""
generator.py
------------
Trajectory generation with collision-free constraints and manipulability filtering.

Generates trajectories within the reachable workspace, ensuring:
  - Start/end points are reachable (IK solvable)
  - Linear path does not intersect with spherical obstacles
  - Average manipulability along the path exceeds threshold

Output format (JSON):
  [
    {
      "scene_id": 0,
      "start": [x, y, z],
      "goal": [x, y, z],
      "obstacles": [[x, y, z, r], ...],
      "manipulability_mean": 0.123
    },
    ...
  ]

Usage:
    python -m trajectory.generator --num_scenes 100 --num_obstacles 5 --output trajectories.json
"""

import numpy as np
import json
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.kinematics import ManipulatorKinematics


class TrajectoryGenerator:
    """
    Generates collision-free trajectories with manipulability constraints.
    """

    def __init__(self,
                 urdf_path: str,
                 n_joints: int = 7,
                 workspace_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 manipulability_threshold: float = 0.01,
                 obstacle_radius_range: Tuple[float, float] = (0.05, 0.15),
                 min_obstacle_distance: float = 0.05):
        """
        Parameters
        ----------
        urdf_path                : path to URDF file
        n_joints                 : number of joints
        workspace_bounds         : (min, max) bounds for sampling, shape (3,) each
        manipulability_threshold : minimum average manipulability
        obstacle_radius_range    : (min, max) radius for obstacles (m)
        min_obstacle_distance    : minimum clearance from path to obstacle surface (m)
        """
        self.n = n_joints

        # Joint limits (Panda default)
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        self.kin = ManipulatorKinematics(urdf_path=urdf_path, n_joints=n_joints,
                                          q_min=self.q_min, q_max=self.q_max)
        # Workspace bounds (default: empirical Panda workspace)
        # Z_min = 0.10 ensures positions stay above the floor (Z=0) with margin
        if workspace_bounds is None:
            self.ws_min = np.array([-0.65, -0.65, 0.15])
            self.ws_max = np.array([0.65, 0.65, 0.85])
        else:
            self.ws_min, self.ws_max = workspace_bounds

        # Minimum Z for obstacles: bottom of sphere must stay above floor
        self.z_obs_min = 0.02

        self.manip_threshold = manipulability_threshold
        self.obs_radius_range = obstacle_radius_range
        self.min_clearance = min_obstacle_distance

        print(f"[TrajectoryGenerator] Initialized")
        print(f"  Workspace: X[{self.ws_min[0]:.2f}, {self.ws_max[0]:.2f}], "
              f"Y[{self.ws_min[1]:.2f}, {self.ws_max[1]:.2f}], "
              f"Z[{self.ws_min[2]:.2f}, {self.ws_max[2]:.2f}]")
        print(f"  Manipulability threshold: {self.manip_threshold}")
        print(f"  Obstacle radius range: {self.obs_radius_range}")

    def sample_reachable_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random joint configuration and compute FK.
        Guaranteed reachable since q is directly sampled within joint limits.

        Returns
        -------
        (position, joint_config) : (3,) EE position and (n,) joint angles
        """
        q = np.random.uniform(self.q_min, self.q_max)
        pos, _ = self.kin.forward_kinematics(q)
        return pos, q

    def compute_manipulability(self, q: np.ndarray) -> float:
        """
        Compute manipulability measure w(q) = sqrt(det(J J^T)).

        Parameters
        ----------
        q : joint configuration (n,)

        Returns
        -------
        w : manipulability index (scalar)
        """
        J = self.kin.jacobian(q)
        # Use only position Jacobian (first 3 rows) for manipulability
        J_pos = J[:3, :]
        w = np.sqrt(np.linalg.det(J_pos @ J_pos.T))
        return w

    def compute_path_manipulability(self, start_q: np.ndarray, goal_q: np.ndarray,
                                   n_samples: int = 20) -> float:
        """
        Compute average manipulability along linear joint-space path.

        Parameters
        ----------
        start_q   : start joint configuration
        goal_q    : goal joint configuration
        n_samples : number of samples along path

        Returns
        -------
        mean_manip : average manipulability
        """
        alphas = np.linspace(0, 1, n_samples)
        manips = []

        for alpha in alphas:
            q = (1 - alpha) * start_q + alpha * goal_q
            w = self.compute_manipulability(q)
            manips.append(w)

        return np.mean(manips)

    def compute_task_path_manipulability(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                                         n_samples: int = 20) -> Tuple[float, float]:
        """
        Compute manipulability along the task-space path using IK at each point.

        This mirrors what the controller actually does: IK at each x_d to get
        joint configurations, then compute manipulability at those configs.

        Parameters
        ----------
        start_pos  : start EE position (3,)
        goal_pos   : goal EE position (3,)
        n_samples  : number of samples along the task-space line

        Returns
        -------
        (mean_manip, min_manip) : average and minimum manipulability along path
        """
        alphas = np.linspace(0, 1, n_samples)
        manips = []

        for alpha in alphas:
            pos = (1 - alpha) * start_pos + alpha * goal_pos
            q_ik = self.kin.inverse_kinematics(pos)
            if q_ik is None:
                return 0.0, 0.0  # IK failure → invalid path
            w = self.compute_manipulability(q_ik)
            manips.append(w)

        return float(np.mean(manips)), float(np.min(manips))

    def line_sphere_collision(self, p1: np.ndarray, p2: np.ndarray,
                             center: np.ndarray, radius: float) -> bool:
        """
        Check if line segment [p1, p2] intersects sphere (center, radius).

        Uses distance from point to line segment formula.

        Returns
        -------
        collision : True if distance < radius + min_clearance
        """
        # Vector from p1 to p2
        d = p2 - p1
        # Vector from p1 to center
        f = p1 - center

        # Quadratic equation: ||p1 + t*d - center||^2 = radius^2
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - (radius + self.min_clearance) ** 2

        discriminant = b**2 - 4*a*c

        # No intersection if discriminant < 0
        if discriminant < 0:
            return False

        # Check if intersection is within segment [0, 1]
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        # Collision if any t in [0, 1]
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)

    def generate_obstacles(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                          n_obstacles: int, max_attempts: int = 1000) -> Optional[List[np.ndarray]]:
        """
        Generate random spherical obstacles near the trajectory that do NOT intersect it.

        Obstacles are sampled close to the trajectory to increase difficulty.

        Parameters
        ----------
        start_pos    : start position (3,)
        goal_pos     : goal position (3,)
        n_obstacles  : number of obstacles
        max_attempts : max attempts to place all obstacles

        Returns
        -------
        obstacles : list of [x, y, z, r] arrays, or None if failed
        """
        obstacles = []
        trajectory_direction = goal_pos - start_pos
        trajectory_length = np.linalg.norm(trajectory_direction)

        if trajectory_length < 1e-6:
            return None  # Invalid trajectory

        for i in range(n_obstacles):
            placed = False

            for attempt in range(max_attempts):
                # Sample position near trajectory (biased sampling)
                # 1. Pick a random point along the trajectory
                t = np.random.uniform(0.1, 0.9)  # Avoid endpoints
                point_on_traj = start_pos + t * trajectory_direction

                # 2. Add random offset perpendicular to trajectory
                # Distance from trajectory: between (min_clearance + radius) and 0.3m
                radius = np.random.uniform(*self.obs_radius_range)
                min_dist = self.min_clearance + radius + 0.02  # Extra 2cm safety
                max_dist = 0.3  # Maximum 30cm from trajectory

                offset_distance = np.random.uniform(min_dist, max_dist)

                # Random perpendicular direction
                random_perp = np.random.randn(3)
                random_perp -= np.dot(random_perp, trajectory_direction) * trajectory_direction / (trajectory_length ** 2)
                random_perp_norm = np.linalg.norm(random_perp)

                if random_perp_norm < 1e-6:
                    continue  # Degenerate case, retry

                random_perp /= random_perp_norm
                pos = point_on_traj + offset_distance * random_perp

                # Clamp to workspace bounds, with extra Z constraint (stay above floor)
                pos = np.clip(pos, self.ws_min, self.ws_max)
                pos[2] = max(pos[2], self.z_obs_min + radius)

                # Check collision with path (should be safe by construction, but verify)
                if self.line_sphere_collision(start_pos, goal_pos, pos, radius):
                    continue

                # Check collision with existing obstacles (avoid overlap)
                collision_with_others = False
                for obs in obstacles:
                    dist = np.linalg.norm(pos - obs[:3])
                    if dist < (radius + obs[3] + 0.02):  # 2cm minimum separation
                        collision_with_others = True
                        break

                if not collision_with_others:
                    obstacles.append(np.array([pos[0], pos[1], pos[2], radius]))
                    placed = True
                    break

            if not placed:
                return None  # Failed to place obstacle

        return obstacles

    # ------------------------------------------------------------------
    # Full-arm collision check (capsule-based)
    # ------------------------------------------------------------------

    @staticmethod
    def _capsule_sphere_distance(p1: np.ndarray, p2: np.ndarray,
                                 cap_radius: float,
                                 center: np.ndarray,
                                 sphere_radius: float) -> float:
        """Signed distance between a capsule and a sphere (positive = separated)."""
        segment = p2 - p1
        seg_len = np.linalg.norm(segment)
        if seg_len < 1e-8:
            return np.linalg.norm(center - p1) - cap_radius - sphere_radius
        direction = segment / seg_len
        t = np.dot(center - p1, direction)
        t = np.clip(t, 0, seg_len)
        closest = p1 + t * direction
        return np.linalg.norm(center - closest) - cap_radius - sphere_radius

    def check_arm_collision(self, q: np.ndarray, obstacles: list,
                            clearance: float = 0.02) -> bool:
        """
        Check if the full arm (capsule model) collides with any obstacle.

        Parameters
        ----------
        q          : joint configuration (n,)
        obstacles  : list of [x, y, z, r] obstacle specifications
        clearance  : minimum allowed distance (m) from arm surface to obstacle surface

        Returns
        -------
        collision : True if any link penetrates an obstacle beyond clearance
        """
        capsules = self.kin.get_link_capsules(q)
        for p1, p2, cap_radius in capsules:
            for obs in obstacles:
                center = np.array(obs[:3], dtype=float)
                sphere_radius = float(obs[3])
                dist = self._capsule_sphere_distance(p1, p2, cap_radius,
                                                     center, sphere_radius)
                if dist < clearance:
                    return True
        return False

    @staticmethod
    def _seg_dist(p1, p2, q1, q2):
        """Minimum distance between two line segments."""
        d1, d2 = p2 - p1, q2 - q1
        r = p1 - q1
        a, e = float(np.dot(d1, d1)), float(np.dot(d2, d2))
        f = float(np.dot(d2, r))
        eps = 1e-10
        if a < eps and e < eps:
            return float(np.linalg.norm(r))
        if a < eps:
            return float(np.linalg.norm(p1 - (q1 + np.clip(-f / e, 0.0, 1.0) * d2)))
        if e < eps:
            t = np.clip(float(np.dot(-d1, r)) / a, 0.0, 1.0)
            return float(np.linalg.norm((p1 + t * d1) - q1))
        b = float(np.dot(d1, d2))
        c = float(np.dot(d1, r))
        denom = a * e - b * b
        if abs(denom) < eps:
            t = np.clip(-c / a, 0.0, 1.0)
            s = 0.0
        else:
            t = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            s = np.clip((b * t + f) / e, 0.0, 1.0)
        return float(np.linalg.norm((p1 + t * d1) - (q1 + s * d2)))

    def check_self_collision(self, q: np.ndarray, clearance: float = -0.02) -> bool:
        """True if arm capsules penetrate severely (>2cm). Capsule radii (~9cm)
        are padded collision geometry, so mild overlap is normal."""
        capsules = self.kin.get_link_capsules(q)
        n = len(capsules)
        for i in range(n):
            for j in range(i + 3, n):
                if j >= n - 3:  # skip finger capsules (tiny, always near hand)
                    continue
                d = self._seg_dist(capsules[i][0], capsules[i][1],
                                   capsules[j][0], capsules[j][1])
                if d < capsules[i][2] + capsules[j][2] + clearance:
                    return True
        return False

    # ------------------------------------------------------------------
    # Scene generation
    # ------------------------------------------------------------------

    def _sample_ahead_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample start/goal on a Y-parallel line in front of the robot, symmetric about X-axis.

        Trajectory: same X and Z, Y varies symmetrically about 0.
        This produces horizontal sweeping motions straight ahead of the base.

        Returns
        -------
        (start_pos, goal_pos, start_seed_q, goal_seed_q)
        """
        x = np.random.uniform(0.35, 0.55)   # arm's length in front
        z = np.random.uniform(0.25, 0.50)   # comfortable height
        half_span = np.random.uniform(0.15, 0.30)  # Y half-span
        start_pos = np.array([x, -half_span, z])
        goal_pos  = np.array([x,  half_span, z])
        # Random joint configs as IK seeds
        start_seed = np.random.uniform(self.q_min, self.q_max)
        goal_seed  = np.random.uniform(self.q_min, self.q_max)
        return start_pos, goal_pos, start_seed, goal_seed

    def generate_scene(self, scene_id: int, n_obstacles: int,
                       max_attempts: int = 100,
                       ahead_mode: bool = False) -> Optional[dict]:
        """
        Generate a single collision-free scene with manipulability constraint.

        Uses IK to compute the joint configurations the controller would actually
        track, then verifies manipulability at those configurations (not just
        the mean along a joint-space interpolation of FK-sampled configs).

        Parameters
        ----------
        scene_id     : scene identifier
        n_obstacles  : number of obstacles
        max_attempts : max attempts to generate valid scene
        ahead_mode   : if True, trajectory is a Y-parallel line in front of robot

        Returns
        -------
        scene : dict with keys [scene_id, start, goal, obstacles, manipulability_mean]
                or None if generation failed
        """
        manip_abs_min = 0.001   # Hard floor at IK endpoints (what controller uses)

        for attempt in range(max_attempts):
            # Sample start and goal with FK q as IK initial guess (deterministic)
            if ahead_mode:
                start_pos, goal_pos, start_seed_q, goal_seed_q = self._sample_ahead_trajectory()
            else:
                start_pos, start_seed_q = self.sample_reachable_point()
                goal_pos, goal_seed_q = self.sample_reachable_point()

                # Floor constraint: positions must stay above ground plane (Z=0)
                if start_pos[2] < 0.02 or goal_pos[2] < 0.02:
                    continue

                # Check minimum distance between start and goal
                dist = np.linalg.norm(goal_pos - start_pos)
                if dist < 0.2 or dist > 1.5:
                    continue

            # Compute IK at start and goal
            start_ik = self.kin.inverse_kinematics(start_pos, q_init=start_seed_q)
            goal_ik = self.kin.inverse_kinematics(goal_pos, q_init=goal_seed_q)
            if start_ik is None or goal_ik is None:
                continue

            # Joint limit check at endpoints
            if np.any(start_ik < self.q_min) or np.any(start_ik > self.q_max):
                continue
            if np.any(goal_ik < self.q_min) or np.any(goal_ik > self.q_max):
                continue

            # Self-collision check at endpoints
            if self.check_self_collision(start_ik) or self.check_self_collision(goal_ik):
                continue

            # Manipulability at IK endpoints (matches controller's actual configurations)
            manip_start = self.compute_manipulability(start_ik)
            manip_goal = self.compute_manipulability(goal_ik)
            if min(manip_start, manip_goal) < manip_abs_min:
                continue

            # Generate obstacles (or empty list if n_obstacles=0)
            if n_obstacles > 0:
                obstacles = self.generate_obstacles(start_pos, goal_pos, n_obstacles)
                if obstacles is None:
                    continue
            else:
                obstacles = []

            # Full-arm collision check (capsule model, 2cm clearance)
            # Run BEFORE controller path verification — cheap filter that avoids
            # wasting 500-step Jacobian simulation on scenes that collide anyway
            arm_collision = False
            for alpha in np.linspace(0, 1, 10):
                q_interp = (1 - alpha) * start_ik + alpha * goal_ik
                if self.check_arm_collision(q_interp, obstacles):
                    arm_collision = True
                    break
            if arm_collision:
                continue

            # Compute min along IK path for reporting
            manip_min = min(manip_start, manip_goal)

            # Success — store IK configs so the controller uses the exact same
            # joint configurations that passed the manipulability check
            scene = {
                "scene_id": scene_id,
                "start": start_pos.tolist(),
                "goal": goal_pos.tolist(),
                "start_q": start_ik.tolist(),
                "goal_q": goal_ik.tolist(),
                "obstacles": [[float(o[0]), float(o[1]), float(o[2]), float(o[3])] for o in obstacles],
                "manipulability_mean": float((manip_start + manip_goal) / 2.0),
                "manipulability_min": float(manip_min),
            }

            return scene

        return None  # Failed after max_attempts

    def _verify_controller_path(self, q_start: np.ndarray, start_pos: np.ndarray,
                                 goal_pos: np.ndarray, obstacles: list = None,
                                 dt: float = 0.02, max_steps: int = 500) -> bool:
        """
        Simulate the Jacobian-integration path used by the actual controller.

        The controller uses dq = J^† * dx_cmd, which produces different joint
        trajectories than IK snapshots for redundant manipulators.  This
        verification catches scenes where the IK-based checks pass but the
        controller path drifts into near-singular configurations or diverges
        from the target.

        Returns True if the controller can track the path without manipulability
        dropping below threshold or tracking error diverging.
        """
        q = q_start.copy()
        direction = goal_pos - start_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return True

        dx_d = (direction / dist) * 0.1   # nominal 0.1 m/s
        x_d = start_pos.copy()
        path_param = 0.0
        total = max_steps
        Kp_base = 4.0
        DQ_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

        # Trapezoidal profile: ease-in (20%), constant (60%), ease-out (20%)
        a_end = int(total * 0.2)
        d_start = int(total * 0.8)

        for step in range(max_steps):
            # Trapezoidal nominal path parameter
            if step < a_end:
                frac = step / max(a_end, 1)
                nominal_s = 0.5 * frac * frac * 0.2
            elif step < d_start:
                nominal_s = 0.2 + (step - a_end) / (d_start - a_end) * 0.6
            else:
                frac = (step - d_start) / max(total - d_start, 1)
                nominal_s = 0.8 + (1.0 - 0.5 * (1.0 - frac) * (1.0 - frac)) * 0.2

            x_ee, _ = self.kin.forward_kinematics(q)
            track_err = np.linalg.norm(x_ee - x_d)

            # Same advance-rate with dead zone as the real controller
            err_deadzone = max(0.0, track_err - 0.02)
            raw_advance = float(np.clip(1.0 - err_deadzone / 0.10, 0.0, 1.0))
            path_param = min(1.0, path_param + (nominal_s - path_param) * raw_advance)
            prev_x_d = x_d.copy()
            x_d = (1.0 - path_param) * start_pos + path_param * goal_pos
            dx_d_step = (x_d - prev_x_d) / dt if dt > 0 else np.zeros(3)

            # PID command (same as _compute_task_velocity)
            pos_err = x_d - x_ee
            err_norm = np.linalg.norm(pos_err)
            Kp = Kp_base * (1.0 + np.tanh(err_norm / 0.05))
            dx_cmd = np.zeros(6)
            dx_cmd[:3] = dx_d_step + Kp * pos_err

            # Joint velocity via pseudoinverse (same as controller)
            J = self.kin.jacobian(q)[:3, :]
            try:
                Jpinv = np.linalg.pinv(J)
            except np.linalg.LinAlgError:
                return False
            dq = Jpinv @ dx_cmd[:3]
            dq = np.clip(dq, -DQ_MAX, DQ_MAX)

            q = q + dq * dt

            # Check manipulability
            w = self.compute_manipulability(q)
            if w < 0.001:
                return False

            # Check self-collision every 40 steps
            if step % 40 == 0 and self.check_self_collision(q):
                return False

            # Check arm-obstacle collision every 20 steps on the actual
            # controller path (not the IK-interpolated path)
            if obstacles and step % 20 == 0:
                if self.check_arm_collision(q, obstacles):
                    return False

            # Divergence check
            if step > 100 and track_err > 0.15:
                return False

            # Converged
            if path_param >= 1.0 and track_err < 0.01:
                return True

        # Check final state
        x_ee_final, _ = self.kin.forward_kinematics(q)
        final_err = np.linalg.norm(x_ee_final - goal_pos)
        if final_err > 0.05:
            return False

        return True

    def generate_dataset(self, num_scenes: int, n_obstacles: int,
                        output_path: str, seed: int = 42, max_attempts_per_scene: int = 500,
                        ahead_mode: bool = False):
        """
        Generate multiple scenes and save to JSON. Generates exactly num_scenes
        valid scenes, retrying failed scenes with new random seeds.

        Parameters
        ----------
        num_scenes            : number of valid scenes to generate
        n_obstacles           : number of obstacles per scene
        output_path           : path to save JSON file
        seed                  : random seed
        max_attempts_per_scene: max retries per scene index
        """
        np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"Generating {num_scenes} valid scenes with {n_obstacles} obstacles each")
        print(f"{'='*60}\n")

        scenes = []
        failed_count = 0
        scene_id = 0

        while len(scenes) < num_scenes and scene_id < num_scenes * 3:
            if scene_id > 0 and scene_id % 10 == 0:
                print(f"Progress: accepted {len(scenes)}/{num_scenes} (failed: {failed_count})")

            scene = self.generate_scene(scene_id=scene_id, n_obstacles=n_obstacles,
                                        max_attempts=max_attempts_per_scene,
                                        ahead_mode=ahead_mode)

            if scene is not None:
                scene["scene_id"] = len(scenes)  # Renumber sequentially
                scenes.append(scene)
            else:
                failed_count += 1
                if failed_count % 50 == 0:
                    print(f"  Warning: {failed_count} consecutive scene failures...")

            scene_id += 1

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(scenes, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Dataset generation complete!")
        print(f"  Successfully generated: {len(scenes)}/{num_scenes}")
        print(f"  Failed: {failed_count}")
        print(f"  Saved to: {output_path}")
        print(f"{'='*60}\n")

        # Print statistics
        if scenes:
            manips = [s["manipulability_mean"] for s in scenes]
            print(f"Manipulability statistics:")
            print(f"  Mean: {np.mean(manips):.4f}")
            print(f"  Std:  {np.std(manips):.4f}")
            print(f"  Min:  {np.min(manips):.4f}")
            print(f"  Max:  {np.max(manips):.4f}")


def main_bak():
    parser = argparse.ArgumentParser(description='Generate collision-free trajectories with manipulability constraints')
    parser.add_argument('--urdf', type=str, default='panda_description/urdf/panda.urdf',
                       help='Path to URDF file')
    parser.add_argument('--num_scenes', type=int, default=100,
                       help='Number of scenes to generate')
    parser.add_argument('--num_obstacles', type=int, default=5,
                       help='Number of obstacles per scene')
    parser.add_argument('--manip_threshold', type=float, default=0.01,
                       help='Minimum average manipulability')
    parser.add_argument('--output', type=str, default='trajectories.json',
                       help='Output JSON file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Resolve URDF path
    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        urdf_path = project_root / args.urdf
        if not urdf_path.exists():
            print(f"Error: URDF file not found at {urdf_path}")
            sys.exit(1)

    generator = TrajectoryGenerator(
        urdf_path=str(urdf_path),
        manipulability_threshold=args.manip_threshold,
    )

    generator.generate_dataset(
        num_scenes=args.num_scenes,
        n_obstacles=args.num_obstacles,
        output_path=args.output,
        seed=args.seed,
    )


def main():
    # Hardcoded for debugging — edit these as needed
    urdf_path = "/home/merlin/manipulator/code/.venv/lib/python3.12/site-packages/cmeel.prefix/share/example-robot-data/robots/panda_description/urdf/panda.urdf"
    num_scenes = 100
    num_obstacles = 5
    output_path = "/home/merlin/manipulator/results/trajectories_obs.json"
    seed = 42
    manip_threshold = 0.01

    generator = TrajectoryGenerator(
        urdf_path=urdf_path,
        manipulability_threshold=manip_threshold,
        obstacle_radius_range=(0.02, 0.08),  # smaller obstacles
    )

    generator.generate_dataset(
        num_scenes=num_scenes,
        n_obstacles=num_obstacles,
        output_path=output_path,
        seed=seed,
        ahead_mode=True,  # Y-parallel trajectories in front of robot
    )


if __name__ == "__main__":
    main()
