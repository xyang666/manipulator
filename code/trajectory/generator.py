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
        self.kin = ManipulatorKinematics(urdf_path=urdf_path, n_joints=n_joints)
        self.n = n_joints

        # Joint limits (Panda default)
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        # Workspace bounds (default: empirical Panda workspace)
        if workspace_bounds is None:
            self.ws_min = np.array([-0.85, -0.85, -0.35])
            self.ws_max = np.array([0.85, 0.85, 1.20])
        else:
            self.ws_min, self.ws_max = workspace_bounds

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

                # Clamp to workspace bounds
                pos = np.clip(pos, self.ws_min, self.ws_max)

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

    def generate_scene(self, scene_id: int, n_obstacles: int,
                      max_attempts: int = 100) -> Optional[dict]:
        """
        Generate a single collision-free scene with manipulability constraint.

        Parameters
        ----------
        scene_id     : scene identifier
        n_obstacles  : number of obstacles
        max_attempts : max attempts to generate valid scene

        Returns
        -------
        scene : dict with keys [scene_id, start, goal, obstacles, manipulability_mean]
                or None if generation failed
        """
        for attempt in range(max_attempts):
            # Sample start and goal (now guaranteed reachable)
            start_pos, start_q = self.sample_reachable_point()
            goal_pos, goal_q = self.sample_reachable_point()

            # Check minimum distance between start and goal
            dist = np.linalg.norm(goal_pos - start_pos)
            if dist < 0.2 or dist > 1.5:  # Too close or too far
                continue

            # Generate obstacles
            obstacles = self.generate_obstacles(start_pos, goal_pos, n_obstacles)
            if obstacles is None:
                continue

            # Compute manipulability
            manip_mean = self.compute_path_manipulability(start_q, goal_q)

            # Check manipulability threshold
            if manip_mean < self.manip_threshold:
                continue

            # Success!
            scene = {
                "scene_id": scene_id,
                "start": start_pos.tolist(),
                "goal": goal_pos.tolist(),
                "obstacles": [obs.tolist() for obs in obstacles],
                "manipulability_mean": float(manip_mean)
            }

            return scene

        return None  # Failed after max_attempts

    def generate_dataset(self, num_scenes: int, n_obstacles: int,
                        output_path: str, seed: int = 42):
        """
        Generate multiple scenes and save to JSON.

        Parameters
        ----------
        num_scenes   : number of scenes to generate
        n_obstacles  : number of obstacles per scene
        output_path  : path to save JSON file
        seed         : random seed
        """
        np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"Generating {num_scenes} scenes with {n_obstacles} obstacles each")
        print(f"{'='*60}\n")

        scenes = []
        failed_count = 0

        for i in range(num_scenes):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{num_scenes} (failed: {failed_count})")

            scene = self.generate_scene(scene_id=i, n_obstacles=n_obstacles)

            if scene is not None:
                scenes.append(scene)
            else:
                failed_count += 1
                print(f"  Warning: Failed to generate scene {i}")

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


def main():
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

    # Create generator
    generator = TrajectoryGenerator(
        urdf_path=str(urdf_path),
        manipulability_threshold=args.manip_threshold
    )

    # Generate dataset
    generator.generate_dataset(
        num_scenes=args.num_scenes,
        n_obstacles=args.num_obstacles,
        output_path=args.output,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
