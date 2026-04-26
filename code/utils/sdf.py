"""
sdf.py
------
Obstacle signed distance field using spherical obstacle primitives.
Obstacles are represented as spheres; manipulator links are modeled as capsules.
Distance computation: capsule-to-sphere using geometric primitives.
"""

import numpy as np


class ObstacleSDF:
    """
    N spherical obstacles in Cartesian space.
    Distance from capsule-modeled links to the nearest obstacle surface.
    """

    def __init__(self, n_obstacles: int = 3, radius: float = 0.1):
        self.n_obs = n_obstacles
        self.radius = radius
        self.centers = np.zeros((n_obstacles, 3))

    def randomize_obstacles(self, center: np.ndarray, margin: float = 0.5):
        """Place obstacles randomly around `center` with given margin."""
        for i in range(self.n_obs):
            offset = np.random.uniform(-margin, margin, 3)
            self.centers[i] = center + offset

    def set_obstacles(self, centers: np.ndarray):
        """Manually set obstacle centers. centers: [N x 3]"""
        self.centers = np.asarray(centers)

    def set_static_obstacles(self, centers: list):
        """Set static obstacle positions for Scenario 1."""
        self.centers = np.array(centers)
        self.n_obs = len(centers)

    def point_distance(self, point: np.ndarray) -> float:
        """
        Minimum signed distance from a 3D point to the nearest obstacle surface.
        Positive = outside, negative = inside (collision).
        """
        if self.n_obs == 0:
            return np.inf
        dists = np.linalg.norm(self.centers - point, axis=1) - self.radius
        return float(np.min(dists))

    def capsule_to_sphere_distance(self, p1: np.ndarray, p2: np.ndarray,
                                   capsule_radius: float,
                                   sphere_center: np.ndarray) -> float:
        """
        Compute signed distance between a capsule and a sphere.

        Capsule: line segment from p1 to p2 with radius capsule_radius
        Sphere: center at sphere_center with radius self.radius

        Returns
        -------
        distance : float
            Signed distance (positive = separated, negative = penetration)
        """
        # Vector from p1 to p2
        segment = p2 - p1
        segment_length = np.linalg.norm(segment)

        if segment_length < 1e-8:
            # Degenerate capsule (point), treat as sphere-sphere
            center_dist = np.linalg.norm(sphere_center - p1)
            return center_dist - capsule_radius - self.radius

        # Normalized direction
        direction = segment / segment_length

        # Project sphere center onto line segment
        t = np.dot(sphere_center - p1, direction)
        t = np.clip(t, 0, segment_length)  # Clamp to segment

        # Closest point on segment to sphere center
        closest_point = p1 + t * direction

        # Distance from sphere center to closest point on capsule axis
        center_dist = np.linalg.norm(sphere_center - closest_point)

        # Signed distance accounting for both radii
        return center_dist - capsule_radius - self.radius

    def min_distance(self, x_ee: np.ndarray, q: np.ndarray | None = None,
                     kinematics=None) -> float:
        """
        Minimum signed distance from manipulator links to the nearest obstacle.

        Parameters
        ----------
        x_ee       : end-effector position [3] (fallback if q/kinematics not provided)
        q          : joint positions [n] (required for capsule-based distance)
        kinematics : ManipulatorKinematics instance (required for capsule-based distance)

        Returns
        -------
        min_dist : float
            Minimum signed distance across all links and obstacles
        """
        if self.n_obs == 0:
            return np.inf

        # If kinematics available, use capsule-based distance
        if q is not None and kinematics is not None:
            capsules = kinematics.get_link_capsules(q)

            min_dist = np.inf
            for p1, p2, cap_radius in capsules:
                for obs_center in self.centers:
                    dist = self.capsule_to_sphere_distance(p1, p2, cap_radius, obs_center)
                    min_dist = min(min_dist, dist)

            return float(min_dist)
        else:
            # Fallback: point-based distance (end-effector only)
            dists = np.linalg.norm(self.centers - x_ee, axis=1) - self.radius
            return float(np.min(dists))
