"""
sdf.py
------
Obstacle signed distance field using spherical obstacle primitives.
Obstacles are represented as spheres; each manipulator link is also
approximated as a sphere at its joint origin for collision checking.
"""

import numpy as np


class ObstacleSDF:
    """
    N spherical obstacles in Cartesian space.
    Distance from a query point to the nearest obstacle surface.
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

    def min_distance(self, x_ee: np.ndarray, q: np.ndarray | None = None,
                     kinematics=None) -> float:
        """
        Minimum signed distance from end-effector to the nearest obstacle.

        Parameters
        ----------
        x_ee       : end-effector position [3]
        q          : joint positions [n] (unused, kept for API compatibility)
        kinematics : ManipulatorKinematics instance (unused, kept for API compatibility)
        """
        if self.n_obs == 0:
            return np.inf

        # Only compute distance for end-effector
        dists = np.linalg.norm(self.centers - x_ee, axis=1) - self.radius
        return float(np.min(dists))
