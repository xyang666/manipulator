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
    Supports variable radii for each obstacle.
    """

    def __init__(self, n_obstacles: int = 3, radius: float = 0.1):
        self.n_obs = n_obstacles
        self.default_radius = radius
        # Store radii as array (one per obstacle)
        self.radii = np.full(n_obstacles, radius)
        self.centers = np.zeros((n_obstacles, 3))

    def randomize_obstacles(self, center: np.ndarray, margin: float = 0.5):
        """Place obstacles randomly around `center` with given margin."""
        for i in range(self.n_obs):
            offset = np.random.uniform(-margin, margin, 3)
            self.centers[i] = center + offset

    def set_obstacles(self, centers: np.ndarray, radii: np.ndarray | None = None):
        """
        Manually set obstacle centers and optionally radii.

        Parameters
        ----------
        centers : [N x 3] array of obstacle centers
        radii   : [N] array of obstacle radii (optional, uses default if None)
        """
        self.centers = np.asarray(centers)
        self.n_obs = len(centers)

        if radii is not None:
            self.radii = np.asarray(radii)
        else:
            self.radii = np.full(self.n_obs, self.default_radius)

    def set_static_obstacles(self, centers: list, radii: list | None = None):
        """
        Set static obstacle positions and optionally radii.

        Parameters
        ----------
        centers : list of [x, y, z] positions
        radii   : list of radii (optional, uses default if None)
        """
        self.centers = np.array(centers)
        self.n_obs = len(centers)

        if radii is not None:
            self.radii = np.array(radii)
        else:
            self.radii = np.full(self.n_obs, self.default_radius)

    def point_distance(self, point: np.ndarray) -> float:
        """
        Minimum signed distance from a 3D point to the nearest obstacle surface.
        Positive = outside, negative = inside (collision).
        """
        if self.n_obs == 0:
            return np.inf
        dists = np.linalg.norm(self.centers - point, axis=1) - self.radii
        return float(np.min(dists))

    def capsule_to_sphere_distance(self, p1: np.ndarray, p2: np.ndarray,
                                   capsule_radius: float,
                                   sphere_center: np.ndarray,
                                   sphere_radius: float) -> float:
        """
        Compute signed distance between a capsule and a sphere.

        Capsule: line segment from p1 to p2 with radius capsule_radius
        Sphere: center at sphere_center with radius sphere_radius

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
            return center_dist - capsule_radius - sphere_radius

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
        return center_dist - capsule_radius - sphere_radius

    def top_k(self, point: np.ndarray, K: int = 5) -> tuple:
        """
        Return top-K nearest obstacle distances, directions, and mask from a point.

        Parameters
        ----------
        point : 3D query position
        K     : number of nearest obstacles to return

        Returns
        -------
        dists : (K,) signed distances from point to obstacle surface (inf for padding)
        dirs  : (K, 3) unit direction vectors from point to obstacle center (zero for padding)
        mask  : (K,) 1.0 for real obstacles, 0.0 for padding
        """
        if self.n_obs == 0:
            return (np.full(K, 0.5, dtype=np.float32),
                    np.zeros((K, 3), dtype=np.float32),
                    np.zeros(K, dtype=np.float32))

        # Vector from point to each obstacle center
        vecs = self.centers - point  # (N, 3)
        center_dists = np.linalg.norm(vecs, axis=1)  # (N,)
        signed_dists = center_dists - self.radii  # (N,) signed to surface

        # Sort by signed distance (closest first)
        idx = np.argsort(signed_dists)
        n = min(K, self.n_obs)

        # Use large finite value instead of np.inf for padded entries,
        # since inf flowing through the neural network produces NaN.
        # The mask tells the network to ignore padded entries.
        result_dists = np.full(K, 0.5, dtype=np.float32)
        result_dirs = np.zeros((K, 3), dtype=np.float32)
        result_mask = np.zeros(K, dtype=np.float32)

        for i in range(n):
            j = idx[i]
            result_dists[i] = float(np.clip(signed_dists[j], -0.5, 0.5))
            if center_dists[j] > 1e-6:
                result_dirs[i] = vecs[j] / center_dists[j]
            result_mask[i] = 1.0

        return result_dists, result_dirs, result_mask

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
                for i, obs_center in enumerate(self.centers):
                    dist = self.capsule_to_sphere_distance(
                        p1, p2, cap_radius, obs_center, self.radii[i]
                    )
                    min_dist = min(min_dist, dist)

            return float(min_dist)
        else:
            # Fallback: point-based distance (end-effector only)
            dists = np.linalg.norm(self.centers - x_ee, axis=1) - self.radii
            return float(np.min(dists))
