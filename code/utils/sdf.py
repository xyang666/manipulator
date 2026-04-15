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
        Minimum signed distance from any robot point to the nearest obstacle.

        Checks the end-effector and, if kinematics is provided, all link
        joint origins via forward kinematics (Pinocchio).

        Parameters
        ----------
        x_ee       : end-effector position [3]
        q          : joint positions [n], required when kinematics is given
        kinematics : ManipulatorKinematics instance (optional)
        """
        if self.n_obs == 0:
            return np.inf

        # Collect robot sample points: always include end-effector
        points = [x_ee]

        if kinematics is not None and q is not None and kinematics.model is not None:
            import pinocchio as pin
            q_arr = np.asarray(q, dtype=float)
            pin.forwardKinematics(kinematics.model, kinematics.data, q_arr)
            pin.updateFramePlacements(kinematics.model, kinematics.data)
            # Sample each joint origin (frame type == JOINT)
            for frame in kinematics.model.frames:
                if frame.type == pin.FrameType.JOINT:
                    fid = kinematics.model.getFrameId(frame.name)
                    pos = kinematics.data.oMf[fid].translation.copy()
                    points.append(pos)

        # Compute minimum distance across all sample points and all obstacles
        # centers: [N x 3], points stacked: [P x 3]
        pts = np.array(points)                          # [P x 3]
        # dists[p, o] = distance from point p to obstacle o surface
        diffs = pts[:, None, :] - self.centers[None, :, :]  # [P x N x 3]
        dists = np.linalg.norm(diffs, axis=2) - self.radius  # [P x N]
        return float(np.min(dists))
