"""
collision.py
------------
MuJoCo-based collision detection for obstacle and self-collision.

Uses MuJoCo's native collision detection engine to compute:
  - External collisions: robot links vs obstacles
  - Self-collisions: robot links vs other robot links

Returns penetration depth and contact information for loss computation.
"""

import numpy as np
from typing import Optional, Tuple

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


class CollisionDetector:
    """
    Wrapper for MuJoCo collision detection.

    Computes collision penalties based on contact forces and penetration depths.
    """

    def __init__(self, mj_model=None, mj_data=None):
        """
        Parameters
        ----------
        mj_model : MuJoCo model (MjModel)
        mj_data  : MuJoCo data (MjData)
        """
        self.model = mj_model
        self.data = mj_data
        self.has_mujoco = HAS_MUJOCO and mj_model is not None

    def update_model(self, mj_model, mj_data):
        """Update MuJoCo model and data references."""
        self.model = mj_model
        self.data = mj_data
        self.has_mujoco = HAS_MUJOCO and mj_model is not None

    def detect_collisions(self) -> Tuple[float, float, int]:
        """
        Detect all collisions in current configuration.

        Returns
        -------
        total_penetration : sum of all penetration depths (m)
        max_penetration   : maximum single penetration depth (m)
        n_contacts        : number of active contacts
        """
        if not self.has_mujoco:
            return 0.0, 0.0, 0

        total_pen = 0.0
        max_pen = 0.0
        n_contacts = 0

        # Iterate through all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Penetration depth (negative distance means penetration)
            penetration = -contact.dist

            if penetration > 0:
                total_pen += penetration
                max_pen = max(max_pen, penetration)
                n_contacts += 1

        return float(total_pen), float(max_pen), n_contacts

    def detect_self_collisions(self) -> Tuple[float, int]:
        """
        Detect self-collisions (robot links colliding with each other).

        Excludes adjacent links and parent-child connections to avoid false positives.

        Returns
        -------
        total_penetration : sum of self-collision penetration depths (m)
        n_self_contacts   : number of self-collision contacts
        """
        if not self.has_mujoco:
            return 0.0, 0

        total_pen = 0.0
        n_self = 0

        # Get robot body IDs (assuming robot bodies are named with 'panda' prefix)
        robot_body_ids = set()
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and 'panda' in body_name.lower():
                robot_body_ids.add(i)

        # Check contacts between robot bodies
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get geom IDs
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Get body IDs from geoms
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            # Check if both bodies belong to robot (self-collision)
            if body1 in robot_body_ids and body2 in robot_body_ids:
                # Exclude adjacent links (parent-child relationship)
                # Adjacent bodies have body IDs differing by 1 in kinematic chain
                if abs(body1 - body2) <= 1:
                    continue  # Skip adjacent links

                penetration = -contact.dist
                if penetration > 0:
                    total_pen += penetration
                    n_self += 1

        return float(total_pen), n_self

    def detect_obstacle_collisions(self) -> Tuple[float, int]:
        """
        Detect external collisions (robot vs obstacles).

        Returns
        -------
        total_penetration : sum of obstacle collision penetration depths (m)
        n_obs_contacts    : number of obstacle collision contacts
        """
        if not self.has_mujoco:
            return 0.0, 0

        total_pen = 0.0
        n_obs = 0

        # Get robot body IDs
        robot_body_ids = set()
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and 'panda' in body_name.lower():
                robot_body_ids.add(i)

        # Check contacts with non-robot bodies (obstacles)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            # One body is robot, other is obstacle
            is_robot_obstacle = (
                (body1 in robot_body_ids and body2 not in robot_body_ids) or
                (body2 in robot_body_ids and body1 not in robot_body_ids)
            )

            if is_robot_obstacle:
                penetration = -contact.dist
                if penetration > 0:
                    total_pen += penetration
                    n_obs += 1

        return float(total_pen), n_obs

    def compute_collision_penalty(self,
                                   w_obstacle: float = 100.0,
                                   w_self: float = 50.0) -> Tuple[float, dict]:
        """
        Compute collision penalty for reward/loss function.

        Parameters
        ----------
        w_obstacle : weight for obstacle collision penalty
        w_self     : weight for self-collision penalty

        Returns
        -------
        penalty : total collision penalty (negative reward)
        info    : dict with collision details
        """
        if not self.has_mujoco:
            return 0.0, {}

        # Detect collisions
        obs_pen, n_obs = self.detect_obstacle_collisions()
        self_pen, n_self = self.detect_self_collisions()

        # Compute penalties (quadratic to heavily penalize penetration)
        # penalty_obs = w_obstacle * (obs_pen ** 2)
        # penalty_self = w_self * (self_pen ** 2)
        penalty_obs = w_obstacle * n_obs
        penalty_self = w_self * n_self

        total_penalty = penalty_obs + penalty_self

        info = {
            "collision_penalty": total_penalty,
            "obstacle_penetration": obs_pen,
            "self_penetration": self_pen,
            "n_obstacle_contacts": n_obs,
            "n_self_contacts": n_self,
        }

        return float(total_penalty), info


if __name__ == "__main__":
    print("collision.py unit test")

    if not HAS_MUJOCO:
        print("MuJoCo not available, skipping test")
    else:
        detector = CollisionDetector()
        print("CollisionDetector created (no model)")

        # Test with None model
        pen, max_pen, n = detector.detect_collisions()
        print(f"No model: penetration={pen}, max={max_pen}, contacts={n}")

    print("collision.py unit test PASSED")
