"""
reward.py
---------
Multi-component reward function:
    r = r_track + r_obs + r_manip + r_energy + r_collision

  r_track     : end-effector tracking error (dense, negative L2)
  r_obs       : obstacle avoidance (SDF-based, large penalty near collision)
  r_manip     : manipulability bonus (encourage non-singular configs)
  r_energy    : energy penalty (penalize large joint velocities)
  r_collision : MuJoCo collision penalty (obstacle + self-collision)
"""

import numpy as np
from typing import Optional


class RewardFunction:

    def __init__(self,
                 w_track:     float = 1.0,
                 w_obs:       float = 5.0,
                 w_manip:     float = 0.1,
                 w_energy:    float = 0.01,
                 w_collision: float = 10.0,
                 d_safe:      float = 0.10,
                 dt:          float = 0.02,
                 collision_detector = None):
        self.w_track     = w_track
        self.w_obs       = w_obs
        self.w_manip     = w_manip
        self.w_energy    = w_energy
        self.w_collision = w_collision
        self.d_safe      = d_safe   # safe distance threshold
        self.dt          = dt
        self.collision_detector = collision_detector

    def compute(self, q, dq, x_ee, x_d, dx_d, d_obs, w):
        """
        Parameters
        ----------
        q     : joint positions [n]
        dq    : joint velocities [n]
        x_ee  : end-effector position [3]
        x_d   : desired EE position [3]
        dx_d  : desired EE velocity [6] (unused here, for extension)
        d_obs : minimum distance to any obstacle (scalar)
        w     : manipulability measure (scalar)

        Returns
        -------
        total_reward : float
        info         : dict with individual components
        """
        # Tracking reward: negative Gaussian of position error
        pos_err = np.linalg.norm(x_ee - x_d)
        r_track = -self.w_track * pos_err

        # Obstacle reward: 0 if safe, large negative if within safety margin
        if d_obs >= self.d_safe:
            r_obs = 0.0
        else:
            r_obs = -self.w_obs * (self.d_safe - d_obs) / self.d_safe

        # Manipulability reward: encourage higher manipulability
        r_manip = self.w_manip * np.log(w + 1e-6)

        # Energy penalty: penalize large joint velocities
        r_energy = -self.w_energy * np.sum(dq ** 2)

        # Collision penalty: MuJoCo-based collision detection
        r_collision = 0.0
        collision_info = {}
        if self.collision_detector is not None:
            collision_penalty, collision_info = self.collision_detector.compute_collision_penalty(
                w_obstacle=self.w_collision,
                w_self=self.w_collision * 0.5
            )
            r_collision = -collision_penalty

        total = r_track + r_obs + r_manip + r_energy + r_collision

        info = {
            "r_track":     r_track,
            "r_obs":       r_obs,
            "r_manip":     r_manip,
            "r_energy":    r_energy,
            "r_collision": r_collision,
            **collision_info
        }
        return float(total), info
