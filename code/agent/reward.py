"""
reward.py
---------
Multi-component reward function (paper Section 3.3):
    r = r_track + r_obs + r_manip + r_energy + r_collision

  r_track     : end-effector tracking error with dynamic weight (paper Eq. 12)
                r_track = -w_track_eff * ||x_ee - x_d||²
                w_track_eff decreases near obstacles to allow task relaxation
  r_obs       : obstacle avoidance (SDF-based, large penalty near collision)
  r_manip     : manipulability bonus (encourage non-singular configs)
  r_energy    : energy penalty (penalize large joint torques, not velocities)
  r_collision : MuJoCo collision penalty (obstacle + self-collision)
"""

import numpy as np
from typing import Optional


class RewardFunction:

    def __init__(self,
                 w_track:       float = 3.0,
                 w_obs:         float = 5.0,
                 w_obs_safe:    float = 0.1,
                 w_manip:       float = 0.05,
                 w_energy:      float = 0.001,
                 w_collision:   float = 100.0,
                 d_safe:        float = 0.06,
                 d_critical:    float = 0.02,
                 alpha_relax:   float = 0.1,
                 dt:            float = 0.02,
                 collision_detector = None):
        self.w_track       = w_track
        self.w_obs         = w_obs
        self.w_obs_safe    = w_obs_safe
        self.w_manip       = w_manip
        self.w_energy      = w_energy
        self.w_collision   = w_collision
        self.d_safe        = d_safe
        self.d_critical    = d_critical
        self.alpha_relax   = alpha_relax   # minimum weight factor when d_obs < d_critical
        self.dt            = dt
        self.collision_detector = collision_detector

    def _effective_track_weight(self, d_obs: float) -> float:
        """
        Dynamic tracking weight (paper Eq. 12, primary task relaxation mechanism).

        w_track_eff = w_track * (alpha_relax + (1-alpha_relax) * d_obs/d_critical)
        when d_obs < d_critical, otherwise w_track_eff = w_track.

        When d_obs is large: w_track_eff = w_track (full tracking)
        When d_obs → 0:      w_track_eff = alpha_relax * w_track (relaxed tracking)
        """
        if d_obs >= self.d_critical:
            return self.w_track
        ratio = max(d_obs / self.d_critical, 0.0)  # clamp for d_obs < 0 (inside obstacle)
        return self.w_track * (self.alpha_relax + (1.0 - self.alpha_relax) * ratio)

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
        # Tracking reward: squared position error with dynamic weight (paper Eq. 12)
        pos_err = np.linalg.norm(x_ee - x_d)
        w_eff = self._effective_track_weight(d_obs)
        r_track = -w_eff * pos_err ** 2

        # Obstacle reward: positive bonus when safe, dense penalty when close
        if d_obs >= self.d_safe:
            r_obs = self.w_obs_safe * min(d_obs / self.d_safe, 2.0)  # positive for staying safe
        else:
            obs_depth = min(self.d_safe - d_obs, self.d_safe * 2.0)  # cap at 2x d_safe
            r_obs = -self.w_obs * obs_depth / self.d_safe

        # Manipulability reward: encourage non-singular configurations
        r_manip = self.w_manip * np.log(max(w, 1e-4))
        r_manip = max(r_manip, -0.5)  # cap negative spikes near singularity

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
            "w_track_eff": w_eff,   # for logging the dynamic weight
            **collision_info
        }
        return float(total), info
