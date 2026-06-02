"""
reward.py
---------
Multi-component reward function:
    r = r_track + r_obs + r_manip + r_energy + r_collision + r_action

  r_track     : end-effector tracking error (exponential, fixed weight)
  r_obs       : obstacle avoidance (SDF-based, per-capsule penalty)
  r_manip     : manipulability bonus (encourage non-singular configs)
  r_energy    : energy penalty (penalize large joint velocities)
  r_collision : MuJoCo collision penalty
"""

import numpy as np
from typing import Optional


class RewardFunction:

    def __init__(self,
                 w_track:       float = 12.0,
                 w_obs:         float = 1.0,
                 w_manip:       float = 0.05,
                 w_energy:      float = 0.001,
                 w_collision:   float = 100.0,
                 w_action:      float = 0.5,
                 d_safe:        float = 0.06,
                 dt:            float = 0.02,
                 collision_detector = None):
        self.w_track       = w_track
        self.w_obs         = w_obs
        self.w_manip       = w_manip
        self.w_energy      = w_energy
        self.w_collision   = w_collision
        self.w_action      = w_action
        self.d_safe        = d_safe
        self.dt            = dt
        self.collision_detector = collision_detector

    def compute(self, q, dq, x_ee, x_d, dx_d, d_obs, w, action=None, prev_dq=None,
                capsule_dists=None):
        """
        Parameters
        ----------
        q       : joint positions [n]
        dq      : joint velocities [n]
        x_ee    : end-effector position [3]
        x_d     : desired EE position [3]
        dx_d    : desired EE velocity [6] (unused here, for extension)
        d_obs   : minimum distance to any obstacle (scalar)
        w       : manipulability measure (scalar)
        action  : RL action [7] = [Δẋ_RL(3), z(4)] (deprecated, use prev_dq instead)
        prev_dq : previous step joint velocities [n] (for smoothness penalty)
        capsule_dists : per-capsule distances [n_caps] (optional)

        Returns
        -------
        total_reward : float
        info         : dict with individual components
        """
        # Tracking reward: exponential of position error.
        # Fixed w_track (no dynamic relaxation — Lagrangian λ handles gating).
        pos_err = np.linalg.norm(x_ee - x_d)
        r_track = self.w_track * np.exp(-self.w_track * pos_err)

        # Obstacle reward: per-capsule dense penalty.
        if capsule_dists is not None:
            total_penalty = 0.0
            for d_cap in capsule_dists:
                if d_cap < self.d_safe:
                    depth = min(self.d_safe - d_cap, self.d_safe * 2.0)
                    total_penalty += depth / self.d_safe
            n_caps = max(len(capsule_dists), 1)
            r_obs = -self.w_obs * total_penalty / n_caps
        else:
            # Fallback: global-min r_obs
            if d_obs >= self.d_safe:
                r_obs = 0.0
            else:
                obs_depth = min(self.d_safe - d_obs, self.d_safe * 2.0)
                r_obs = -self.w_obs * obs_depth / self.d_safe

        # Manipulability reward: encourage non-singular configurations
        r_manip = self.w_manip * np.log(max(w, 1e-4))
        r_manip = max(r_manip, -0.5)

        # Energy penalty: penalize large joint velocities
        r_energy = -self.w_energy * np.sum(dq ** 2)

        # Collision penalty: MuJoCo-based collision detection.
        r_collision = 0.0
        collision_info = {}
        if self.collision_detector is not None:
            collision_penalty, collision_info = self.collision_detector.compute_collision_penalty(
                d_ref=0.05
            )
            r_collision = -self.w_collision * collision_penalty

        # Action smoothness penalty
        r_action = 0.0
        if prev_dq is not None and self.w_action > 0.0:
            r_action = -self.w_action * np.sum((dq - prev_dq) ** 2)

        total = r_track + r_obs + r_manip + r_energy + r_collision + r_action

        info = {
            "r_track":     r_track,
            "r_obs":       r_obs,
            "r_manip":     r_manip,
            "r_energy":    r_energy,
            "r_collision": r_collision,
            "r_action":    r_action,
            **collision_info
        }
        return float(total), info
