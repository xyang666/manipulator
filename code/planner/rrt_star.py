"""
rrt_star.py
-----------
RRT* (Rapidly-exploring Random Tree Star) motion planner for the Franka Panda
7-DOF manipulator.  Plans collision-free paths in joint space.

Usage:
    from planner.rrt_star import RRTStar
    planner = RRTStar(kin=kin, q_min=q_min, q_max=q_max, obstacles=obs)
    path, time_s, n_nodes = planner.plan(start_q, goal_q)
"""

import numpy as np
import time
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Geometric helpers (standalone — no dependency on TrajectoryGenerator)
# ---------------------------------------------------------------------------

def capsule_sphere_distance(p1: np.ndarray, p2: np.ndarray,
                            cap_radius: float,
                            center: np.ndarray,
                            sphere_radius: float) -> float:
    """
    Signed distance between a capsule (p1-p2 axis, cap_radius) and a sphere.
    Positive = separated, negative = penetration.
    """
    segment = p2 - p1
    seg_len = np.linalg.norm(segment)
    if seg_len < 1e-10:
        closest = p1
    else:
        direction = segment / seg_len
        t = np.dot(center - p1, direction)
        t = np.clip(t, 0.0, seg_len)
        closest = p1 + t * direction
    return float(np.linalg.norm(center - closest) - cap_radius - sphere_radius)


def segment_segment_distance(a1: np.ndarray, a2: np.ndarray,
                             b1: np.ndarray, b2: np.ndarray) -> float:
    """Minimum distance between two line segments."""
    d1, d2 = a2 - a1, b2 - b1
    r = a1 - b1
    a, e = float(np.dot(d1, d1)), float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    eps = 1e-10
    if a < eps and e < eps:
        return float(np.linalg.norm(r))
    if a < eps:
        return float(np.linalg.norm(a1 - (b1 + np.clip(-f / e, 0.0, 1.0) * d2)))
    if e < eps:
        t = np.clip(float(np.dot(-d1, r)) / a, 0.0, 1.0)
        return float(np.linalg.norm((a1 + t * d1) - b1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d1, r))
    denom = a * e - b * b
    if abs(denom) < eps:
        t = np.clip(-c / a, 0.0, 1.0)
        s = 0.0
    else:
        t = np.clip((b * f - c * e) / denom, 0.0, 1.0)
        s = np.clip((b * t + f) / e, 0.0, 1.0)
    return float(np.linalg.norm((a1 + t * d1) - (b1 + s * d2)))


# ---------------------------------------------------------------------------
# RRT* node
# ---------------------------------------------------------------------------

class RRTNode:
    """A node in the RRT* tree."""
    __slots__ = ("q", "parent", "cost")

    def __init__(self, q: np.ndarray, parent: int = -1, cost: float = 0.0):
        self.q = q.copy()
        self.parent = parent
        self.cost = cost


# ---------------------------------------------------------------------------
# RRT* planner
# ---------------------------------------------------------------------------

class RRTStar:
    """
    Joint-space RRT* planner for redundant manipulators.

    Parameters
    ----------
    kin            : ManipulatorKinematics instance (for FK + capsule queries)
    q_min          : (n,) lower joint limits
    q_max          : (n,) upper joint limits
    obstacles      : list of [x, y, z, r] spheres
    goal_bias      : probability of sampling the goal directly
    max_iterations : max RRT* iterations
    step_size      : max extension distance in *normalized* joint space
    goal_tolerance : threshold in normalized distance to consider goal reached
    rewire_factor  : multiplier for the RRT* rewiring radius gamma
    clearance      : minimum capsule-surface-to-sphere-surface distance
    n_interpolation_steps : collision sub-steps per unit normalized distance
    """

    def __init__(self,
                 kin,
                 q_min: np.ndarray,
                 q_max: np.ndarray,
                 obstacles: list,
                 goal_bias: float = 0.1,
                 max_iterations: int = 3000,
                 step_size: float = 0.15,
                 goal_tolerance: float = 0.10,
                 rewire_factor: float = 2.0,
                 clearance: float = 0.02,
                 n_interpolation_steps: int = 20):
        self.kin = kin
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        self.q_range = self.q_max - self.q_min
        self.n_dims = len(self.q_min)
        self.obstacles = obstacles
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.rewire_factor = rewire_factor
        self.clearance = clearance
        self.n_interpolation_steps = n_interpolation_steps

        # Gamma for RRT* rewiring: gamma = rewire_factor * sqrt(n_dims)
        self.gamma = rewire_factor * np.sqrt(self.n_dims)

    # ----- distance metric (normalized Euclidean) --------------------------

    def _distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Weighted Euclidean distance with per-joint normalization."""
        diff = (q1 - q2) / self.q_range
        return float(np.sqrt(np.dot(diff, diff)))

    # ----- sampling --------------------------------------------------------

    def _sample_random(self) -> np.ndarray:
        """Uniform random configuration within joint limits."""
        return np.random.uniform(self.q_min, self.q_max)

    # ----- tree operations -------------------------------------------------

    @staticmethod
    def _nearest(tree: List[RRTNode], q: np.ndarray,
                 dist_fn) -> int:
        """Return index of the node in tree closest to q."""
        best_idx = 0
        best_dist = dist_fn(tree[0].q, q)
        for i, node in enumerate(tree[1:], start=1):
            d = dist_fn(node.q, q)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _near(self, tree: List[RRTNode], q: np.ndarray,
              radius: float) -> List[int]:
        """Return indices of nodes within radius of q."""
        indices = []
        for i, node in enumerate(tree):
            if self._distance(node.q, q) <= radius:
                indices.append(i)
        return indices

    # ----- steering --------------------------------------------------------

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """Extend from q_from toward q_to by at most step_size."""
        d = self._distance(q_from, q_to)
        if d <= self.step_size:
            return q_to.copy()
        # Direction in normalized space
        direction = (q_to - q_from) / self.q_range
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-10:
            return q_from.copy()
        return q_from + (direction / direction_norm) * self.step_size * self.q_range

    # ----- rewiring radius -------------------------------------------------

    def _rewire_radius(self, n_nodes: int) -> float:
        """Adaptive radius from standard RRT* formula."""
        if n_nodes <= 1:
            return self.step_size * 3
        return float(self.gamma * (np.log(n_nodes) / n_nodes) ** (1.0 / self.n_dims))

    # ----- collision checking ----------------------------------------------

    def _segment_collision(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        """
        Check whether the straight-line segment from q1 to q2 collides
        with any obstacle (capsule-sphere) or has self-collision.
        """
        d = self._distance(q1, q2)
        n_steps = max(2, int(np.ceil(d * self.n_interpolation_steps)))

        for alpha in np.linspace(0, 1, n_steps):
            q = (1 - alpha) * q1 + alpha * q2

            # Clamp to joint limits
            if np.any(q < self.q_min) or np.any(q > self.q_max):
                return True

            # Get capsules
            capsules = self.kin.get_link_capsules(q)

            # Capsule-sphere obstacle check
            for p1, p2, cap_r in capsules:
                for obs in self.obstacles:
                    center = np.array(obs[:3], dtype=float)
                    sphere_r = float(obs[3])
                    d = capsule_sphere_distance(p1, p2, cap_r, center, sphere_r)
                    if d < self.clearance:
                        return True

            # Self-collision check (use -0.02 clearance: capsule radii are padded
            # collision geometry, so mild overlap is normal)
            n_cap = len(capsules)
            for i in range(n_cap):
                for j in range(i + 3, n_cap):  # skip adjacent links
                    if j >= n_cap - 3:  # skip finger capsules
                        continue
                    cp1, cp2, cr1 = capsules[i]
                    cq1, cq2, cr2 = capsules[j]
                    sd = segment_segment_distance(cp1, cp2, cq1, cq2)
                    if sd < cr1 + cr2 - 0.02:  # tolerance for padded geometry
                        return True

        return False

    # ----- path extraction -------------------------------------------------

    def _extract_path(self, node: RRTNode, tree: List[RRTNode]) -> List[np.ndarray]:
        """Walk parent pointers and return [q_start, ..., q_node]."""
        path = [node.q]
        idx = node.parent
        while idx >= 0:
            path.append(tree[idx].q)
            idx = tree[idx].parent
        path.reverse()
        return path

    def _shortcut(self, path: List[np.ndarray],
                  max_iterations: int = 100) -> List[np.ndarray]:
        """
        Path shortcutting: randomly pick two non-adjacent waypoints and
        attempt a direct collision-free connection.
        """
        if len(path) <= 2:
            return path

        improved_path = list(path)
        for _ in range(max_iterations):
            if len(improved_path) <= 2:
                break
            i, j = sorted(np.random.choice(len(improved_path), 2, replace=False))
            if j - i <= 1:
                continue
            q_i = improved_path[i]
            q_j = improved_path[j]
            if not self._segment_collision(q_i, q_j):
                # Direct connection is safe — remove intermediate nodes
                improved_path = improved_path[:i + 1] + improved_path[j:]
        return improved_path

    # ----- main planning loop ----------------------------------------------

    def plan(self, start_q: np.ndarray, goal_q: np.ndarray
             ) -> Tuple[List[np.ndarray], float, int]:
        """
        Run RRT* and return (path, planning_time_s, n_nodes_explored).

        ``path`` is a list of joint-space waypoints from start_q to goal_q,
        or an empty list if no path is found.
        """
        start_time = time.time()

        # Fast path: if direct connection is collision-free, skip RRT*
        if not self._segment_collision(start_q, goal_q):
            path = [start_q.copy(), goal_q.copy()]
            elapsed = time.time() - start_time
            return path, elapsed, 2

        tree: List[RRTNode] = [RRTNode(start_q, parent=-1, cost=0.0)]
        best_path: List[np.ndarray] = []
        best_cost = float("inf")

        for iteration in range(self.max_iterations):
            # --- sample ---
            if np.random.random() < self.goal_bias:
                q_rand = goal_q.copy()
            else:
                q_rand = self._sample_random()

            # --- nearest ---
            nearest_idx = self._nearest(tree, q_rand, self._distance)
            nearest_node = tree[nearest_idx]

            # --- steer ---
            q_new = self._steer(nearest_node.q, q_rand)

            # --- validate ---
            if self._segment_collision(nearest_node.q, q_new):
                continue

            # --- find near nodes for rewiring ---
            radius = self._rewire_radius(len(tree))
            near_indices = self._near(tree, q_new, radius)

            # --- choose best parent ---
            best_parent_idx = nearest_idx
            best_new_cost = nearest_node.cost + self._distance(nearest_node.q, q_new)

            for idx in near_indices:
                candidate = tree[idx]
                cost = candidate.cost + self._distance(candidate.q, q_new)
                if cost < best_new_cost and not self._segment_collision(candidate.q, q_new):
                    best_new_cost = cost
                    best_parent_idx = idx

            # --- add node ---
            new_node = RRTNode(q_new, parent=best_parent_idx, cost=best_new_cost)
            tree.append(new_node)
            new_idx = len(tree) - 1

            # --- rewire ---
            for idx in near_indices:
                if idx == best_parent_idx:
                    continue
                n = tree[idx]
                cost_via_new = best_new_cost + self._distance(q_new, n.q)
                if cost_via_new < n.cost and not self._segment_collision(q_new, n.q):
                    n.parent = new_idx
                    n.cost = cost_via_new

            # --- check goal region ---
            d_to_goal = self._distance(q_new, goal_q)
            if d_to_goal < self.goal_tolerance:
                path = self._extract_path(tree[-1], tree)
                path.append(goal_q)
                # Quick validation of full path
                full_valid = True
                for i in range(len(path) - 1):
                    if self._segment_collision(path[i], path[i + 1]):
                        full_valid = False
                        break
                if full_valid:
                    path = self._shortcut(path)
                    elapsed = time.time() - start_time
                    return path, elapsed, len(tree)

            # --- track best partial path (fallback) ---
            if len(best_path) == 0 or d_to_goal < best_cost:
                # Found a node closer to goal
                best_cost = d_to_goal
                partial_path = self._extract_path(tree[-1], tree)
                # Check if the best node can connect to goal
                if not self._segment_collision(partial_path[-1], goal_q):
                    partial_path.append(goal_q)
                    for i in range(len(partial_path) - 1):
                        if self._segment_collision(partial_path[i], partial_path[i + 1]):
                            break
                    else:
                        # All segments validated
                        best_path = self._shortcut(partial_path)

        elapsed = time.time() - start_time

        # Fallback: return best partial path if we have one
        if best_path:
            return best_path, elapsed, len(tree)

        return [], elapsed, len(tree)
