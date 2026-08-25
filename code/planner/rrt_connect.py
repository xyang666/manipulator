"""Fast bidirectional RRT-Connect feasibility oracle for scene generation.

This planner is not a tested controller and its path must never be exposed to
the policy.  It is used only to certify that a challenge scene admits at least
one collision-free joint-space detour when the direct connection is blocked.
"""

from __future__ import annotations

import time

import numpy as np

from planner.rrt_star import RRTNode, RRTStar


class RRTConnect(RRTStar):
    """Bidirectional feasibility planner sharing the project's collision model."""

    def __init__(self, *args, seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed)

    def _sample_random(self) -> np.ndarray:
        return self.rng.uniform(self.q_min, self.q_max)

    def _trace(self, tree: list[RRTNode], index: int) -> list[np.ndarray]:
        path = []
        while index >= 0:
            path.append(tree[index].q.copy())
            index = tree[index].parent
        path.reverse()
        return path

    def _extend_once(self, tree: list[RRTNode], target: np.ndarray) -> int | None:
        nearest = self._nearest(tree, target, self._distance)
        candidate = self._steer(tree[nearest].q, target)
        if np.allclose(candidate, tree[nearest].q) or self._segment_collision(
                tree[nearest].q, candidate):
            return None
        tree.append(RRTNode(candidate, parent=nearest,
                            cost=tree[nearest].cost + self._distance(
                                tree[nearest].q, candidate)))
        return len(tree) - 1

    def _connect(self, tree: list[RRTNode], target: np.ndarray
                 ) -> tuple[int | None, bool]:
        """Greedily extend a tree toward target until blocked or reached."""
        last = None
        while True:
            index = self._extend_once(tree, target)
            if index is None:
                return last, False
            last = index
            if self._distance(tree[index].q, target) <= 1e-8:
                return index, True

    def plan(self, start_q: np.ndarray, goal_q: np.ndarray
             ) -> tuple[list[np.ndarray], float, int]:
        started = time.perf_counter()
        start_q = np.asarray(start_q, dtype=float)
        goal_q = np.asarray(goal_q, dtype=float)
        if not self._segment_collision(start_q, goal_q):
            return [start_q.copy(), goal_q.copy()], time.perf_counter() - started, 2

        tree_a = [RRTNode(start_q)]
        tree_b = [RRTNode(goal_q)]
        a_root_is_start = True
        for _ in range(self.max_iterations):
            target = (goal_q if self.rng.random() < self.goal_bias
                      else self._sample_random())
            new_a = self._extend_once(tree_a, target)
            if new_a is not None:
                new_b, reached = self._connect(tree_b, tree_a[new_a].q)
                if reached and new_b is not None:
                    path_a = self._trace(tree_a, new_a)
                    path_b = self._trace(tree_b, new_b)
                    if a_root_is_start:
                        path = path_a + list(reversed(path_b[:-1]))
                    else:
                        path = path_b + list(reversed(path_a[:-1]))
                    path = self._shortcut(path, max_iterations=40)
                    return (path, time.perf_counter() - started,
                            len(tree_a) + len(tree_b))
            tree_a, tree_b = tree_b, tree_a
            a_root_is_start = not a_root_is_start
        return [], time.perf_counter() - started, len(tree_a) + len(tree_b)
