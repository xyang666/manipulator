import numpy as np

from planner.rrt_connect import RRTConnect


class PointKinematics:
    """A 2-D point represented as a zero-radius capsule."""

    def get_link_capsules(self, q):
        point = np.array([q[0], q[1], 0.0])
        return [(point, point, 0.0)]


def test_rrt_connect_finds_detour_around_blocked_direct_edge():
    planner = RRTConnect(
        PointKinematics(), np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
        obstacles=[[0.0, 0.0, 0.0, 0.2]], seed=7,
        max_iterations=500, step_size=0.12, clearance=0.02,
        n_interpolation_steps=80,
    )
    start = np.array([-0.8, 0.0])
    goal = np.array([0.8, 0.0])
    assert planner._segment_collision(start, goal)
    path, _, _ = planner.plan(start, goal)
    assert len(path) >= 3
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)
    assert all(not planner._segment_collision(path[i], path[i + 1])
               for i in range(len(path) - 1))
