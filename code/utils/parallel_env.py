"""
parallel_env.py
---------------
Run multiple ManipulatorEnv instances in parallel worker processes.
Each worker runs its own MuJoCo simulation independently on CPU,
enabling GPU to receive data faster without waiting for a single env.

Usage:
    pool = ParallelEnvPool(4, lambda: ManipulatorEnv(...))
    obss = pool.reset_all()                     # [n, obs_dim]
    result = pool.step_all(actions)             # actions: [n, act_dim]
    obss = result["obs"]                        # auto-reset if done
"""

import multiprocessing as mp
from multiprocessing.connection import Connection
import numpy as np
from typing import Callable


class ParallelEnvPool:
    """N environment workers in separate processes, stepped in parallel."""

    def __init__(self, n_envs: int, env_creator: Callable):
        self.n_envs = n_envs
        self._pipes: list[Connection] = []
        self._workers: list[mp.Process] = []

        ctx = mp.get_context("fork")
        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_worker_loop, args=(child_conn, env_creator, i))
            p.start()
            child_conn.close()
            self._pipes.append(parent_conn)
            self._workers.append(p)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_all(self) -> np.ndarray:
        """Reset every environment. Returns obs [n_envs, obs_dim]."""
        for pipe in self._pipes:
            pipe.send("reset")
        return np.stack([pipe.recv() for pipe in self._pipes]).astype(np.float32)

    def step_all(self, actions: np.ndarray) -> dict:
        """
        Step every environment with the provided actions.
        Automatically resets any env that finished (done -> True).

        Parameters
        ----------
        actions : ndarray of shape (n_envs, act_dim)

        Returns
        -------
        dict with keys:
            obs, reward, done, info,
            q_before, dq_before, dq_after,
            J (3×n Jacobian), sigma, dx_nom
        """
        _send = self._pipes
        for pipe, a in zip(_send, actions):
            pipe.send(("step", np.ascontiguousarray(a)))
        results = [pipe.recv() for pipe in _send]
        return self._combine(results, self.n_envs)

    def close(self):
        """Shut down all worker processes."""
        for pipe in self._pipes:
            try:
                pipe.send("close")
            except Exception:
                pass
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.kill()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _combine(results: list, n: int) -> dict:
        out = {
            "reward":    np.empty(n, dtype=np.float32),
            "done":      np.empty(n, dtype=bool),
            "info":      [None] * n,
            "sigma":     np.empty((n, 1), dtype=np.float32),
            "scene_id":  np.empty(n, dtype=np.int32),
        }
        for k in ("obs", "q_before", "dq_before", "dq_after", "J", "dx_nom"):
            arr = np.concatenate([r[k][np.newaxis, ...] for r in results], axis=0)
            out[k] = arr
        for i, r in enumerate(results):
            out["reward"][i] = r["reward"]
            out["done"][i] = r["done"]
            out["info"][i] = r["info"]
            out["sigma"][i] = r["sigma"]
            out["scene_id"][i] = r["scene_id"]
        return out


def _worker_loop(pipe: Connection, env_creator: Callable, worker_id: int = 0):
    """Worker process: owns one env, responds to parent commands."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Seed RNG uniquely per worker to avoid synchronized scene sampling
    # when using fork (all workers inherit the same parent RNG state).
    import numpy as np
    np.random.seed((worker_id * 9973 + 42) & 0xFFFFFFFF)

    env = env_creator()

    while True:
        try:
            msg = pipe.recv()
        except (EOFError, BrokenPipeError):
            break

        if msg == "reset":
            pipe.send(env.reset())

        elif msg == "close":
            break

        elif isinstance(msg, tuple) and msg[0] == "step":
            action = msg[1]
            q_before = env.q.copy()
            dq_before = env.dq.copy()

            obs, reward, done, info = env.step(action)
            dq_after = env.dq.copy()
            scene_id = getattr(env, '_current_scene_id', -1)

            # Auto-reset: if done, start a new episode so the next iteration
            # can immediately continue without a separate reset call.
            if done:
                next_obs = env.reset()
            else:
                next_obs = obs

            # Grab physics metadata saved by step()
            J = getattr(env, "_last_J",
                        np.zeros((3, env.n), dtype=np.float32))
            sigma = getattr(env, "_last_sigma", 0.0)
            dx_nom = getattr(env, "_last_dx_nom",
                             np.zeros(3, dtype=np.float32))

            pipe.send({
                "obs":     next_obs.astype(np.float32),
                "reward":  np.float32(reward),
                "done":    done,
                "info":    info,
                "scene_id": int(scene_id),
                "q_before": q_before.astype(np.float32),
                "dq_before": dq_before.astype(np.float32),
                "dq_after": dq_after.astype(np.float32),
                "J":       J.astype(np.float32),
                "sigma":   np.float32(sigma),
                "dx_nom":  dx_nom.astype(np.float32),
            })
