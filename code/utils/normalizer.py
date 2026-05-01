"""Running mean/std normalizer for observations and rewards."""

import numpy as np


class RunningMeanStd:
    """Tracks running mean and variance of a data stream.

    Usage::
        normalizer = RunningMeanStd(shape=(25,))
        for obs in stream:
            normalized = normalizer(obs)   # updates stats + returns normalized
    """

    def __init__(self, shape=(), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Update statistics with new sample and return normalized value."""
        self.update(x)
        return self.normalize(x)

    def update(self, x: np.ndarray):
        """Update running mean/var with a new sample or batch."""
        x = np.asarray(x, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.5, neginf=-0.5)  # sanitize
        batch_mean = x.mean(axis=0) if x.ndim > 1 else x
        batch_var = x.var(axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / max(tot_count, 1)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize using current statistics (no update)."""
        return ((np.asarray(x, dtype=np.float32) - self.mean) /
                np.sqrt(self.var + self.epsilon)).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        return (np.asarray(x, dtype=np.float32) *
                np.sqrt(self.var + self.epsilon) + self.mean).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]
