"""Repository-wide defaults for the E-Walker-inspired research model.

The geometry/topology follows the public E-Walker paper at prototype scale.
Mass and inertia values in the repository URDF are modelling assumptions, not
identified flight-hardware parameters.
"""

from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URDF = str(REPOSITORY_ROOT / "ewalker_description/urdf/ewalker.urdf")
DEFAULT_XML = str(REPOSITORY_ROOT / "models/ewalker_scene.xml")

DEFAULT_DQ_MAX = np.full(7, 0.5)
DEFAULT_TAU_MAX = np.full(7, 42.0)


def model_limits(model, attribute: str, fallback: np.ndarray) -> np.ndarray:
    """Read a finite positive actuator limit from Pinocchio when available."""
    values = getattr(model, attribute, None) if model is not None else None
    if values is None:
        return np.asarray(fallback, dtype=float).copy()
    values = np.asarray(values, dtype=float).reshape(-1)
    fallback = np.asarray(fallback, dtype=float)
    if values.shape != fallback.shape:
        return fallback.copy()
    valid = np.isfinite(values) & (values > 0)
    return np.where(valid, values, fallback)
