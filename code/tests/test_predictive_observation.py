import numpy as np

from env.manipulator_env import (learned_reference_rate,
                                 predictive_obstacle_features)


def test_predictive_obstacle_features_encode_approach():
    features = predictive_obstacle_features(
        np.array([1.0, 0.0, 0.0]), np.array([-0.5, 0.0, 0.0]),
        radius=0.1, horizons=[10, 50], dt=0.02,
    )
    np.testing.assert_allclose(features[:3], [0.9, 0.0, 0.0])
    np.testing.assert_allclose(features[3:6], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(features[-3:], [1.0, 0.4, 0.5])


def test_predictive_obstacle_features_static_obstacle():
    features = predictive_obstacle_features(
        np.array([0.3, 0.0, 0.0]), np.zeros(3),
        radius=0.1, horizons=[10], dt=0.02,
    )
    np.testing.assert_allclose(features, [0.3, 0.0, 0.0, 0.0, 0.2, 0.0])


def test_learned_reference_rate_is_risk_gated_and_bounded():
    assert learned_reference_rate(-0.2, 0.2, 1.0) == 0.0
    assert learned_reference_rate(0.2, 0.2, 1.0) == 2.0
    assert learned_reference_rate(-0.2, 0.2, 0.0) == 1.0
