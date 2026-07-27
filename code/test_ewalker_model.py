"""Cross-checks for the E-Walker-inspired URDF, MJCF and capsule model."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

from env.kinematics import ManipulatorKinematics
from robot_config import DEFAULT_URDF, DEFAULT_XML


ROBOT_GEOMS = [
    "ewalker_base_capsule",
    "ewalker_link1_capsule",
    "ewalker_link2_capsule",
    "ewalker_link3_capsule",
    "ewalker_link4_capsule",
    "ewalker_link5_capsule",
    "ewalker_link6_capsule",
    "ewalker_link7_capsule",
    "ewalker_lee_capsule",
]


def _mujoco_capsules(model, data):
    capsules = []
    for name in ROBOT_GEOMS:
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name)
        center = data.geom_xpos[geom_id].copy()
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        axis = rotation[:, 2]
        radius, half_length = model.geom_size[geom_id, :2]
        capsules.append((
            center - half_length * axis,
            center + half_length * axis,
            float(radius),
        ))
    return capsules


def test_model_metadata_states_reconstruction_boundary():
    metadata_path = (
        Path(DEFAULT_URDF).parents[1] / "model_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "not flight hardware" in metadata["status"]
    assert metadata["source"]["doi"] == "10.3389/frobt.2022.995813"
    assert metadata["modelled_parameters"]["model_total_mass_kg"] == 10.6


def test_urdf_and_mjcf_have_matching_seven_dof_limits_and_tcp():
    kin = ManipulatorKinematics(DEFAULT_URDF, n_joints=7)
    mj_model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    mj_data = mujoco.MjData(mj_model)
    q = np.zeros(7)
    mj_data.qpos[:] = q
    mujoco.mj_forward(mj_model, mj_data)

    assert kin.model.nq == kin.model.nv == 7
    assert mj_model.nq == mj_model.nv == mj_model.nu == 7
    np.testing.assert_allclose(
        kin.q_min, mj_model.jnt_range[:, 0], atol=1e-10)
    np.testing.assert_allclose(
        kin.q_max, mj_model.jnt_range[:, 1], atol=1e-10)

    x_urdf, _ = kin.forward_kinematics(q)
    site_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    np.testing.assert_allclose(x_urdf, mj_data.site_xpos[site_id], atol=1e-9)
    np.testing.assert_allclose(x_urdf, [0.0, 0.0, 1.3], atol=1e-9)


def test_analytical_and_mujoco_capsules_match_at_random_configs():
    kin = ManipulatorKinematics(DEFAULT_URDF, n_joints=7)
    mj_model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    mj_data = mujoco.MjData(mj_model)
    rng = np.random.default_rng(20260727)

    for _ in range(20):
        q = rng.uniform(0.8 * kin.q_min, 0.8 * kin.q_max)
        mj_data.qpos[:] = q
        mujoco.mj_forward(mj_model, mj_data)
        analytical = kin.get_link_capsules(q)
        simulated = _mujoco_capsules(mj_model, mj_data)
        assert len(analytical) == len(simulated) == 9
        for (a1, a2, ar), (m1, m2, mr) in zip(analytical, simulated):
            direct = np.linalg.norm(a1 - m1) + np.linalg.norm(a2 - m2)
            reversed_order = (
                np.linalg.norm(a1 - m2) + np.linalg.norm(a2 - m1))
            assert min(direct, reversed_order) < 1e-8
            assert abs(ar - mr) < 1e-12


def test_urdf_modelled_mass_is_recorded_value():
    model = pin.buildModelFromUrdf(DEFAULT_URDF)
    total_mass = sum(float(inertia.mass) for inertia in model.inertias)
    assert abs(total_mass - 10.6) < 1e-10
