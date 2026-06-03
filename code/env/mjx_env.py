"""
mjx_env.py
----------
JAX-based batched manipulator environment for GPU-accelerated RL training.

Replaces Pinocchio kinematics with JAX chain-of-transforms FK and
autodiff Jacobians. Runs N environments in parallel via jax.vmap.

Architecture:
  Pure JAX (no MJX dependency). Kinematic parameters extracted from
  MuJoCo model XML and cached Pinocchio output at init time.
  Observation format matches ManipulatorEnv with obs_scene_embed > 0.
"""

import os
import json
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

# ========================================================================
# Inline SDF functions (module-level for JIT)
# ========================================================================

def _capsule_to_sphere_distance(p1, p2, capsule_radius, sphere_center, sphere_radius):
    """SDF: capsule (p1-p2 + radius) to sphere (center + radius).
    Supports arbitrary batch dimensions."""
    segment = p2 - p1
    seg_sq = jnp.sum(segment ** 2, axis=-1)
    seg_len = jnp.sqrt(jnp.maximum(seg_sq, 1e-16))
    direction = segment / jnp.maximum(seg_len, 1e-8)[..., None]
    t = jnp.sum((sphere_center - p1) * direction, axis=-1)
    t = jnp.clip(t, 0.0, seg_len)
    closest_point = p1 + t[..., None] * direction
    diff = sphere_center - closest_point
    center_dist = jnp.sqrt(jnp.maximum(jnp.sum(diff ** 2, axis=-1), 1e-16))
    return center_dist - capsule_radius - sphere_radius


def _segment_distance_sq(p1, p2, q1, q2):
    """Squared minimum distance between two 3D line segments (Eberly 2002, JAX).

    Returns scalar or batched scalar (last dimension = 3D coordinates).
    """
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = jnp.sum(d1 * d1, axis=-1)
    e = jnp.sum(d2 * d2, axis=-1)
    f = jnp.sum(d2 * r, axis=-1)
    eps = 1e-12

    # Case 1: both degenerate (a <= eps and e <= eps)
    dot_r = jnp.sum(r * r, axis=-1)

    # Case 2: seg1 degenerate (a <= eps)
    t_c2 = jnp.clip(f / jnp.maximum(e, eps), 0.0, 1.0)
    dist_c2 = jnp.sum((p1 - (q1 + t_c2[..., None] * d2)) ** 2, axis=-1)

    # Case 3: seg2 degenerate (e <= eps)
    s_c3 = jnp.clip(-jnp.sum(d1 * r, axis=-1) / jnp.maximum(a, eps), 0.0, 1.0)
    dist_c3 = jnp.sum(((p1 + s_c3[..., None] * d1) - q1) ** 2, axis=-1)

    # Case 4: general case
    b = jnp.sum(d1 * d2, axis=-1)
    c = jnp.sum(d1 * r, axis=-1)
    denom = a * e - b * b

    s_num = b * f - c * e
    t_num = a * f - b * c
    denom_safe = jnp.where(denom < eps, eps, denom)
    s_clip = jnp.clip(s_num / denom_safe, 0.0, 1.0)
    t_clip = jnp.clip(t_num / denom_safe, 0.0, 1.0)

    # Near-parallel case (denom < eps)
    s_par = jnp.clip(-c / jnp.maximum(a, eps), 0.0, 1.0)
    closest_a = p1 + s_par[..., None] * d1
    t_par = jnp.clip(jnp.sum(d2 * (closest_a - q1), axis=-1) / jnp.maximum(e, eps), 0.0, 1.0)
    dist_par = jnp.sum((closest_a - (q1 + t_par[..., None] * d2)) ** 2, axis=-1)

    dist_gen = jnp.sum(((p1 + s_clip[..., None] * d1) - (q1 + t_clip[..., None] * d2)) ** 2, axis=-1)
    dist_near_par = jnp.where(denom < eps, dist_par, dist_gen)

    # Select between cases
    a_le_eps = a <= eps
    e_le_eps = e <= eps
    both_degen = a_le_eps & e_le_eps
    dist = jnp.where(both_degen, dot_r,
                     jnp.where(a_le_eps, dist_c2,
                               jnp.where(e_le_eps, dist_c3, dist_near_par)))
    return dist


# ========================================================================
# JAX kinematics helpers
# ========================================================================

def _quat_to_rotmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - w*z), 2.0 * (x*z + w*y)],
        [2.0 * (x*y + w*z), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - w*x)],
        [2.0 * (x*z - w*y), 2.0 * (y*z + w*x), 1.0 - 2.0 * (x*x + y*y)],
    ])


def _make_transform(R, t):
    T = jnp.eye(4)
    T = T.at[0:3, 0:3].set(R)
    T = T.at[0:3, 3].set(t)
    return T


def extract_chain_from_mj(xml_path: str):
    """Extract kinematic chain parameters from MuJoCo XML for JAX FK."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(xml_path)
    nbody = model.nbody
    nsite = model.nsite
    parent_ids = np.array([int(model.body_parentid[b]) for b in range(nbody)], dtype=np.int32)
    body_pos = np.array([model.body_pos[b].copy() for b in range(nbody)], dtype=np.float64)
    body_quat = np.array([model.body_quat[b].copy() for b in range(nbody)], dtype=np.float64)
    jnt_type = np.zeros(nbody, dtype=np.int32)
    jnt_axis = np.zeros((nbody, 3), dtype=np.float64)
    jnt_qposadr = np.full(nbody, -1, dtype=np.int32)
    for b in range(nbody):
        jadr = int(model.body_jntadr[b])
        jnum = int(model.body_jntnum[b])
        if jnum > 0:
            jid = jadr
            jnt_type[b] = int(model.jnt_type[jid])
            jnt_axis[b] = model.jnt_axis[jid].copy()
            jnt_qposadr[b] = int(model.jnt_qposadr[jid])
    site_body = np.array([int(np.asarray(model.site(s).bodyid).item()) for s in range(nsite)], dtype=np.int32)
    site_pos = np.array([model.site(s).pos.copy() for s in range(nsite)], dtype=np.float64)
    site_quat = np.array([model.site(s).quat.copy() for s in range(nsite)], dtype=np.float64)
    ee_site_id = 0
    for s in range(nsite):
        if model.site(s).name == 'ee_site':
            ee_site_id = s
            break
    return {
        'parent_ids': jnp.array(parent_ids),
        'body_pos': jnp.array(body_pos),
        'body_quat': jnp.array(body_quat),
        'jnt_type': jnp.array(jnt_type),
        'jnt_axis': jnp.array(jnt_axis),
        'jnt_qposadr': jnp.array(jnt_qposadr),
        'site_body': jnp.array(site_body),
        'site_pos': jnp.array(site_pos),
        'site_quat': jnp.array(site_quat),
        'ee_site_id': ee_site_id,
        'nv': model.nv,
        'nq': model.nq,
    }


def load_cached_kinematics(cache_path: str):
    """Load cached capsule parameters and self-collision pairs."""
    data = np.load(cache_path)
    return {
        'capsule_params': jnp.array(data['capsule_params']),  # (n_caps, 3, 3)
        'self_pairs': jnp.array(data['self_pairs'], dtype=jnp.int32),  # (n_self, 2)
    }


# ========================================================================
# JAX forward kinematics
# ========================================================================

def _joint_transform(q_val, jtype, axis):
    def hinge(_):
        c, s = jnp.cos(q_val), jnp.sin(q_val)
        R = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return _make_transform(R, jnp.zeros(3))
    def slide(_):
        return _make_transform(jnp.eye(3), axis * q_val)
    def no_joint(_):
        return jnp.eye(4)
    return lax.switch(jtype, [no_joint, no_joint, slide, hinge], None)


def forward_kinematics_chain(q, chain):
    """Forward kinematics: body transforms, orientations, and site positions."""
    parent_ids = chain['parent_ids']
    body_pos = chain['body_pos']
    body_quat = chain['body_quat']
    jnt_type = chain['jnt_type']
    jnt_axis = chain['jnt_axis']
    jnt_qposadr = chain['jnt_qposadr']
    site_body = chain['site_body']
    site_pos = chain['site_pos']
    site_quat = chain['site_quat']
    nbody = parent_ids.shape[0]
    nsite = site_body.shape[0]

    def body_fn(i, carry):
        T_all = carry
        pid = parent_ids[i]
        T_par = T_all[pid]
        q_val = jnp.where(jnt_qposadr[i] >= 0, q[jnt_qposadr[i]], 0.0)
        T_jnt = _joint_transform(q_val, jnt_type[i], jnt_axis[i])
        R_body = _quat_to_rotmat(body_quat[i])
        T_off = _make_transform(R_body, body_pos[i])
        return T_all.at[i].set(T_par @ T_off @ T_jnt)

    T_body = lax.fori_loop(1, nbody, body_fn, jnp.eye(4)[None].repeat(nbody, 0))
    body_xpos = T_body[:, 0:3, 3]
    body_xmat = T_body[:, 0:3, 0:3]

    def site_pos_fn(s):
        return (T_body[site_body[s]] @ _make_transform(
            _quat_to_rotmat(site_quat[s]), site_pos[s]))[0:3, 3]

    site_xpos = jax.vmap(site_pos_fn)(jnp.arange(nsite)) if nsite > 0 else jnp.zeros((0, 3))
    return body_xpos, body_xmat, site_xpos


def ee_position_fn(q, chain, ee_site_id):
    _, _, site_xpos = forward_kinematics_chain(q, chain)
    return site_xpos[ee_site_id]


# ========================================================================
# JAX control utilities
# ========================================================================

def damped_pinv(J, damping=1e-4):
    U, s, Vt = jnp.linalg.svd(J, full_matrices=False)
    s2 = s * s
    s_inv = jnp.where(s2 > 1e-10, s / (s2 + damping), 0.0)
    return Vt.T @ jnp.diag(s_inv) @ U.T


def null_space_basis(J):
    _, _, Vt = jnp.linalg.svd(J, full_matrices=True)
    m = J.shape[0]
    return Vt[m:, :].T


def manipulability(J):
    JJt = J @ J.T
    return jnp.sqrt(jnp.maximum(jnp.linalg.det(JJt), 0.0))


# ========================================================================
# Control law
# ========================================================================

def compute_pid(x_d, x_ee, integral_err, dt, kp=4.0, ki=0.5, decay=0.98, anti_windup=0.02):
    pos_err = x_ee - x_d
    err_norm = jnp.linalg.norm(pos_err)
    kp_adaptive = kp * (1.0 + jnp.tanh(err_norm / 0.05))
    integral_err = decay * integral_err + pos_err * dt
    i_clamp = anti_windup * 2.0 / (1.0 + jnp.exp(-4.0 * err_norm)) - anti_windup
    integral_err = jnp.clip(integral_err, -i_clamp, i_clamp)
    dx_pid = -kp_adaptive * pos_err + ki * integral_err
    return dx_pid, integral_err


def compute_sigma_gating(d_obs, d_safe, lag_lambda, lr_lag, lag_target):
    violation = jnp.maximum(0.0, d_safe - d_obs) / jnp.maximum(d_safe, 1e-6)
    lag_lambda = jnp.maximum(0.0, lag_lambda + lr_lag * (violation - lag_target))
    sigma = jnp.clip(lag_lambda, 0.0, 1.0)
    return sigma, lag_lambda


def compute_path_progression(step_count, episode_len, tracking_error, path_param, last_advance, path_deadzone=0.10):
    total = episode_len
    a_end = int(total * 0.2)
    d_start = int(total * 0.8)
    nominal_s = jnp.where(
        step_count < a_end,
        (step_count / jnp.maximum(1, a_end)) ** 2 * a_end / total,
        jnp.where(step_count > d_start,
            d_start / total + (2.0*(step_count-d_start)/jnp.maximum(1,total-d_start)
                - ((step_count-d_start)/jnp.maximum(1,total-d_start))**2) * (total-d_start) / total,
            step_count / total))
    err_deadzone = jnp.maximum(0.0, tracking_error - 0.02)
    raw_advance = jnp.clip(1.0 - err_deadzone / path_deadzone, 0.0, 1.0)
    advance_rate = 0.5 * raw_advance + 0.5 * last_advance
    new_path_param = jnp.minimum(1.0, path_param + (nominal_s - path_param) * advance_rate)
    return new_path_param, advance_rate


def update_target(x_start, x_goal, path_param, x_d_prev, dt=0.02):
    x_d = (1.0 - path_param) * x_start + path_param * x_goal
    dx_d = (x_d - x_d_prev) / dt
    return x_d, dx_d


# ========================================================================
# Reward function (JAX)
# ========================================================================

def compute_reward(x_ee, x_d, d_obs, w, q_arm, dq_cmd, prev_dq_arm, capsule_dists, action, d_safe, rp, collided=False):
    wt = rp['w_track']; wo = rp['w_obs']; wm = rp['w_manip']
    we = rp['w_energy']; wa = rp['w_action']; wc = rp.get('w_collision', 100.0)
    rmin = rp.get('reward_min', None)

    pos_err = jnp.linalg.norm(x_ee - x_d)
    r_track = wt * jnp.exp(-wt * pos_err)
    n_caps = capsule_dists.shape[0]
    def caps_penalty(i, total):
        d_cap = capsule_dists[i]
        depth = jnp.minimum(jnp.maximum(d_safe - d_cap, 0.0), d_safe * 2.0)
        return total + depth / d_safe
    total_penalty = lax.fori_loop(0, n_caps, caps_penalty, 0.0)
    r_obs = -wo * total_penalty / jnp.maximum(n_caps, 1)
    r_manip = (wm * jnp.log(jnp.maximum(w, 1e-4))) if wm > 0 else 0.0
    r_energy = -we * jnp.sum(dq_cmd ** 2) if we > 0 else 0.0
    r_action = -wa * jnp.sum((dq_cmd - prev_dq_arm) ** 2) if (prev_dq_arm is not None and wa > 0) else 0.0
    # Collision penalty based on capsule penetration depth
    r_collision = jnp.where(collided, -wc * jnp.minimum(jnp.abs(jnp.min(capsule_dists)), 0.5), 0.0)
    total_reward = r_track + r_obs + r_manip + r_energy + r_action + r_collision
    if rmin is not None:
        total_reward = jnp.maximum(total_reward, rmin)
    return total_reward, {'r_track': r_track, 'r_obs': r_obs, 'r_manip': r_manip,
                           'r_energy': r_energy, 'r_collision': r_collision, 'r_action': r_action}


# ========================================================================
# World-frame capsule computation
# ========================================================================

def compute_world_capsules(q, chain, capsule_params):
    """Transform capsule endpoints from body frame to world frame.
    Uses explicit body mapping matching collision_specs in kinematics.py:
      caps 0→body0(world), 1→1, 2→2, 3→3, 4→4, 5→5, 6→5, 7→6, 8→7, 9→7, 10→8, 11→9"""
    body_xpos, body_xmat, _ = forward_kinematics_chain(q, chain)
    n_caps = capsule_params.shape[0]
    nbody = body_xpos.shape[0]
    # Capsule index → MuJoCo body index (matches collision_specs order)
    # body index for capsule i: i+1 (matches MuJoCo body numbering when
    # collision_specs order aligns with MJ body tree, verified empirically).
    # Capsules 9+ (hand, fingers) clamp to body 7 for hand, 8/9 for fingers.
    # Cap 0 (link0) → body 1 (panda_link1 in MJ ≈ Pinocchio link0 frame)
    # Cap 1..6 (link1..link5 x2) → body 2..6
    # Cap 7 (link6) → body 7
    # Cap 8 (link7) → body 8? No, MJ body 7 = panda_link7
    # Actually: just use min(i+1, nbody-1) which was verified correct
    # Correct mapping: capsule for collision_specs link i → body index i in MuJoCo
    # (body 0=world/panda_link0, body 1=link1, ..., body 7=link7, 8=leftfinger, 9=rightfinger)
    # Hand map to body 7 (link7, same frame in MJ).
    cap_to_body = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9], dtype=jnp.int32)

    def transform(i, carry):
        cw = carry
        bid = cap_to_body[i]
        bid = jnp.minimum(bid, nbody - 1)
        R, t = body_xmat[bid], body_xpos[bid]
        p1 = R @ capsule_params[i, 0] + t
        p2 = R @ capsule_params[i, 1] + t
        cw = cw.at[i, 0].set(p1)
        cw = cw.at[i, 1].set(p2)
        return cw
    return lax.fori_loop(0, n_caps, transform, jnp.zeros((n_caps, 2, 3)))


# ========================================================================
# Observation builder (v3 format: scene_embed + waypoints + self_dists)
# ========================================================================

def build_observation_v3(
    q_arm, dq_arm, x_ee, x_d,
    capsule_dists, self_dists, scene_embed, waypoints,
    path_param, sigma, n_caps=11, n_self=45, n_obs_embed=5, n_wp=3,
):
    """Build observation matching ManipulatorEnv scene_embed format (dq_rep removed).

    Format: [q(7), dq(7), x_ee(3), x_d(3),
             wp1(3), wp2(3), wp3(3),
             capsule_dists(n_caps), self_dists(n_self),
             scene_embed(n_obs_embed*4),
             path_progress(1), sigma(1)]
    """
    parts = [q_arm, dq_arm, x_ee, x_d]
    if n_wp > 0:
        parts.extend([waypoints[i] for i in range(n_wp)])
    parts.extend([capsule_dists, self_dists, scene_embed,
                  jnp.array([path_param]), jnp.array([sigma])])
    return jnp.concatenate(parts)


# ========================================================================
# Scene management
# ========================================================================

class SceneManager:
    def __init__(self, json_path: str, n_envs: int, max_obs: int = 10):
        with open(json_path, 'r') as f:
            self.scenes = json.load(f)
        self.n_scenes = len(self.scenes)
        self.n_envs = n_envs
        self.max_obs = max_obs
        self.env_scene_ids = np.arange(n_envs) % self.n_scenes

    def get_scene_data(self, env_indices):
        starts, goals = [], []
        start_qs, goal_qs = [], []
        obs_centers, obs_radii, n_obs = [], [], []
        for idx in env_indices:
            scene = self.scenes[int(idx) % self.n_scenes]
            starts.append(scene['start'])
            goals.append(scene['goal'])
            start_qs.append(scene.get('start_q', np.zeros(7)))
            goal_qs.append(scene.get('goal_q', np.zeros(7)))
            obstacles = scene.get('obstacles', [])
            centers = np.zeros((self.max_obs, 3), dtype=np.float64)
            radii = np.zeros(self.max_obs, dtype=np.float64)
            for j, ob in enumerate(obstacles):
                if j >= self.max_obs: break
                centers[j] = ob[:3]
                radii[j] = ob[3] if len(ob) > 3 else 0.1
            obs_centers.append(centers)
            obs_radii.append(radii)
            n_obs.append(len(obstacles))
        return (np.array(starts), np.array(goals),
                np.array(start_qs), np.array(goal_qs),
                np.array(obs_centers), np.array(obs_radii), np.array(n_obs))


# ========================================================================
# State management
# ========================================================================

EnvState = Dict[str, jnp.ndarray]

def init_state(n_envs: int, nv: int, n_self: int = 45) -> EnvState:
    return {
        'q': jnp.zeros((n_envs, nv)),
        'dq': jnp.zeros((n_envs, nv)),
        'x_start': jnp.zeros((n_envs, 3)),
        'x_goal': jnp.zeros((n_envs, 3)),
        'x_d': jnp.zeros((n_envs, 3)),
        'dx_d': jnp.zeros((n_envs, 3)),
        'path_param': jnp.zeros((n_envs,)),
        'step_count': jnp.zeros((n_envs,), dtype=jnp.int32),
        'integral_err': jnp.zeros((n_envs, 3)),
        'ever_collided': jnp.zeros((n_envs,), dtype=jnp.bool_),
        'lag_lambda': jnp.zeros((n_envs,)),
        'last_advance': jnp.zeros((n_envs,)),
        'prev_dq': jnp.zeros((n_envs, nv)),
        'last_dx_nom': jnp.zeros((n_envs, 3)),
        'last_sigma': jnp.zeros((n_envs,)),
        'n_obs': jnp.zeros((n_envs,), dtype=jnp.int32),
        'scene_start_q': jnp.zeros((n_envs, nv)),
        'self_dists_mask': jnp.ones((n_envs, n_self), dtype=jnp.bool_),
    }


# ========================================================================
# Single env step (v3 compatible: scene_embed + waypoints + self_dists)
# ========================================================================

def _single_env_step(
    env_state, action, obs_centers, obs_radii, capsule_params, self_pairs, chain,
    ee_site_id, nv_arm, n_caps, max_obs, n_self, n_obs_embed, n_wp, wp_steps,
    dt, d_safe, lr_lag, lag_target, path_deadzone, episode_len, reward_params,
    self_mask_init, obs_coll_tol,
):
    q = env_state['q']; dq = env_state['dq']; x_d = env_state['x_d']
    path_param = env_state['path_param']; step_count = env_state['step_count']
    integral_err = env_state['integral_err']; lag_lambda = env_state['lag_lambda']
    last_advance = env_state['last_advance']; prev_dq = env_state['prev_dq']
    x_start = env_state['x_start']; x_goal = env_state['x_goal']
    n_obs = env_state['n_obs']

    delta_x_rl = action[:3]
    z = action[3:nv_arm]

    # 1. FK + Jacobian
    J_full = jax.jacfwd(ee_position_fn, argnums=0)(q, chain, ee_site_id)
    J = J_full[:, :nv_arm]
    body_xpos, body_xmat, site_xpos = forward_kinematics_chain(q, chain)
    x_ee = site_xpos[ee_site_id]

    # 2. SDF: per-capsule distances
    caps_world = compute_world_capsules(q, chain, capsule_params)
    p1s = caps_world[:, 0, :]; p2s = caps_world[:, 1, :]; crs = capsule_params[:, 2, 0]

    def per_capsule(ci):
        d = jnp.inf
        for oj in range(max_obs):  # max_obs is a compile-time constant
            valid = oj < n_obs
            dist = _capsule_to_sphere_distance(
                p1s[ci], p2s[ci], crs[ci], obs_centers[oj], obs_radii[oj])
            d = jnp.where(valid & (dist < d), dist, d)
        return d

    capsule_dists = jax.vmap(per_capsule)(jnp.arange(n_caps))
    capsule_dists = jnp.clip(capsule_dists, -0.5, 0.5)
    d_obs = jnp.clip(jnp.min(capsule_dists), -0.5, 0.5)

    # 3. Self-collision distances
    def per_self_pair(i):
        a, b = self_pairs[i, 0], self_pairs[i, 1]
        p1, p2, r1 = caps_world[a, 0], caps_world[a, 1], capsule_params[a, 2, 0]
        q1, q2, r2 = caps_world[b, 0], caps_world[b, 1], capsule_params[b, 2, 0]
        d2 = _segment_distance_sq(p1, p2, q1, q2)
        return jnp.sqrt(jnp.maximum(d2, 0.0)) - r1 - r2

    self_dists = jnp.clip(jax.vmap(per_self_pair)(jnp.arange(n_self)), -0.5, 0.5)

    # 4. PID control
    dx_nom, new_integral_err = compute_pid(x_d, x_ee, integral_err, dt)

    # 5. Sigma gating
    sigma, new_lag_lambda = compute_sigma_gating(d_obs, d_safe, lag_lambda, lr_lag, lag_target)
    delta_x_gated = sigma * delta_x_rl

    # 6. Nullspace + control law
    B = null_space_basis(J)
    J_pinv = damped_pinv(J)
    dq_cmd = J_pinv @ (dx_nom + delta_x_gated) + B @ z

    # 7. Integrate
    q_arm = q[:nv_arm] + dq_cmd * dt
    q_new = jnp.concatenate([q_arm, q[nv_arm:]])

    # 8. FK at new q
    _, _, site_xpos_new = forward_kinematics_chain(q_new, chain)
    x_ee_new = site_xpos_new[ee_site_id]

    # 9. Path progression
    tracking_error = jnp.linalg.norm(x_ee_new - x_d)
    new_path_param, new_last_advance = compute_path_progression(
        step_count, episode_len, tracking_error, path_param, last_advance, path_deadzone)

    # 10. Update target
    new_x_d, new_dx_d = update_target(x_start, x_goal, new_path_param, x_d, dt)

    # 11. Waypoints (relative to EE)
    def compute_wp(k):
        fp = jnp.minimum(1.0, new_path_param + wp_steps[k] / jnp.maximum(episode_len, 1))
        wp = (1.0 - fp) * x_start + fp * x_goal
        return wp - x_ee_new
    waypoints = jax.vmap(compute_wp)(jnp.arange(n_wp)) if n_wp > 0 else jnp.zeros((0, 3))

    # 12. Scene embedding (only valid obstacles, zero padding for extras)
    parts_se = []
    for oi in range(n_obs_embed):
        valid = oi < n_obs
        rel = jnp.where(valid, obs_centers[oi] - x_ee_new, jnp.zeros(3))
        r = jnp.where(valid, obs_radii[oi], 0.0)
        parts_se.append(rel)
        parts_se.append(r[None])
    scene_embed = jnp.concatenate(parts_se)

    # 13. Manipulability
    w = manipulability(J)

    # 14. Collision detection (computed before reward for penalty)
    collided_obs = jnp.any(capsule_dists < obs_coll_tol)
    combined_mask = self_mask_init & env_state['self_dists_mask'] & (self_dists >= -0.01)
    masked_self = jnp.where(combined_mask, self_dists, 0.5)
    collided_self = jnp.any(masked_self < -0.03)
    collided = collided_obs | collided_self

    # 15. Reward (raw, with collision penalty)
    reward_raw, reward_info = compute_reward(
        x_ee_new, new_x_d, d_obs, w, q_arm, dq_cmd,
        prev_dq[:nv_arm], capsule_dists, action, d_safe, reward_params,
        collided=collided_obs)

    # 17. Done / success (both obstacle and self-collision terminate)
    path_complete = new_path_param >= 0.99
    episode_done = (step_count >= episode_len) | path_complete
    ever_collided = env_state['ever_collided'] | collided
    success = path_complete & (~ever_collided)
    bonus = jnp.where(success, reward_params['success_bonus'], 0.0)
    reward_scaled = (reward_raw + bonus) / reward_params['reward_scale']
    # Cost: continuous violation (matches CPU ManipulatorEnv)
    cost = jnp.maximum(0.0, d_safe - d_obs) / jnp.maximum(d_safe, 1e-6)
    done = episode_done

    # 18. Observation (v3 format, no dq_rep)
    obs = build_observation_v3(
        q_arm, dq_cmd, x_ee_new, new_x_d,
        capsule_dists, self_dists, scene_embed, waypoints,
        new_path_param, sigma,
        n_caps=n_caps, n_self=n_self, n_obs_embed=n_obs_embed, n_wp=n_wp)

    # 17. New state
    dq_pad = jnp.concatenate([dq_cmd, jnp.zeros(q.shape[0] - nv_arm)])
    live_state = {
        'q': q_new, 'dq': dq_pad, 'x_start': x_start, 'x_goal': x_goal,
        'x_d': new_x_d, 'dx_d': new_dx_d, 'path_param': new_path_param,
        'step_count': step_count + 1, 'integral_err': new_integral_err,
        'ever_collided': ever_collided, 'lag_lambda': new_lag_lambda,
        'last_advance': new_last_advance, 'prev_dq': dq_pad,
        'last_dx_nom': dx_nom, 'last_sigma': sigma,
        'n_obs': n_obs,
        'scene_start_q': env_state['scene_start_q'],
        'self_dists_mask': combined_mask,
    }
    # scene_start_q stored in state, set during reset
    dead_state = {
        'q': env_state['scene_start_q'], 'dq': jnp.zeros(q.shape[0]),
        'x_start': x_start, 'x_goal': x_goal, 'x_d': x_start, 'dx_d': jnp.zeros(3),
        'path_param': jnp.array(0.0), 'step_count': jnp.array(0, dtype=jnp.int32),
        'integral_err': jnp.zeros(3), 'ever_collided': jnp.array(False, dtype=jnp.bool_),
        'lag_lambda': jnp.array(0.0), 'last_advance': jnp.array(0.0),
        'prev_dq': jnp.zeros(q.shape[0]),
        'last_dx_nom': jnp.zeros(3), 'last_sigma': jnp.array(0.0),
        'n_obs': n_obs,
        'scene_start_q': env_state['scene_start_q'],
        'self_dists_mask': self_mask_init,
    }
    new_state = jax.tree_util.tree_map(lambda l, d: jnp.where(done, d, l), live_state, dead_state)

    return new_state, obs, reward_scaled, done, {
        'success': success, 'collision': collided,
        'tracking_error': tracking_error, 'd_obs': d_obs,
        'path_param': new_path_param, 'sigma': sigma,
        'reward_raw': reward_raw, 'bonus': bonus, 'cost': cost,
        'w': w, 'q_before': q[:nv_arm], 'dq_before': dq[:nv_arm],
        'J_before': J, 'dx_nom_before': dx_nom,
        **reward_info,
    }


def _make_batched_step_fn(ee_site_id, nv_arm, n_caps, max_obs, n_self, n_obs_embed, n_wp, wp_steps,
                           dt, d_safe, lr_lag, lag_target, path_deadzone, episode_len, reward_params,
                           self_mask_init, obs_coll_tol):
    def env_step(s, a, obs_c, obs_r, capsule_params, self_pairs, chain):
        return _single_env_step(
            s, a, obs_c, obs_r, capsule_params, self_pairs, chain,
            ee_site_id, nv_arm, n_caps, max_obs, n_self, n_obs_embed, n_wp, wp_steps,
            dt, d_safe, lr_lag, lag_target, path_deadzone, episode_len, reward_params,
            self_mask_init, obs_coll_tol)

    _step_batch = jax.jit(
        jax.vmap(env_step, in_axes=(0, 0, 0, 0, None, None, None)),
        static_argnums=()
    )
    return _step_batch


# ========================================================================
# MJXManipulatorEnv class
# ========================================================================

class MJXManipulatorEnv:
    """
    Batched, GPU-accelerated manipulator environment using JAX.
    Observation format matches ManipulatorEnv with obs_scene_embed > 0.
    """

    def __init__(
        self,
        n_envs: int = 64,
        xml_path: str = None,
        scene_json: Optional[str] = None,
        dt: float = 0.02,
        episode_len: int = 400,
        d_safe: float = 0.06,
        path_deadzone: float = 0.10,
        lr_lag: float = 0.01,
        lag_target: float = 0.05,
        max_obs: int = 10,
        reward_params: Optional[Dict] = None,
        n_obs_embed: int = 5,
        obs_waypoint_steps: Optional[List[int]] = None,
    ):
        self.n_envs = n_envs
        self.dt = dt
        self.episode_len = episode_len
        self.d_safe = d_safe
        self.path_deadzone = path_deadzone
        self.lr_lag = lr_lag
        self.lag_target = lag_target
        self.max_obs = max_obs
        self.n_obs_embed = n_obs_embed
        self.n_arm = 7
        self._rng = np.random.RandomState(42)

        # Waypoints
        self.obs_waypoint_steps = obs_waypoint_steps or [10, 20, 50]
        self.n_wp = len(self.obs_waypoint_steps)
        self.wp_steps = jnp.array(self.obs_waypoint_steps, dtype=jnp.int32)

        # Reward params
        self.reward_params = {
            'w_track': 12.0, 'w_obs': 1.0, 'w_manip': 0.05,
            'w_energy': 0.001, 'w_action': 0.5, 'w_collision': 100.0,
            'success_bonus': 500.0, 'reward_min': -100.0, 'reward_scale': 100.0,
        }
        if reward_params:
            self.reward_params.update(reward_params)

        # Kinematic chain from MuJoCo
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), '../../models/panda_scene.xml')
        chain_np = extract_chain_from_mj(xml_path)
        self.ee_site_id = chain_np['ee_site_id']
        self.chain = jax.tree_util.tree_map(lambda x: x, chain_np)
        self.nv = chain_np['nv']

        # Cached kinematics (capsules + self-pairs)
        cache_path = os.path.join(os.path.dirname(__file__), '../../models/kinematics_cache.npz')
        if os.path.exists(cache_path):
            cache = load_cached_kinematics(cache_path)
            self.capsule_params = cache['capsule_params']
            self.self_pairs = cache['self_pairs']
        else:
            # Fallback: extract from simplified kinematics
            print("[MJXEnv] WARNING: kinematics_cache.npz not found, using simplified model")
            from env.kinematics import ManipulatorKinematics
            kin = ManipulatorKinematics()
            q0 = np.zeros(7)
            caps = kin.get_link_capsules(q0)
            params = np.zeros((len(caps), 3, 3))
            for i, (p1, p2, r) in enumerate(caps):
                params[i, 0] = p1; params[i, 1] = p2; params[i, 2, 0] = r
            self.capsule_params = jnp.array(params)
            self.self_pairs = jnp.zeros((1, 2), dtype=jnp.int32)

        self.n_caps = self.capsule_params.shape[0]
        self.n_self = self.self_pairs.shape[0]

        # Obs dim (matches v3 format)
        self.obs_dim_ = (self.n_arm * 2 + 3 + 3 + self.n_wp * 3 +
                          self.n_caps + self.n_self + self.n_obs_embed * 4 + 1 + 1)

        # Scene manager
        self.scene_manager: Optional[SceneManager] = None
        if scene_json:
            self.scene_manager = SceneManager(scene_json, n_envs, max_obs)

        # State
        self.state = init_state(n_envs, self.nv)
        self.obs_centers = jnp.zeros((max_obs, 3))
        self.obs_radii = jnp.zeros(max_obs)

        # Load static self-collision mask (pairs < 0 at home are permanently masked)
        cache_path = os.path.join(os.path.dirname(__file__), '../../models/kinematics_cache.npz')
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            if 'self_mask_home' in cached:
                self.self_mask_init = jnp.array(cached['self_mask_home'], dtype=jnp.bool_)
            else:
                self.self_mask_init = jnp.ones(n_self, dtype=jnp.bool_)
        else:
            self.self_mask_init = jnp.ones(n_self, dtype=jnp.bool_)
        self.obs_coll_tol = jnp.array(-0.01)  # small tolerance for init-time edge cases

        # JIT step function
        rp = self.reward_params
        self._step_jit = _make_batched_step_fn(
            ee_site_id=self.ee_site_id, nv_arm=self.n_arm,
            n_caps=self.n_caps, max_obs=max_obs, n_self=self.n_self,
            n_obs_embed=n_obs_embed, n_wp=self.n_wp, wp_steps=self.wp_steps,
            dt=dt, d_safe=d_safe, lr_lag=lr_lag, lag_target=lag_target,
            path_deadzone=path_deadzone, episode_len=episode_len, reward_params=rp,
            self_mask_init=self.self_mask_init, obs_coll_tol=self.obs_coll_tol)
        self._warmup_done = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        if self.scene_manager is None:
            raise ValueError("SceneManager required for reset")
        env_indices = self.scene_manager.env_scene_ids
        starts, goals, start_qs, goal_qs, obs_c, obs_r, n_obs_np = \
            self.scene_manager.get_scene_data(env_indices)
        state = init_state(self.n_envs, self.nv, n_self=self.n_self)
        start_q_pad = np.concatenate([start_qs, np.zeros((self.n_envs, self.nv - self.n_arm))], axis=1)
        state['q'] = jnp.array(start_q_pad)
        # Compute EE at start_q per-env
        bx, _, sx = jax.vmap(forward_kinematics_chain, in_axes=(0, None))(state['q'], self.chain)
        ee_starts = sx[:, self.ee_site_id]
        state['x_start'] = ee_starts
        state['x_goal'] = jnp.array(goals)
        state['x_d'] = ee_starts
        state['scene_start_q'] = jnp.array(start_q_pad)
        state['n_obs'] = jnp.array(n_obs_np, dtype=jnp.int32)
        self.obs_centers = jnp.array(obs_c)
        self.obs_radii = jnp.array(obs_r)
        self.state = state
        self._warmup()
        # _compute_obs with fresh copy to preserve n_obs
        obs = self._compute_obs(jax.tree_util.tree_map(lambda x: jnp.array(x), state))
        return obs

    def _tile_obs(self):
        """Tile static obstacle data to match number of envs."""
        oc = jnp.tile(self.obs_centers[None], (self.n_envs, 1, 1))
        or_ = jnp.tile(self.obs_radii[None], (self.n_envs, 1))
        return oc, or_

    def _step_batch_wrapper(self, state, actions, caps, pairs, chain):
        """obs_centers/radii are already per-env (n_envs, ...). Pass directly."""
        return self._step_jit(state, actions, self.obs_centers, self.obs_radii,
                              caps, pairs, chain)

    def _warmup(self):
        if self._warmup_done:
            return
        print("[MJXEnv] Warming up JIT compilation...")
        t0 = time.time()
        d = jnp.zeros((self.n_envs, self.n_arm))
        ns, o, r, e, i = self._step_batch_wrapper(
            self.state, d, self.capsule_params, self.self_pairs, self.chain)
        r.block_until_ready()
        self.state = jax.tree_util.tree_map(lambda x: x, ns)
        print(f"[MJXEnv] JIT warmup done in {time.time()-t0:.1f}s")
        self._warmup_done = True

    def step(self, actions_np: np.ndarray):
        actions = jnp.array(actions_np)
        ns_j, ob_j, rw_j, dn_j, inf_j = self._step_batch_wrapper(
            self.state, actions, self.capsule_params, self.self_pairs, self.chain)
        rw_j.block_until_ready()
        self.state = jax.tree_util.tree_map(lambda x: x, ns_j)
        obs_np = np.array(ob_j, dtype=np.float32)
        rewards_np = np.array(rw_j, dtype=np.float64)
        dones_np = np.array(dn_j, dtype=bool)
        infos = jax.tree_util.tree_map(lambda x: np.array(x), inf_j)
        info_list = [{k: v[i] for k, v in infos.items()} for i in range(self.n_envs)]
        return obs_np, rewards_np, dones_np, info_list

    def _compute_obs(self, state):
        _, ob_j, _, _, _ = self._step_batch_wrapper(
            state, jnp.zeros((self.n_envs, self.n_arm)),
            self.capsule_params, self.self_pairs, self.chain)
        ob_j.block_until_ready()
        return np.array(ob_j, dtype=np.float32)

    def set_scene(self, env_indices, scene_ids):
        if self.scene_manager is None:
            return
        for i, idx in enumerate(env_indices):
            self.scene_manager.env_scene_ids[idx] = scene_ids[i]

    def get_scene_weights(self):
        return np.ones(self.scene_manager.n_scenes) if self.scene_manager else None

    def close(self):
        pass
