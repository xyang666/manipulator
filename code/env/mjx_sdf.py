"""
mjx_sdf.py
----------
JAX-based signed distance field for obstacle avoidance.
Obstacles are spheres; robot links are capsules.
All functions are JAX-compatible (pure, jittable, vmappable).
"""

import jax.numpy as jnp
from jax import lax


def capsule_to_sphere_distance(p1, p2, capsule_radius, sphere_center, sphere_radius):
    """
    Signed distance between a capsule (p1-p2 segment + radius) and a sphere.

    Returns: scalar (positive = separated, negative = penetration).
    """
    segment = p2 - p1
    segment_length = jnp.linalg.norm(segment)

    def degenerate(_):
        center_dist = jnp.linalg.norm(sphere_center - p1)
        return center_dist - capsule_radius - sphere_radius

    def normal_case(_):
        direction = segment / segment_length
        t = jnp.dot(sphere_center - p1, direction)
        t = jnp.clip(t, 0.0, segment_length)
        closest_point = p1 + t * direction
        center_dist = jnp.linalg.norm(sphere_center - closest_point)
        return center_dist - capsule_radius - sphere_radius

    return lax.cond(segment_length < 1e-8, degenerate, normal_case, None)


def per_capsule_distances(capsule_params, obs_centers, obs_radii):
    """
    For each capsule, min distance to any obstacle.

    Args:
        capsule_params: (n_caps, 2, 3) for (p1, p2) and (n_caps,) for radii.
        obs_centers: (n_obs, 3)
        obs_radii: (n_obs,)

    Returns: (n_caps,) signed distances, clipped to [-0.5, 0.5].
    """
    p1 = capsule_params[:, 0, :]  # (n_caps, 3)
    p2 = capsule_params[:, 1, :]  # (n_caps, 3)
    cap_radii = capsule_params[:, 2, 0]  # (n_caps,)
    n_caps = p1.shape[0]
    n_obs = obs_centers.shape[0]

    def per_capsule_dist(i):
        def per_obs_dist(j):
            dist = capsule_to_sphere_distance(
                p1[i], p2[i], cap_radii[i],
                obs_centers[j], obs_radii[j],
            )
            return dist

        if n_obs > 0:
            dists = lax.map(per_obs_dist, jnp.arange(n_obs))
            d_min = jnp.min(dists)
        else:
            d_min = 0.5
        return jnp.clip(d_min, -0.5, 0.5)

    if n_caps > 0:
        return lax.map(per_capsule_dist, jnp.arange(n_caps))
    return jnp.array([], dtype=jnp.float32)


def min_distance(x_ee, capsule_params, obs_centers, obs_radii):
    """Global min distance from all capsules to all obstacles."""
    caps = capsule_params
    n_obs = obs_centers.shape[0]
    if n_obs == 0:
        return jnp.inf

    n_caps = caps.shape[0]

    def body_fn(i, d):
        p1 = caps[i, 0]
        p2 = caps[i, 1]
        cr = caps[i, 2, 0]

        def obs_loop(j, d2):
            dist = capsule_to_sphere_distance(p1, p2, cr, obs_centers[j], obs_radii[j])
            return jnp.minimum(d2, dist)

        return jnp.minimum(d, lax.fori_loop(0, n_obs, obs_loop, jnp.inf))

    return jnp.where(n_caps > 0, lax.fori_loop(0, n_caps, body_fn, jnp.inf), jnp.inf)
