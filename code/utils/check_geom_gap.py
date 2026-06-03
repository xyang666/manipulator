"""
Compare capsule approximations (used by SDF) against actual MuJoCo mesh STL geometry.

For each link, at a given joint configuration:
  - Get capsule endpoints + radius from kinematics
  - Get mesh vertices from STL file, transformed to world frame via Pinocchio FK
  - Compute: how far does the capsule surface deviate from the mesh surface?

Usage:
    code/.venv/bin/python code/utils/check_geom_gap.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.kinematics import ManipulatorKinematics


# ── STL parsing (binary only, little-endian) ──────────────────────────

def read_stl_vertices(path: str) -> np.ndarray:
    """Return (N, 3) array of unique vertex positions from a binary STL file."""
    with open(path, "rb") as f:
        header = f.read(80)
        n_faces = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        vertices = []
        for _ in range(n_faces):
            f.read(12)  # normal
            v0 = np.frombuffer(f.read(12), dtype=np.float32)
            v1 = np.frombuffer(f.read(12), dtype=np.float32)
            v2 = np.frombuffer(f.read(12), dtype=np.float32)
            vertices.extend([v0, v1, v2])
            f.read(2)  # attribute byte count
    return np.unique(np.array(vertices).reshape(-1, 3), axis=0)


# ── Capsule point cloud sampling ──────────────────────────────────────

def sample_capsule(p1, p2, radius, n_pts=200):
    """
    Sample points on capsule surface (including endcaps).
    Returns (n_pts, 3) array.
    """
    axis = p2 - p1
    length = np.linalg.norm(axis)
    if length < 1e-12:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / length

    # Build local frame
    z = axis
    if abs(z[2]) < 0.99:
        x = np.cross(z, [0, 0, 1])
    else:
        x = np.cross(z, [0, 1, 0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    pts = []
    n_per_hemisphere = n_pts // 3
    n_cylinder = n_pts - 2 * n_per_hemisphere

    # Cylinder body
    for _ in range(n_cylinder):
        t = np.random.uniform(0, length)
        theta = np.random.uniform(0, 2 * np.pi)
        pt = p1 + t * z + radius * (np.cos(theta) * x + np.sin(theta) * y)
        pts.append(pt)

    # Two hemispheres
    for sign, center in [(1, p2), (-1, p1)]:
        for _ in range(n_per_hemisphere):
            # Von Mises Fisher for uniform sphere sampling
            theta = np.random.uniform(0, 2 * np.pi)
            cos_phi = np.random.uniform(0, 1)
            sin_phi = np.sqrt(1 - cos_phi * cos_phi)
            r = radius
            pt = center + sign * r * cos_phi * z + r * sin_phi * (np.cos(theta) * x + np.sin(theta) * y)
            pts.append(pt)

    return np.array(pts)


def point_to_mesh_distance(pt, mesh_verts):
    """Minimum Euclidean distance from a point to any mesh vertex."""
    return float(np.min(np.linalg.norm(mesh_verts - pt.reshape(1, 3), axis=1)))


# ── Link name → capsule index mapping ─────────────────────────────────

# From kinematics.py _get_capsule_link_indices / get_link_capsules
LINK_NAMES = [
    "panda_link0", "panda_link1", "panda_link2", "panda_link3",
    "panda_link4", "panda_link5",
    "panda_link6", "panda_link7", "panda_hand",
    "panda_leftfinger", "panda_rightfinger",
]

# Each (link_name, mesh_stl_file, capsule_index)
# capsule_index accounts for link5 having 2 capsules in the kinematics list
LINK_MESH_FILES = [
    ("panda_link0", "link0.stl", 0),
    ("panda_link1", "link1.stl", 1),
    ("panda_link2", "link2.stl", 2),
    ("panda_link3", "link3.stl", 3),
    ("panda_link4", "link4.stl", 4),
    ("panda_link5", "link5.stl", 5),   # first of link5's 2 capsules
    ("panda_link6", "link6.stl", 7),
    ("panda_link7", "link7.stl", 8),
    ("panda_hand", "hand.stl", 9),
    # fingers use box geoms, not STL meshes — skip
]

# Finger geoms are boxes in XML — for completeness, add approximate spheres
# (leftfinger and rightfinger have small box geoms, not STL meshes)


def main_pinocchio():
    """Compare capsules vs meshes using Pinocchio for FK transforms."""
    import pinocchio as pin

    _code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _urdf = os.path.join(
        _code_dir,
        ".venv/lib/python3.12/site-packages/cmeel.prefix"
        "/share/example-robot-data/robots/panda_description/urdf/panda.urdf",
    )
    _mesh_dir = os.path.join(
        _code_dir,
        ".venv/lib/python3.12/site-packages/cmeel.prefix"
        "/share/example-robot-data/robots/panda_description/meshes/collision",
    )

    kin = ManipulatorKinematics(_urdf)
    model = kin.model
    data = kin.data

    # Full link list (all bodies, not just capsules)
    all_link_names = [n for n in model.names if n.startswith("panda_") and n != "panda"]
    print(f"Found {len(all_link_names)} links: {all_link_names}")

    # Load meshes
    mesh_data = {}
    for link_name, mesh_file, _ in LINK_MESH_FILES:
        path = os.path.join(_mesh_dir, mesh_file)
        if os.path.exists(path):
            mesh_data[link_name] = read_stl_vertices(path)

    # Frame names from URDF (need to check naming)
    # In the Panda URDF, visual/collision frames may be named differently
    # Let's use model.frames to find the right frame for each link
    frame_names = list(model.frames)
    print(f"Frame names (first 20): {frame_names[:20]}")

    n_configs = 10
    rng = np.random.RandomState(42)

    print(f"\n{'='*100}")
    hdr = f"{'Link':<20} {'Cap→Mesh':>10} {'Mesh→Cap':>10} {'Max Dev':>10}"
    print(hdr)
    print(f"{'='*100}")

    results_overall = {"cap_to_mesh": [], "mesh_to_cap": [], "max_dev": []}

    for ci in range(n_configs):
        q = rng.uniform(low=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                        high=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        q = q.astype(np.float64)

        pin.framesForwardKinematics(model, data, q)
        capsules = kin.get_link_capsules(q)  # 12 capsules, world frame

        # Build pin frame name → link name mapping
        for link_name, mesh_file, cap_idx in LINK_MESH_FILES:
            if link_name not in mesh_data:
                continue

            mesh_verts = mesh_data[link_name]  # local frame

            # Find frame in pinocchio model
            try:
                frame_id = model.getFrameId(link_name)
            except:
                try:
                    frame_id = model.getFrameId(link_name.replace("panda_", ""))
                except:
                    continue

            # Transform mesh vertices to world frame
            oMf = data.oMf[frame_id]  # SE3 transform
            R = oMf.rotation
            t = oMf.translation.reshape(1, 3)
            verts_world = (R @ mesh_verts.T).T + t  # (N, 3)

            # Capsule for this link (using explicit capsule index)
            if cap_idx < len(capsules):
                p1, p2, cap_r = capsules[cap_idx]
            else:
                continue

            # Sample points on capsule surface
            cap_pts = sample_capsule(p1, p2, cap_r, n_pts=300)

            # Capsule → mesh: for each capsule point, min distance to mesh
            # Use KDTree-like approach (brute force, small enough)
            # Actually, for small meshes brute force is fine
            d_cap_to_mesh = []
            for pt in cap_pts:
                dists = np.linalg.norm(verts_world - pt.reshape(1, 3), axis=1)
                d_cap_to_mesh.append(dists.min())

            # Mesh → capsule: for each mesh vertex, min distance to capsule surface
            d_mesh_to_cap = []
            for vert in verts_world:
                # Distance to capsule segment
                v = vert - p1
                axis = p2 - p1
                seg_len = np.linalg.norm(axis)
                if seg_len < 1e-12:
                    closest = p1
                else:
                    t_ = np.dot(v, axis) / (seg_len * seg_len)
                    t_ = np.clip(t_, 0, 1)
                    closest = p1 + t_ * axis
                dist = np.linalg.norm(vert - closest) - cap_r
                d_mesh_to_cap.append(dist)

            d_cap_to_mesh = np.array(d_cap_to_mesh)
            d_mesh_to_cap = np.array(d_mesh_to_cap)

            avg_c2m = d_cap_to_mesh.mean()
            avg_m2c = d_mesh_to_cap.mean()
            max_dev = max(d_cap_to_mesh.max(), abs(d_mesh_to_cap).max())

            results_overall["cap_to_mesh"].append(avg_c2m)
            results_overall["mesh_to_cap"].append(avg_m2c)
            results_overall["max_dev"].append(max_dev)

            if ci == 0:  # Print first config only
                print(f"{link_name:<20} {avg_c2m:>10.4f} {avg_m2c:>10.4f} {max_dev:>10.4f}")

    print(f"\n{'='*100}")
    print(f"Summary over {n_configs} random configurations:")
    print(f"  Avg capsule→mesh distance: {np.mean(results_overall['cap_to_mesh']):.4f}m")
    print(f"  Avg mesh→capsule distance:  {np.mean(results_overall['mesh_to_cap']):.4f}m")
    print(f"  Max deviation:              {np.max(results_overall['max_dev']):.4f}m")
    print(f"  Capsule→mesh 95th pct:     {np.percentile(results_overall['cap_to_mesh'], 95):.4f}m")
    print(f"  Mesh→capsule 95th pct:     {np.percentile(results_overall['mesh_to_cap'], 95):.4f}m")


if __name__ == "__main__":
    main_pinocchio()


