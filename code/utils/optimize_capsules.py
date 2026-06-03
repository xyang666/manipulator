"""
Optimize capsule parameters (p1, p2, radius) per link to tightly fit mesh STL geometry.
Outputs optimized collision_specs for kinematics.py.

Usage:
    code/.venv/bin/python code/utils/optimize_capsules.py
"""

import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


# ── Distance functions ───────────────────────────────────────────────

def point_to_segment_distance(pts, p1, p2):
    """Distance from point(s) to line segment p1-p2.

    Args:
        pts: (N, 3) or (3,) array
        p1: (3,) segment start
        p2: (3,) segment end
    Returns:
        (N,) or scalar distance from each point to the segment centerline
    """
    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 3)
    seg = p2 - p1
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-12:
        d = np.linalg.norm(pts - p1, axis=1)
    else:
        t = np.dot(pts - p1, seg) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest = p1 + t[:, None] * seg
        d = np.linalg.norm(pts - closest, axis=1)
    return d.item() if single else d


def capsule_signed_distance(pts, p1, p2, r):
    """Signed distance from points to capsule surface. Positive = outside."""
    return point_to_segment_distance(pts, p1, p2) - r


# ── Capsule surface sampling ─────────────────────────────────────────

def sample_capsule_surface(p1, p2, r, n_pts=500):
    """Deterministic sampling of points on capsule surface."""
    axis = p2 - p1
    length = np.linalg.norm(axis)
    if length < 1e-12:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / length

    # Local frame
    z = axis
    if abs(z[2]) < 0.99:
        x = np.cross(z, [0, 0, 1])
    else:
        x = np.cross(z, [0, 1, 0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    pts = []
    n_cyl = n_pts // 2
    n_hemi = n_pts - n_cyl

    # Cylinder body: deterministic grid (t=0..length, theta=0..2pi)
    n_t = int(np.sqrt(n_cyl * length / (2 * np.pi * r))) or 1
    n_theta = n_cyl // n_t
    for ti in range(n_t):
        t = (ti + 0.5) / n_t * length
        for thi in range(n_theta):
            theta = 2 * np.pi * thi / n_theta
            pt = p1 + t * z + r * (np.cos(theta) * x + np.sin(theta) * y)
            pts.append(pt)

    # Hemispheres: Fibonacci sphere
    n_each = n_hemi // 2
    for center in [p2, p1]:
        for i in range(n_each):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_each)
            theta = np.pi * (1 + np.sqrt(5)) * i
            offset = r * (np.sin(phi) * (np.cos(theta) * x + np.sin(theta) * y) + np.cos(phi) * z)
            pts.append(center + offset)

    return np.array(pts[:n_pts])


# ── Per-link optimization ────────────────────────────────────────────

def kmeans_split(pts, k=2, n_iter=20):
    """Simple k-means clustering. Returns k cluster indices."""
    n = len(pts)
    idx = np.random.RandomState(0).choice(k, n)
    centers = np.zeros((k, 3))
    for _ in range(n_iter):
        for j in range(k):
            mask = idx == j
            centers[j] = pts[mask].mean(axis=0) if mask.sum() > 0 else pts[j % n]
        # assign
        d = np.linalg.norm(pts[:, None] - centers[None], axis=2)
        idx = d.argmin(axis=1)
    return idx


def optimize_capsule_for_mesh(mesh_verts, p1_init, p2_init, r_init,
                               w_outside=5.0, w_fat=1.0, margin=0.005, n_caps=1):
    """Optimize capsule (p1, p2, r) to tightly cover mesh vertices.

    Objective:
        L = w_outside * mean(ReLU(signed_dist(v, capsule)))  -- mesh outside = bad
          + w_fat * mean(cap_surface_dist_to_mesh)            -- capsule too fat = bad
          + w_dev * max(cap_surface_dist_to_mesh)             -- max deviation penalty

    Returns: list of (p1_opt, p2_opt, r_opt) tuples (length = n_caps)
    """
    from scipy.optimize import minimize

    mesh_verts = np.asarray(mesh_verts, dtype=np.float64)
    margin_ = margin

    if n_caps > 1:
        # Strategy: first fit 1 capsule to the whole mesh, then fit a second
        # capsule to the vertices with highest residual error.
        cap0 = _optimize_single(mesh_verts, None, None, None,
                                w_outside, w_fat, margin_)
        p1_0, p2_0, r_0 = cap0
        # Find worst-covered vertices
        signed_d = capsule_signed_distance(mesh_verts, p1_0, p2_0, r_0)
        # Take top 20% worst vertices
        threshold = np.percentile(signed_d, 80)
        outlier_mask = signed_d >= threshold
        if outlier_mask.sum() < 10:
            results = [cap0]
        else:
            outlier_verts = mesh_verts[outlier_mask]
            cap1 = _optimize_single(outlier_verts, None, None, None,
                                    w_outside, w_fat, margin_ * 0.5)
            results = [cap0, cap1]
        return results

    # n_caps == 1
    return [_optimize_single(mesh_verts, p1_init, p2_init, r_init,
                             w_outside, w_fat, margin_)]

    # Precompute mesh bounding info for initialization
    centroid = mesh_verts.mean(axis=0)
    pca_init = p1_init is None or p2_init is None

    if pca_init:
        # PCA-based initialization
        cov = np.cov(mesh_verts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_axis = eigvecs[:, -1]  # largest eigenvector
        projections = mesh_verts @ principal_axis
        t_min, t_max = projections.min(), projections.max()
        # Extend slightly beyond vertex extents
        ext = 0.02 * (t_max - t_min + 0.01)
        p1_init = centroid + (t_min - ext) * principal_axis
        p2_init = centroid + (t_max + ext) * principal_axis
        # Initial radius: 95th percentile distance to segment
        dists = point_to_segment_distance(mesh_verts, p1_init, p2_init)
        r_init = np.percentile(dists, 95) + margin
        print(f"    PCA init: p1=({p1_init[0]:.3f},{p1_init[1]:.3f},{p1_init[2]:.3f}) "
              f"p2=({p2_init[0]:.3f},{p2_init[1]:.3f},{p2_init[2]:.3f}) r={r_init:.3f}")

    # Fixed capsule surface sample points for deterministic optimization
    cap_samples = sample_capsule_surface(p1_init, p2_init, r_init, n_pts=300)
    # Fixed: we'll re-sample at each evaluation based on current params

    def objective(params):
        p1 = params[:3]
        p2 = params[3:6]
        r = params[6]

        # Constraints via barrier
        if r < 0.003:
            return 1e6
        seg_len = np.linalg.norm(p2 - p1)
        if seg_len < 0.003:
            return 1e6

        # Mesh -> capsule: how much mesh sticks out
        signed_d = capsule_signed_distance(mesh_verts, p1, p2, r)
        outside = np.clip(signed_d, 0, None)  # ReLU
        loss_outside = np.mean(outside)
        max_outside = np.max(outside) if len(outside) > 0 else 0.0

        # Capsule -> mesh: how fat is the capsule
        samp = sample_capsule_surface(p1, p2, r, n_pts=300)
        d_cap = np.min(np.linalg.norm(samp[:, None] - mesh_verts[None], axis=2), axis=1)
        loss_fat = np.mean(d_cap)
        max_fat = np.max(d_cap)

        total = (w_outside * loss_outside
                 + 2.0 * w_outside * max_outside
                 + w_fat * loss_fat
                 + 0.5 * max_fat)
        return total

    # Initial guess
    x0 = np.concatenate([p1_init, p2_init, [r_init]])

    # Bounds
    bounds = [(None, None)] * 6 + [(0.005, 0.15)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})

    p1_opt = result.x[:3]
    p2_opt = result.x[3:6]
    r_opt = result.x[6]

    return p1_opt, p2_opt, r_opt


def _optimize_single(mesh_verts, p1_init, p2_init, r_init,
                      w_outside=5.0, w_fat=1.0, margin=0.005):
    """Single-capsule optimization helper (called by optimize_capsule_for_mesh)."""
    from scipy.optimize import minimize

    mesh_verts = np.asarray(mesh_verts, dtype=np.float64)
    centroid = mesh_verts.mean(axis=0)

    if p1_init is None:
        # PCA init
        cov = np.cov(mesh_verts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_axis = eigvecs[:, -1]
        projections = mesh_verts @ principal_axis
        t_min, t_max = projections.min(), projections.max()
        ext = 0.02 * (t_max - t_min + 0.01)
        p1_init = centroid + (t_min - ext) * principal_axis
        p2_init = centroid + (t_max + ext) * principal_axis
        dists = point_to_segment_distance(mesh_verts, p1_init, p2_init)
        r_init = np.percentile(dists, 95) + margin
        print(f"      PCA init: r={r_init:.3f}")

    def objective(params):
        p1 = params[:3]
        p2 = params[3:6]
        r = params[6]
        if r < 0.003:
            return 1e6
        if np.linalg.norm(p2 - p1) < 0.003:
            return 1e6

        signed_d = capsule_signed_distance(mesh_verts, p1, p2, r)
        outside = np.clip(signed_d, 0, None)
        loss_outside = np.mean(outside)
        max_outside = np.max(outside) if len(outside) > 0 else 0.0

        samp = sample_capsule_surface(p1, p2, r, n_pts=300)
        d_cap = np.min(np.linalg.norm(samp[:, None] - mesh_verts[None], axis=2), axis=1)
        loss_fat = np.mean(d_cap)
        max_fat = np.max(d_cap)

        return (w_outside * loss_outside + 2.0 * w_outside * max_outside
                + w_fat * loss_fat + 0.5 * max_fat)

    x0 = np.concatenate([p1_init, p2_init, [r_init]])
    bounds = [(None, None)] * 6 + [(0.005, 0.15)]
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
    return result.x[:3], result.x[3:6], result.x[6]


# ── Link definitions ─────────────────────────────────────────────────

LINK_MESH_FILES = [
    ("panda_link0", "link0.stl", 1),
    ("panda_link1", "link1.stl", 1),
    ("panda_link2", "link2.stl", 1),
    ("panda_link3", "link3.stl", 1),
    ("panda_link4", "link4.stl", 1),
    ("panda_link5", "link5.stl", 2),  # 2 capsules (bent link)
    ("panda_link6", "link6.stl", 1),
    ("panda_link7", "link7.stl", 1),
    ("panda_hand", "hand.stl", 1),
]

# Frame names used in Pinocchio for each link
LINK_FRAME_NAMES = {
    "panda_link0": "panda_link0",
    "panda_link1": "panda_link1",
    "panda_link2": "panda_link2",
    "panda_link3": "panda_link3",
    "panda_link4": "panda_link4",
    "panda_link5": "panda_link5",
    "panda_link6": "panda_link6",
    "panda_link7": "panda_link7",
    "panda_hand": "panda_hand",
}


# ── Main ─────────────────────────────────────────────────────────────

def main():
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

    model = pin.buildModelFromUrdf(_urdf)
    data = model.createData()

    # q=0 (home position, model.nq=9 for Panda with fingers)
    q = np.zeros(model.nq)
    pin.framesForwardKinematics(model, data, q)

    results = {}
    current_specs = {}

    print("=" * 100)
    print("Optimizing capsule parameters for each link...")
    print("=" * 100)

    for link_name, mesh_file, n_caps in LINK_MESH_FILES:
        mesh_path = os.path.join(_mesh_dir, mesh_file)
        if not os.path.exists(mesh_path):
            print(f"  [SKIP] {link_name}: mesh not found at {mesh_path}")
            continue

        mesh_verts = read_stl_vertices(mesh_path)
        print(f"\n  [{link_name}] loaded mesh with {len(mesh_verts)} vertices (n_caps={n_caps})")

        # Get frame transform from Pinocchio at q=0
        try:
            frame_id = model.getFrameId(link_name)
        except Exception:
            print(f"  [SKIP] {link_name}: frame not found in URDF")
            continue

        oMf = data.oMf[frame_id]
        R, t = oMf.rotation, oMf.translation

        # Transform mesh to world frame at q=0
        verts_world = (R @ mesh_verts.T).T + t

        # Run optimization
        caps_opt = optimize_capsule_for_mesh(verts_world, None, None, None,
                                              n_caps=n_caps)

        caps_body_list = []
        for ci, (p1_opt, p2_opt, r_opt) in enumerate(caps_opt):
            # Transform back to body frame
            p1_body = R.T @ (p1_opt - t)
            p2_body = R.T @ (p2_opt - t)

            # Evaluate fit quality
            signed_d = capsule_signed_distance(verts_world, p1_opt, p2_opt, r_opt)
            outside_pct = 100.0 * np.mean(signed_d > 0)
            max_outside = float(np.max(signed_d)) if len(signed_d) > 0 else 0.0
            cap_samp = sample_capsule_surface(p1_opt, p2_opt, r_opt, n_pts=300)
            d_cap = np.mean(np.min(np.linalg.norm(
                cap_samp[:, None] - verts_world[None], axis=2), axis=1))

            caps_body_list.append({
                "p1_body": p1_body,
                "p2_body": p2_body,
                "radius": r_opt,
                "outside_pct": outside_pct,
                "max_outside": max_outside,
                "cap_to_mesh": d_cap,
            })

            # Combine all capsule metrics for reporting
            print(f"    → Capsule {ci}: r={r_opt:.3f}, "
                  f"outside={outside_pct:.1f}% (max={max_outside:.4f}m), "
                  f"cap→mesh={d_cap:.4f}m")
            print(f"      p1_body=({p1_body[0]:.3f},{p1_body[1]:.3f},{p1_body[2]:.3f})")
            print(f"      p2_body=({p2_body[0]:.3f},{p2_body[1]:.3f},{p2_body[2]:.3f})")

        results[link_name] = caps_body_list

    # ── Summary output ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY: Optimized collision_specs for kinematics.py")
    print("=" * 100)
    print()

    for link_name, mesh_file, n_caps in LINK_MESH_FILES:
        if link_name not in results:
            continue
        caps_list = results[link_name]
        print(f'        "{link_name}": [')
        for ci, caps in enumerate(caps_list):
            p1 = caps["p1_body"]
            p2 = caps["p2_body"]
            radius = caps["radius"]
            print(f"            (np.array([{p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f}]),")
            print(f"             np.array([{p2[0]:.4f}, {p2[1]:.4f}, {p2[2]:.4f}]), {radius:.3f}),")
        print(f"        ],")

    # ── Quality report ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("QUALITY REPORT (vs original)")
    print("=" * 100)
    print(f"{'Link':<20} {'Original Cap→Mesh':>18} {'Optimized Cap→Mesh':>20} {'Mesh outside%':>14}")
    print("-" * 100)
    # Original values from check_geom_gap run
    orig = {
        "panda_link0": 0.0455, "panda_link1": 0.1009, "panda_link2": 0.0648,
        "panda_link3": 0.0827, "panda_link4": 0.0615, "panda_link5": 0.0688,
        "panda_link6": 0.1244, "panda_link7": 0.1000, "panda_hand": 0.1027,
    }
    avg_orig = 0
    avg_new = 0
    n_links = 0
    for link_name, mesh_file, n_caps in LINK_MESH_FILES:
        if link_name not in results:
            continue
        caps_list = results[link_name]
        o = orig.get(link_name, 0)
        # Best capsule for this link (lowest cap->mesh)
        best_cap = min(caps_list, key=lambda c: c["cap_to_mesh"])
        avg_orig += o
        avg_new += best_cap["cap_to_mesh"]
        n_links += 1
        outside_str = f"{best_cap['outside_pct']:.1f}% (max {best_cap['max_outside']:.3f})"
        print(f"{link_name:<20} {o:>18.4f} {best_cap['cap_to_mesh']:>20.4f} {outside_str:>14}")
    print("-" * 100)
    if n_links > 0:
        print(f"{'AVERAGE':<20} {avg_orig/n_links:>18.4f} {avg_new/n_links:>20.4f}")


if __name__ == "__main__":
    main()
