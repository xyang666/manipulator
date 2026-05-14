"""
Batch test: run KP and SAC on all challenge_close scenes.
Reports success rate and collision rate for each method.

Usage:
  code/.venv/bin/python -u code/batch_test_challenge.py
"""
import sys, os, json, numpy as np, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCENE_JSON = os.path.join(_ROOT, "results/challenge_close.json")
CKPT_DIR = os.path.join(_ROOT, "checkpoints/phase3_smallrad_v2")
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt_best.pt")

URDF = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                    "/share/example-robot-data/robots/panda_description/urdf/panda.urdf")
XML = os.path.join(_ROOT, "models/panda_scene.xml")

# ---------------------------------------------------------------------------
# Load scenes
# ---------------------------------------------------------------------------
with open(SCENE_JSON) as f:
    SCENES = json.load(f)
print(f"Loaded {len(SCENES)} scenes from {SCENE_JSON}")


def evaluate_scene(env, scene, get_action=None, max_steps=1000):
    """Run a single scene and return metrics."""
    sys.path.insert(0, _HERE)
    from test import apply_scene
    apply_scene(env, scene)

    tracking_errors = []
    obstacle_distances = []
    collisions = 0
    success = False

    for step in range(max_steps):
        if get_action is not None:
            action = get_action(env._get_obs())
        else:
            action = np.zeros(env.act_dim)

        obs, reward, done, info = env.step(action)

        x_ee, _ = env.kin.forward_kinematics(env.q)
        track_err = np.linalg.norm(x_ee - env.x_d)
        d_obs = info.get("d_obs", 0)

        tracking_errors.append(track_err)
        obstacle_distances.append(d_obs)

        if info.get("collision", False):
            collisions += 1

        if done:
            if info.get("success", False):
                success = True
            break

    # Final state
    x_ee_final, _ = env.kin.forward_kinematics(env.q)
    final_dist = np.linalg.norm(x_ee_final - np.array(env.x_goal))

    return {
        "success": success,
        "collisions": collisions,
        "final_distance": float(final_dist),
        "mean_track_error": float(np.mean(tracking_errors)),
        "mean_d_obs": float(np.mean(obstacle_distances)),
        "min_d_obs": float(np.min(obstacle_distances)),
        "steps": len(tracking_errors),
    }


def batch_test(get_action, label, max_steps=1000):
    """Run all scenes with the given policy."""
    from env.manipulator_env import ManipulatorEnv

    results = []
    for sid, scene in enumerate(SCENES):
        n_obs = len(scene["obstacles"])
        env = ManipulatorEnv(
            urdf_path=URDF, xml_path=XML, n_joints=7,
            dt=0.02, episode_len=max_steps,
            n_obstacles=n_obs, obs_radius=0.02,
            controller="rl",
        )
        env.reset()
        env.reward_fn.w_collision = 100.0

        t0 = time.time()
        metrics = evaluate_scene(env, scene, get_action, max_steps)
        elapsed = time.time() - t0
        results.append(metrics)

        flag = "✓" if metrics["success"] else "✗"
        coll_flag = f" COLL={metrics['collisions']}" if metrics["collisions"] > 0 else ""
        print(f"  [{label}] scene {sid:2d} ({n_obs}obs): {flag}"
              f" final_dist={metrics['final_distance']:.3f}"
              f" min_d_obs={metrics['min_d_obs']:.3f}{coll_flag}"
              f"  ({elapsed:.1f}s)")

        env.close()

    return results


# ---------------------------------------------------------------------------
# Quick KP test on a few scenes first
# ---------------------------------------------------------------------------
def quick_kp_test(n_scenes=5):
    """Quick KP test to see if obstacles are close enough."""
    from env.manipulator_env import ManipulatorEnv

    print(f"\n{'='*60}")
    print(f"Quick KP test: {n_scenes} scenes")
    print(f"{'='*60}")

    collisions = 0
    for sid in range(min(n_scenes, len(SCENES))):
        scene = SCENES[sid]
        n_obs = len(scene["obstacles"])
        env = ManipulatorEnv(
            urdf_path=URDF, xml_path=XML, n_joints=7,
            dt=0.02, episode_len=1000,
            n_obstacles=n_obs, obs_radius=0.02,
            controller="rl",
        )
        env.reset()
        t0 = time.time()
        metrics = evaluate_scene(env, scene, get_action=None, max_steps=1000)
        elapsed = time.time() - t0
        flag = "✓" if metrics["success"] else "✗"
        c = "COLLISION" if metrics["collisions"] > 0 else "clear"
        print(f"  KP scene {sid:2d} ({n_obs}obs): {flag} final={metrics['final_distance']:.3f}"
              f" min_d_obs={metrics['min_d_obs']:.3f} {c} ({elapsed:.1f}s)")
        if metrics["collisions"] > 0:
            collisions += 1
        env.close()

    print(f"\nKP collisions: {collisions}/{n_scenes}")
    return collisions


if __name__ == "__main__":
    # First do a quick KP test
    kp_collisions = quick_kp_test(10)
    if kp_collisions == 0:
        print("\n⚠ KP clears all tested scenes — obstacles not close enough!")
        print("  Consider regenerating with tighter obstacle distances.")
        print("  Testing all 45 scenes to be sure...\n")

    # Full KP test on all 45 scenes
    print(f"\n{'='*60}")
    print("Full KP batch test (all 45 scenes)")
    print(f"{'='*60}")
    kp_results = batch_test(get_action=None, label="KP", max_steps=1000)

    kp_success = sum(1 for r in kp_results if r["success"])
    kp_coll = sum(1 for r in kp_results if r["collisions"] > 0)
    print(f"\nKP Results: {kp_success}/{len(kp_results)} success, {kp_coll}/{len(kp_results)} collided")

    if kp_coll == 0:
        print("⚠ KP collided on 0 scenes — regenerating with closer obstacles!")
        sys.exit(1)

    # SAC test
    print(f"\n{'='*60}")
    print("SAC batch test (all 45 scenes)")
    print(f"{'='*60}")

    if not os.path.exists(CKPT_PATH):
        print(f"SAC checkpoint not found at {CKPT_PATH}")
        sys.exit(1)

    # Build SAC agent once
    import torch
    from agent.sac_agent import SACAgent
    from env.dynamics import ManipulatorDynamics

    # Load config to get dimensions
    with open(os.path.join(CKPT_DIR, "config.json")) as f:
        cfg = json.load(f)
    cli = cfg.get("cli_args", {})

    # Use first scene to determine obs_dim
    env_proto = ManipulatorEnv(
        urdf_path=URDF, xml_path=XML, n_joints=7,
        dt=0.02, episode_len=5,
        n_obstacles=len(SCENES[0]["obstacles"]), obs_radius=0.02,
        controller="rl",
        # Match training config for obs
        obs_scene_embed=cli.get("obs_scene_embed", 5),
        obs_waypoint_steps=[int(x) for x in cli.get("obs_waypoint_steps", "10,20,50").split(",")],
    )
    env_proto.reset()
    obs_dim = env_proto.obs_dim
    act_dim = env_proto.act_dim
    env_proto.close()
    print(f"[SAC] obs_dim={obs_dim}, act_dim={act_dim}")

    hidden_dims = tuple(cli.get("hidden_dims", [256, 256]))
    dyn = ManipulatorDynamics(URDF)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = SACAgent(state_dim=obs_dim, action_dim=act_dim, dynamics=dyn,
                     hidden_dims=hidden_dims, device=device)
    meta = agent.load(CKPT_PATH, load_optimizers=False)
    agent.actor.eval()
    print(f"[SAC] Loaded: {meta}")

    sac_results = batch_test(
        get_action=lambda obs: agent.select_action(obs, deterministic=True),
        label="SAC", max_steps=1000,
    )

    sac_success = sum(1 for r in sac_results if r["success"])
    sac_coll = sum(1 for r in sac_results if r["collisions"] > 0)
    print(f"\nSAC Results: {sac_success}/{len(sac_results)} success, {sac_coll}/{len(sac_results)} collided")

    # Summary comparison
    print(f"\n{'='*60}")
    print("KP vs SAC comparison (challenge_close.json)")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'KP':>15} {'SAC':>15}")
    print("-" * 62)
    print(f"{'Success rate':<30} {kp_success/len(kp_results)*100:>14.1f}% {sac_success/len(sac_results)*100:>14.1f}%")
    print(f"{'Collision rate':<30} {kp_coll/len(kp_results)*100:>14.1f}% {sac_coll/len(sac_results)*100:>14.1f}%")
    print(f"{'Mean min d_obs':<30} {np.mean([r['min_d_obs'] for r in kp_results]):>15.4f} {np.mean([r['min_d_obs'] for r in sac_results]):>15.4f}")
    print(f"{'Mean final distance':<30} {np.mean([r['final_distance'] for r in kp_results]):>15.4f} {np.mean([r['final_distance'] for r in sac_results]):>15.4f}")
