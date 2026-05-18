"""
Per-scene evaluation of a trained checkpoint.
Evaluates each validation scene individually and logs detailed metrics.
"""
import json, sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from env.manipulator_env import ManipulatorEnv
from env.dynamics import ManipulatorDynamics
from agent.sac_agent import SACAgent
from agent.ppo_agent import PPOAgent

# ---------------------------------------------------------------------------
# Paths (matches train.py defaults)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENV_DATA = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                          "/share/example-robot-data/robots/panda_description")
URDF = os.path.join(_VENV_DATA, "urdf/panda.urdf")
XML = os.path.join(_ROOT, "models/panda_scene.xml")

DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'


def env_kwargs_from_config(config: dict) -> dict:
    """Build env kwargs from train config (mimics train.py logic)."""
    hp = config.get("hyperparams", config)
    rw = config.get("reward_weights", {})
    return dict(
        n_joints=7, dt=0.02, episode_len=400,
        n_obstacles=5,
        d_critical=rw.get("d_critical", 0.05),
        d_safe=rw.get("d_safe", 0.06),
        alpha_relax=rw.get("alpha_relax", 0.1),
        w_track=rw.get("w_track", 3.0),
        w_obs=rw.get("w_obs", 5.0),
        w_manip=rw.get("w_manip", 0.05),
        w_energy=rw.get("w_energy", 0.001),
        w_collision=rw.get("w_collision", 100.0),
        reward_scale=20.0,
        success_bonus=rw.get("success_bonus", 50.0),
        use_collision_term=True,
    )


def evaluate_one_scene(env, agent, scene: dict, max_steps: int = 400) -> dict:
    """Run one episode on a specific scene. Returns detailed metrics."""
    from test import apply_scene  # reuse scene loader
    apply_scene(env, scene)

    # Run episode
    track_errors = []
    d_obs_values = []
    sigma_values = []
    dx_relax_norms = []
    success = False
    collision = False
    path_completion = 0.0

    # Reset env to clear persistent state (collision flag, step_count, etc.)
    env.reset()
    # Apply scene to override with desired configuration
    apply_scene(env, scene)
    # Reset again internally (apply_scene sets step_count=0 but we also need
    # _ever_collided=False which comes from _reset_state)
    env._reset_state()
    # Refresh obstacles in SDF
    obs_centers = [np.array(o[:3]) for o in scene["obstacles"]]
    obs_radii = [o[3] for o in scene["obstacles"]]
    env.sdf.set_static_obstacles(obs_centers, obs_radii)
    env._sync_obstacles_to_mujoco()

    # Step once to get the observation for this scene
    action_zero = np.zeros(env.act_dim)
    obs, _, _, _ = env.step(action_zero)

    for step in range(max_steps):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Decode action
        sigma = info.get("sigma", 0.0)
        dx_relax = info.get("dx_relax_mag", 0.0)

        x_ee, _ = env.kin.forward_kinematics(env.q)
        track_err = np.linalg.norm(x_ee - env.x_d)
        d_obs = info.get("d_obs", 1.0)

        track_errors.append(track_err)
        d_obs_values.append(d_obs)
        sigma_values.append(sigma)
        dx_relax_norms.append(dx_relax)
        path_completion = info.get("path_param", 0.0)

        if info.get("collision", False):
            collision = True
        if info.get("success", False):
            success = True
            break

    # Summarize
    min_d_obs = min(d_obs_values) if d_obs_values else 1.0
    return dict(
        scene_id=scene["scene_id"],
        success=success,
        collision=collision,
        path_completion=float(path_completion),
        steps=len(track_errors),
        mean_track_error=float(np.mean(track_errors)),
        max_track_error=float(np.max(track_errors)),
        min_d_obs=float(min_d_obs),
        mean_d_obs=float(np.mean(d_obs_values)) if d_obs_values else 1.0,
        mean_sigma=float(np.mean(sigma_values)) if sigma_values else 0.0,
        max_sigma=float(np.max(sigma_values)) if sigma_values else 0.0,
        mean_dx_relax=float(np.mean(dx_relax_norms)) if dx_relax_norms else 0.0,
        max_dx_relax=float(np.max(dx_relax_norms)) if dx_relax_norms else 0.0,
        obstacle=str(scene["obstacles"]),
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    p.add_argument("--val_json", type=str,
                   default=str(_ROOT + "/results/val_scenes_easy_sub10.json"))
    p.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"])
    p.add_argument("--config", type=str, default=None,
                   help="Path to config.json (auto from checkpoint dir)")
    args = p.parse_args()

    # Load config
    config_path = args.config or os.path.join(os.path.dirname(args.checkpoint), "config.json")
    config = json.load(open(config_path))

    # Load validation scenes
    val_scenes = json.load(open(args.val_json))
    print(f"Loaded {len(val_scenes)} validation scenes")

    # Create env
    dyn = ManipulatorDynamics(URDF)
    env = ManipulatorEnv(
        urdf_path=URDF, xml_path=XML, n_joints=7, dt=0.02,
        episode_len=400, n_obstacles=5, controller="rl",
    )
    env.reset()

    # Create agent
    if args.algo == "sac":
        kw = dict(state_dim=env.obs_dim, action_dim=env.act_dim, dynamics=dyn, device=DEVICE)
        agent = SACAgent(**kw)
    else:
        kw = dict(state_dim=env.obs_dim, action_dim=env.act_dim, dynamics=dyn, n_envs=1, device=DEVICE)
        agent = PPOAgent(**kw)
    # Use reset_critic=True for backward compat with old critic arch
    agent.load(args.checkpoint, reset_critic=True)
    agent.actor.eval()

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print()

    # Evaluate each scene
    results = []
    for scene in val_scenes:
        sid = scene["scene_id"]
        r = evaluate_one_scene(env, agent, scene)
        results.append(r)
        flag = ""
        if r["collision"]:
            flag = " 💥 COLLISION"
        elif r["success"]:
            flag = " ✅ SUCCESS"
        else:
            flag = " ❌ FAIL"
        print(f"scene {sid:3d}: success={r['success']} collision={r['collision']}"
              f"  path={r['path_completion']:.2f}  track_err={r['mean_track_error']:.4f}"
              f"  min_d_obs={r['min_d_obs']:.4f}  sigma={r['mean_sigma']:.3f}"
              f"  dx_relax={r['mean_dx_relax']:.3f}{flag}")

    # Summary
    success_rate = np.mean([r["success"] for r in results])
    collision_rate = np.mean([r["collision"] for r in results])
    print(f"\n{'='*60}")
    print(f"Summary: success_rate={success_rate:.1%}  collision_rate={collision_rate:.1%}")
    print(f"{'='*60}")

    # Print failure details
    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"\nFailures ({len(failures)} scenes):")
        for r in failures:
            print(f"  Scene {r['scene_id']}: min_d_obs={r['min_d_obs']:.4f}, "
                  f"collision={r['collision']}, path={r['path_completion']:.2f}")


if __name__ == "__main__":
    main()
