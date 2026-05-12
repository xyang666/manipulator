"""
eval_compare.py
---------------
Compare action magnitudes between v10 (no w_action) and v11 (w_action=0.05).
"""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from env.manipulator_env import ManipulatorEnv
from env.dynamics import ManipulatorDynamics
from agent.sac_agent import SACAgent
from utils.validation import ValidationSet

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENV_DATA = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                          "/share/example-robot-data/robots/panda_description")
URDF = os.path.join(_VENV_DATA, "urdf/panda.urdf")
XML = os.path.join(_ROOT, "models/panda_scene.xml")
VAL_JSON = os.path.join(_ROOT, "results", "train_scenes.json")

checkpoints = {
    "v10 (no w_action)": "/root/manipulator/checkpoints/phase3_v10_scene0_fix/ckpt_best.pt",
    "v11 (w_action=0.05)": "/root/manipulator/checkpoints/phase3_v11_scene0_wact005/ckpt_ep01000.pt",
}

def eval_ckpt(name, ckpt_path):
    ckpt_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    cli = cfg["cli_args"]

    env = ManipulatorEnv(
        urdf_path=URDF, xml_path=XML, n_joints=7, dt=0.02, episode_len=400,
        n_obstacles=5, controller="rl",
        obs_k=cli.get("obs_k", 5),
        obs_scene_embed=cli.get("obs_scene_embed", 5),
        obs_waypoint_steps=[int(s) for s in cli.get("obs_waypoint_steps", "10,20,50").split(",")],
        sigma_d_safe=cli.get("sigma_d_safe", 0.12),
        sigma_d_critical=cli.get("sigma_d_critical", 0.03),
        action_smooth=cli.get("action_smooth", 0.0),
        d_safe=cli.get("d_safe", 0.03),
        d_critical=cli.get("d_critical", 0.05),
    )
    env.reset()
    val_set = ValidationSet(VAL_JSON)
    scene = val_set.get_scene(0)
    val_set.apply_scene_to_env(env, scene)

    n_critics = cli.get("n_critics", 2)
    hidden_dims = tuple(cli.get("hidden_dims", [256, 256]))
    dyn = ManipulatorDynamics(URDF)
    agent = SACAgent(state_dim=env.obs_dim, action_dim=env.act_dim, dynamics=dyn,
                     hidden_dims=hidden_dims, n_critics=n_critics,
                     device='cuda' if __import__('torch').cuda.is_available() else 'cpu')
    agent.load(ckpt_path)
    agent.actor.eval()

    obs = env._get_obs()
    dx_rl_list = []
    z_list = []
    track_errs = []
    d_obs_list = []
    sigma_list = []

    done = False
    steps = 0
    while not done and steps < 400:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        sigma_list.append(float(env._last_sigma))
        dx_rl_list.append(action[:3].copy())
        z_list.append(action[3:].copy())
        x_ee, _ = env.kin.forward_kinematics(env.q)
        track_errs.append(np.linalg.norm(x_ee - env.x_d))
        d_obs_list.append(info.get("d_obs", 0.0))
        steps += 1

    dx_rl = np.array(dx_rl_list)
    z = np.array(z_list)
    dx_mag = np.sqrt(np.sum(dx_rl**2, axis=1))
    z_mag = np.sqrt(np.sum(z**2, axis=1))

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Steps:              {steps}")
    print(f"  Mean track error:   {np.mean(track_errs):.4f} m")
    print(f"  Final goal dist:    {np.linalg.norm(env.x_d - env.x_goal):.4f} m")
    print(f"  Min d_obs:          {np.min(d_obs_list):.4f} m")
    print(f"  Mean d_obs:         {np.mean(d_obs_list):.4f} m")
    print(f"")
    print(f"  Δẋ_RL stats:")
    print(f"    Mean magnitude:   {np.mean(dx_mag):.3f} m/s")
    print(f"    Max magnitude:    {np.max(dx_mag):.3f} m/s")
    print(f"    Std magnitude:    {np.std(dx_mag):.3f} m/s")
    print(f"    Mean |dx|/step:   {np.mean(np.abs(dx_rl), axis=0)}")
    print(f"  z (nullspace) stats:")
    print(f"    Mean magnitude:   {np.mean(z_mag):.4f}")
    print(f"    Max magnitude:    {np.max(z_mag):.4f}")
    print(f"  Sigma stats:")
    print(f"    Mean:             {np.mean(sigma_list):.3f}")
    print(f"    Max:              {np.max(sigma_list):.3f}")

    # Print per-20-step detail for middle section (obstacle zone)
    print(f"\n  Mid-episode detail (obstacle zone):")
    print(f"  {'Step':>5} {'TrackErr':>8} {'d_obs':>7} {'sigma':>6} {'|Δẋ_RL|':>8} {'|z|':>7}")
    for i in range(60, min(300, steps), 20):
        print(f"  {i:>5d} {track_errs[i]:>8.4f} {d_obs_list[i]:>7.4f} {sigma_list[i]:>6.3f} {dx_mag[i]:>8.3f} {z_mag[i]:>7.4f}")

    return np.mean(dx_mag), np.max(dx_mag), np.mean(sigma_list)

print("Comparing action magnitudes: v10 (no penalty) vs v11 (w_action=0.05)")
results = {}
for name, ckpt in checkpoints.items():
    results[name] = eval_ckpt(name, ckpt)

print(f"\n{'='*55}")
print(f"  Summary Comparison")
print(f"{'='*55}")
for name, (mean_dx, max_dx, mean_sig) in results.items():
    print(f"  {name:25s}: mean|Δẋ_RL|={mean_dx:.3f} m/s, max={max_dx:.3f} m/s, mean σ={mean_sig:.3f}")
