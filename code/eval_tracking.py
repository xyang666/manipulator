"""
eval_tracking.py
----------------
Load checkpoint and run one episode on scene 0, printing per-step tracking error.
"""
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.manipulator_env import ManipulatorEnv
from env.dynamics import ManipulatorDynamics
from agent.sac_agent import SACAgent

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENV_DATA = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                          "/share/example-robot-data/robots/panda_description")
URDF = os.path.join(_VENV_DATA, "urdf/panda.urdf")
XML = os.path.join(_ROOT, "models/panda_scene.xml")
CKPT_DIR = os.path.join(_ROOT, "checkpoints", "phase3_v10_scene0_fix")
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt_best.pt")
VAL_JSON = os.path.join(_ROOT, "results", "train_scenes.json")

with open(os.path.join(CKPT_DIR, "config.json")) as f:
    cfg = json.load(f)
cli = cfg["cli_args"]

env = ManipulatorEnv(
    urdf_path=URDF, xml_path=XML, n_joints=7, dt=0.02, episode_len=400,
    n_obstacles=5, controller="rl",
    obs_scene_embed=cli.get("obs_scene_embed", 5),
    obs_waypoint_steps=[int(s) for s in cli.get("obs_waypoint_steps", "10,20,50").split(",")],
    sigma_d_safe=cli.get("sigma_d_safe", 0.12),
    sigma_d_critical=cli.get("sigma_d_critical", 0.03),
    d_safe=cli.get("d_safe", 0.03),
    d_critical=cli.get("d_critical", 0.05),
)
env.reset()

# Load scene 0
from utils.validation import ValidationSet
val_set = ValidationSet(VAL_JSON)
scene = val_set.get_scene(0)
val_set.apply_scene_to_env(env, scene)
obs = env._get_obs()

# Load agent
dyn = ManipulatorDynamics(URDF)
n_critics = cli.get("n_critics", 2)
hidden_dims = tuple(cli.get("hidden_dims", [256, 256]))
agent = SACAgent(
    state_dim=env.obs_dim, action_dim=env.act_dim, dynamics=dyn,
    hidden_dims=hidden_dims, n_critics=n_critics,
    device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
)
meta = agent.load(CKPT_PATH)
agent.actor.eval()

# Run one episode
print(f"{'Step':>6} {'TrackErr':>9} {'d_obs':>8} {'sigma':>7} {'coll':>5} {'d_ee':>8}")
print("-" * 55)

tracking_errors = []
d_obs_list = []
all_x_ee = []
done = False
steps = 0

while not done and steps < 400:
    action = agent.select_action(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    x_ee, _ = env.kin.forward_kinematics(env.q)
    track_err = np.linalg.norm(x_ee - env.x_d)
    d_obs = info.get("d_obs", 0.0)
    d_ee = np.linalg.norm(x_ee - env.x_goal)

    tracking_errors.append(track_err)
    d_obs_list.append(d_obs)
    all_x_ee.append(x_ee.copy())

    if steps % 20 == 0:
        coll = "Y" if info.get("collision") else "N"
        sigma = env._last_sigma  # sigma stored in env, not in info
        delta_x = action[:3]
        z = action[3:]
        print(f"{steps:>6d} {track_err:>9.4f} {d_obs:>8.4f} {sigma:>7.3f} {coll:>5s} {d_ee:>8.4f} | dx_rl=({delta_x[0]:>.3f},{delta_x[1]:>.3f},{delta_x[2]:>.3f}) z=({z[0]:>.3f},{z[1]:>.3f},{z[2]:>.3f},{z[3]:>.3f})")

    steps += 1

print("-" * 55)
print(f"Steps: {steps}")
print(f"Mean tracking error: {np.mean(tracking_errors):.4f} m")
print(f"Max tracking error:  {np.max(tracking_errors):.4f} m")
print(f"Mean d_obs:          {np.mean(d_obs_list):.4f} m")
print(f"Min d_obs:           {np.min(d_obs_list):.4f} m")
print(f"Final distance:      {np.linalg.norm(env.x_d - env.x_goal):.4f} m")

# Also print final segment stats
print()
print("--- Last 100 steps ---")
print(f"{'Step':>6} {'TrackErr':>9} {'d_obs':>8} {'d_ee':>8}")
for i in range(max(0, steps-100), steps):
    print(f"{i:>6d} {tracking_errors[i]:>9.4f} {d_obs_list[i]:>8.4f} {np.linalg.norm(all_x_ee[i] - env.x_goal):>8.4f}")
