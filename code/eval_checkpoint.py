"""
eval_checkpoint.py
------------------
Standalone validation script that loads a checkpoint and evaluates it
using the latest validation.py (MuJoCo-based success/collision metrics).
"""
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.manipulator_env import ManipulatorEnv
from env.dynamics import ManipulatorDynamics
from agent.sac_agent import SACAgent
from utils.validation import ValidationSet, evaluate_on_validation_set

# Paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENV_DATA = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                          "/share/example-robot-data/robots/panda_description")
URDF = os.path.join(_VENV_DATA, "urdf/panda.urdf")
XML = os.path.join(_ROOT, "models/panda_scene.xml")

CKPT_DIR = os.path.join(_ROOT, "checkpoints", "phase3_v10_scene0_fix")
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt_best.pt")
VAL_JSON = os.path.join(_ROOT, "results", "train_scenes.json")

# Load config
with open(os.path.join(CKPT_DIR, "config.json")) as f:
    cfg = json.load(f)
cli = cfg["cli_args"]

print("=" * 60)
print("Standalone Evaluation")
print("=" * 60)
print(f"Checkpoint: {CKPT_PATH}")
print(f"Config: n_critics={cli.get('n_critics', 2)}, obs_dim={cfg['hyperparams']['state_dim']}")
print()

# Create environment
env = ManipulatorEnv(
    urdf_path=URDF,
    xml_path=XML,
    n_joints=7,
    dt=0.02,
    episode_len=400,
    n_obstacles=5,
    controller="rl",
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
print(f"Env obs_dim={env.obs_dim}, act_dim={env.act_dim}")

# Create dynamics
dyn = ManipulatorDynamics(URDF)

# Create agent with correct n_critics
n_critics = cli.get("n_critics", 2)
hidden_dims = tuple(cli.get("hidden_dims", [256, 256]))
agent = SACAgent(
    state_dim=env.obs_dim,
    action_dim=env.act_dim,
    dynamics=dyn,
    hidden_dims=hidden_dims,
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    n_critics=n_critics,
)
meta = agent.load(CKPT_PATH)
agent.actor.eval()
print(f"Agent loaded (n_critics={n_critics}). Metadata: {meta}")
print()

# Load validation set
val_set = ValidationSet(VAL_JSON)
print()

# Evaluate on scene 0 only
results = evaluate_on_validation_set(
    agent, env, val_set,
    num_scenes=1,
    max_steps=400,
)

print()
print("=" * 60)
print("Scene 0 Validation Results (MuJoCo-based)")
print("=" * 60)
print(f"  Success Rate:      {results['success_rate']*100:.1f}%")
print(f"  Avg Reward:        {results['avg_reward']:.3f}")
print(f"  Avg Tracking Error: {results['avg_tracking_error']:.4f}m")
print(f"  Avg Min Distance:   {results['avg_min_distance']:.4f}m (capsule SDF)")
print(f"  Collision Rate:     {results['collision_rate']*100:.1f}%")
print()
for sr in results["scene_results"]:
    print(f"  Scene {sr['scene_id']}: success={sr['success']}, "
          f"reward={sr['reward']:.1f}, steps={sr['steps']}, "
          f"min_dist={sr['min_distance']:.4f}m")
print("=" * 60)
