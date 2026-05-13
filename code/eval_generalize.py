"""
eval_generalize.py
------------------
Evaluate a checkpoint on multiple validation scenes to test generalization.
Runs independently so training can continue in parallel.
"""
import json, sys, os, numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.manipulator_env import ManipulatorEnv
from env.dynamics import ManipulatorDynamics
from agent.sac_agent import SACAgent
from utils.validation import ValidationSet, evaluate_on_validation_set

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_VENV_DATA = os.path.join(_HERE, ".venv/lib/python3.12/site-packages/cmeel.prefix"
                          "/share/example-robot-data/robots/panda_description")
URDF = os.path.join(_VENV_DATA, "urdf/panda.urdf")
XML = os.path.join(_ROOT, "models/panda_scene.xml")

CKPT_DIR = os.path.join(_ROOT, "checkpoints", "phase3_v12_scene0_wact02")
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt_best.pt")
VAL_JSON = os.path.join(_ROOT, "results", "train_scenes.json")

# Load config
with open(os.path.join(CKPT_DIR, "config.json")) as f:
    cli = json.load(f)["cli_args"]

# Create env
env = ManipulatorEnv(
    urdf_path=URDF, xml_path=XML, n_joints=7, dt=0.02, episode_len=400,
    n_obstacles=5, controller="rl",
    obs_k=cli.get("obs_k", 5),
    obs_scene_embed=cli.get("obs_scene_embed", 5),
    obs_waypoint_steps=[int(s) for s in cli.get("obs_waypoint_steps", "10,20,50").split(",")],
    sigma_d_safe=cli.get("sigma_d_safe", 0.12),
    sigma_d_critical=cli.get("sigma_d_critical", 0.03),
    d_safe=cli.get("d_safe", 0.03),
    d_critical=cli.get("d_critical", 0.05),
)
env.reset()

# Load agent
dyn = ManipulatorDynamics(URDF)
agent = SACAgent(
    state_dim=env.obs_dim, action_dim=env.act_dim, dynamics=dyn,
    hidden_dims=tuple(cli.get("hidden_dims", [256, 256])),
    n_critics=cli.get("n_critics", 2),
    device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
)
agent.load(CKPT_PATH)
agent.actor.eval()

# Load validation set (all 80 scenes)
val_set = ValidationSet(VAL_JSON)
print(f"Evaluating on all {len(val_set.scenes)} scenes...\n")

# Run evaluation on ALL scenes
results = evaluate_on_validation_set(
    agent, env, val_set,
    num_scenes=None,  # all scenes
    max_steps=400,
)

# Print summary
print("=" * 65)
print("GENERALIZATION RESULTS (MuJoCo-based)")
print("=" * 65)
print(f"  Success Rate:       {results['success_rate']*100:.1f}%")
print(f"  Collision Rate:     {results['collision_rate']*100:.1f}%")
print(f"  Avg Reward:         {results['avg_reward']:.1f}")
print(f"  Avg Tracking Error: {results['avg_tracking_error']:.4f}m")
print(f"  Avg Min Distance:   {results['avg_min_distance']:.4f}m (capsule SDF)")
print()

# Per-scene breakdown: show successes and failures
scene_results = results["scene_results"]
successes = [s for s in scene_results if s["success"]]
failures = [s for s in scene_results if not s["success"]]

print(f"  Successful: {len(successes)}/{len(scene_results)}")
print(f"  Failed:     {len(failures)}/{len(scene_results)}")
print()

if failures:
    print("  --- Failure Details ---")
    print(f"  {'Scene':>5} {'Reward':>8} {'Steps':>5} {'MinDist':>8} {'TrackErr':>8}")
    for s in sorted(failures, key=lambda x: x["scene_id"]):
        print(f"  {s['scene_id']:>5d} {s['reward']:>8.1f} {s['steps']:>5d} {s['min_distance']:>8.4f} {s['tracking_error']:>8.4f}")

# Group by difficulty: scenes where obstacles are close to path
print()
print("  --- Scene Difficulty Analysis ---")
d_obs_all = [s["min_distance"] for s in scene_results]
print(f"  Min d_obs across all scenes:")
print(f"    Mean:     {np.mean(d_obs_all):.4f}m")
print(f"    Median:   {np.median(d_obs_all):.4f}m")
print(f"    Min:      {np.min(d_obs_all):.4f}m")
print(f"    < 0.0:    {sum(1 for d in d_obs_all if d < 0.0)}/{len(d_obs_all)} scenes")

# Save results
out_path = os.path.join(CKPT_DIR, "generalization_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\n  Results saved to {out_path}")
print("=" * 65)
