"""Generate the complete phase-one training/evaluation manifest without running jobs."""

import argparse
import json
from pathlib import Path

from experiment_config import (ALGORITHM, EVALUATION, PHASE1_METHODS, PHASE1_SCENARIOS,
                               phase1_defaults)


def build_manifest(checkpoint_root: Path, result_root: Path,
                   scene_json: Path = Path("../results/paper_scenes/curriculum/train.json"),
                   val_json: Path = Path("../results/paper_scenes/curriculum/validation.json")) -> dict:
    training_jobs = []
    learned = {
        "sac_joint": ["--agent_type", "joint", "--lambda_dyn", "0", "--no_safety_critic"],
        "sac_residual": ["--agent_type", "residual", "--lambda_dyn", "0", "--no_safety_critic"],
        "ours_no_physics": ["--agent_type", "structured", "--lambda_dyn", "0", "--no_safety_critic", "--disable_gate"],
        "ours_physics": ["--agent_type", "structured", "--lambda_dyn", "1", "--no_safety_critic", "--disable_gate"],
        "ours_full": ["--agent_type", "structured", "--lambda_dyn", "1"],
    }
    protocols = {
        "curriculum": (scene_json, val_json, 40),
        "generalization": (
            Path("../results/paper_scenes/generalization/train.json"),
            Path("../results/paper_scenes/generalization/validation.json"), 20,
        ),
    }
    for method, flags in learned.items():
        for protocol, (train_scenes, validation_scenes, val_count) in protocols.items():
            for seed in EVALUATION.train_seeds:
                run_name = f"{method}/{protocol}/seed_{seed}"
                command = [
                    "python", "train.py", "--seed", str(seed), "--run_name", run_name,
                    "--save_path", str(checkpoint_root),
                    "--scene_json", str(train_scenes), "--val_json", str(validation_scenes),
                    "--n_envs", str(ALGORITHM.parallel_envs),
                    "--grad_steps", str(ALGORITHM.gradient_steps),
                    "--val_every_steps", str(ALGORITHM.validation_interval_steps),
                    "--val_scenes", str(val_count), *flags,
                ]
                training_jobs.append({"method": method, "protocol": protocol,
                                      "seed": seed, "run_name": run_name,
                                      "command": command})
    jobs = []
    for method in PHASE1_METHODS:
        for scenario in PHASE1_SCENARIOS:
            for seed in EVALUATION.train_seeds:
                output = result_root / method / scenario / f"seed_{seed}.jsonl"
                protocol = "generalization" if scenario == "generalization" else "curriculum"
                checkpoint = (checkpoint_root / method / protocol /
                              f"seed_{seed}" / "ckpt_best.pt")
                command = [
                    "python", "-m", "experiments.runner", "--method", method,
                    "--scenario", scenario, "--seed", str(seed), "--episodes",
                    str(EVALUATION.episodes_per_seed), "--output", str(output),
                ]
                scene_path = Path("../results/paper_scenes") / scenario / "test.json"
                command.extend(["--scene-json", str(scene_path)])
                if method not in ("pd", "gradient_projection", "cbf_qp"):
                    command.extend(["--checkpoint", str(checkpoint)])
                jobs.append({"method": method, "scenario": scenario, "seed": seed,
                             "checkpoint": str(checkpoint) if "--checkpoint" in command else None,
                             "output": str(output), "command": command})
    return {"schema_version": 1, "defaults": phase1_defaults(),
            "training_jobs": training_jobs, "evaluation_jobs": jobs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("/root/autodl-tmp/manipulator/checkpoints/phase1"))
    parser.add_argument("--result-root", type=Path, default=Path("results/phase1"))
    parser.add_argument("--output", type=Path, default=Path("results/phase1/manifest.json"))
    args = parser.parse_args()
    manifest = build_manifest(args.checkpoint_root, args.result_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
