"""Generate the complete phase-one training/evaluation manifest without running jobs."""

import argparse
import json
from pathlib import Path

from experiment_config import (ALGORITHM, EVALUATION, PHASE1_METHODS, PHASE1_SCENARIOS,
                               phase1_defaults)


LEARNED_METHODS = {
    "sac_joint", "sac_residual", "ours_no_physics", "ours_physics",
    "ours_shielded", "ours_hybrid", "ours_full",
}


def build_manifest(checkpoint_root: Path, result_root: Path,
                   scene_json: Path = Path("../results/ewalker_scenes/curriculum/train.json"),
                   val_json: Path = Path("../results/ewalker_scenes/curriculum/validation.json")) -> dict:
    training_jobs = []
    learned = {
        "sac_joint": ["--agent_type", "joint", "--lambda_dyn", "0", "--no_safety_critic"],
        "sac_residual": ["--agent_type", "residual", "--lambda_dyn", "0", "--no_safety_critic"],
        "ours_no_physics": ["--agent_type", "structured", "--lambda_dyn", "0", "--no_safety_critic", "--disable_gate"],
        "ours_physics": ["--agent_type", "structured", "--lambda_dyn", "1", "--no_safety_critic", "--disable_gate"],
        "ours_shielded": ["--agent_type", "structured", "--lambda_dyn", "0",
                           "--no_safety_critic", "--disable_gate", "--use_cbf",
                           "--cbf_self_distance", "0.02",
                           "--cbf_multi_self_constraints"],
        "ours_hybrid": ["--agent_type", "structured", "--lambda_dyn", "0",
                        "--no_safety_critic", "--disable_gate",
                        "--gradient_prior_scale", "0.3",
                        "--gradient_prior_smoothing", "0.8",
                        "--learned_residual_scale", "0.2",
                        "--confined_deterministic_prior",
                        "--free_deterministic_cbf",
                        "--generalization_deterministic_cbf", "--use_cbf",
                        "--cbf_self_distance", "0.02",
                        "--cbf_multi_self_constraints"],
        "ours_full": ["--agent_type", "structured", "--lambda_dyn", "1"],
    }
    protocols = {
        "curriculum": (scene_json, val_json, 40),
        "generalization": (
            Path("../results/ewalker_scenes/generalization/train.json"),
            Path("../results/ewalker_scenes/generalization/validation.json"), 20,
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
                    "--task_scale", str(ALGORITHM.task_scale),
                    "--val_every_steps", str(ALGORITHM.validation_interval_steps),
                    "--val_scenes", str(val_count), *flags,
                ]
                training_jobs.append({"method": method, "protocol": protocol,
                                      "seed": seed, "run_name": run_name,
                                      "command": command})
    # A bounded diagnostic run isolates the task that curriculum validation
    # showed to be under-learned. It is not part of the 50-job paper matrix.
    diagnostic_jobs = [{
        "method": "ours_full",
        "protocol": "whole_body_diagnostic",
        "seed": EVALUATION.train_seeds[0],
        "run_name": f"ours_full/whole_body_diagnostic/seed_{EVALUATION.train_seeds[0]}",
        "command": [
            "python", "train.py", "--seed", str(EVALUATION.train_seeds[0]),
            "--run_name",
            f"ours_full/whole_body_diagnostic/seed_{EVALUATION.train_seeds[0]}",
            "--save_path", str(checkpoint_root),
            "--scene_json", "../results/ewalker_scenes/whole_body/train.json",
            "--val_json", "../results/ewalker_scenes/whole_body/validation.json",
            "--n_envs", str(ALGORITHM.parallel_envs),
            "--grad_steps", str(ALGORITHM.gradient_steps),
            "--task_scale", str(ALGORITHM.task_scale),
            "--val_every_steps", str(ALGORITHM.validation_interval_steps),
            "--val_scenes", "20", "--agent_type", "structured",
            "--lambda_dyn", "1",
        ],
    }]
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
                scene_path = Path("../results/ewalker_scenes") / scenario / "test.json"
                command.extend(["--scene-json", str(scene_path)])
                if method in LEARNED_METHODS:
                    command.extend(["--checkpoint", str(checkpoint)])
                jobs.append({"method": method, "scenario": scenario, "seed": seed,
                             "checkpoint": str(checkpoint) if "--checkpoint" in command else None,
                             "output": str(output), "command": command})
    return {"schema_version": 2, "defaults": phase1_defaults(),
            "training_jobs": training_jobs,
            "diagnostic_jobs": diagnostic_jobs,
            "evaluation_jobs": jobs}


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
