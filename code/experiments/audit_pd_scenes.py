"""Audit certified paper scenes with the nominal zero-residual PD controller."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from env.manipulator_env import ManipulatorEnv
from experiment_config import ENVIRONMENT
from utils.validation import ValidationSet


_ENV: ManipulatorEnv | None = None
_APPLIER: ValidationSet | None = None


def _init_worker(urdf: str, xml: str, max_obstacles: int) -> None:
    global _ENV, _APPLIER
    _ENV = ManipulatorEnv(
        urdf_path=urdf,
        xml_path=xml,
        dt=ENVIRONMENT.dt,
        episode_len=ENVIRONMENT.episode_len,
        trajectory_steps=ENVIRONMENT.trajectory_steps,
        n_obstacles=max_obstacles,
        w_track=ENVIRONMENT.w_track,
        w_obs=ENVIRONMENT.w_obs,
        w_manip=ENVIRONMENT.w_manip,
        w_energy=ENVIRONMENT.w_energy,
        w_collision=ENVIRONMENT.w_collision,
        w_action=ENVIRONMENT.w_action,
        d_safe=ENVIRONMENT.d_safe,
        success_bonus=ENVIRONMENT.success_bonus,
        use_trajectory_generator=False,
    )
    _APPLIER = ValidationSet.__new__(ValidationSet)


def _audit_one(item: tuple[str, dict]) -> dict:
    source, scene = item
    assert _ENV is not None and _APPLIER is not None
    _APPLIER.apply_scene_to_env(_ENV, scene)
    collided = False
    min_clearance = float("inf")
    info: dict = {}
    for step in range(_ENV.episode_len):
        obs, _, done, info = _ENV.step(np.zeros(_ENV.act_dim))
        del obs
        collided = collided or bool(info["collision"])
        min_clearance = min(min_clearance, float(info["d_obs"]))
        if done:
            break
    x_ee, _ = _ENV.kin.forward_kinematics(_ENV.q)
    success = bool(info.get("success", False)) and not collided
    oracle_valid = bool(scene.get("oracle", scene.get("certificate", True)))
    if not oracle_valid:
        category = "invalid"
    elif not success or collided:
        category = "hard"
    elif min_clearance < ENVIRONMENT.d_safe:
        category = "medium"
    else:
        category = "easy"
    return {
        "source": source,
        "scene_id": scene["scene_id"],
        "category": category,
        "success": success,
        "collision": collided,
        "min_clearance_m": min_clearance,
        "final_path_param": float(_ENV.path_param),
        "final_distance_m": float(np.linalg.norm(x_ee - _ENV.x_goal)),
        "termination_reason": info.get("termination_reason"),
        "steps": step + 1,
    }


def _scene_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.glob("*/*.json")):
        if path.parent.name == "curriculum" or path.name == "manifest.json":
            continue
        files.append(path)
    return files


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path,
                        default=repository / "results/paper_scenes")
    parser.add_argument("--output", type=Path,
                        default=repository / "results/paper_scenes/pd_audit.json")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--urdf", default=str(
        repository / "panda_description/urdf/panda.urdf"))
    parser.add_argument("--xml", default=str(repository / "models/panda_scene.xml"))
    args = parser.parse_args()

    jobs: list[tuple[str, dict]] = []
    for path in _scene_files(args.scene_root):
        relative = str(path.relative_to(args.scene_root))
        for scene in json.loads(path.read_text(encoding="utf-8")):
            jobs.append((relative, scene))
    if not jobs:
        raise ValueError(f"no base scene files found below {args.scene_root}")

    max_obstacles = max(len(scene.get("obstacles", [])) for _, scene in jobs)
    # Fork keeps the large MuJoCo/Pinocchio/PyTorch shared libraries
    # copy-on-write.  A spawn pool duplicates them in every worker and exceeds
    # the 2 GiB memory cgroup used by the training server.
    context = mp.get_context("fork")
    with context.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.urdf, args.xml, max_obstacles),
    ) as pool:
        rows = list(pool.imap_unordered(_audit_one, jobs, chunksize=1))

    rows.sort(key=lambda row: (row["source"], str(row["scene_id"])))
    by_source: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_source[row["source"]][row["category"]] += 1
    payload = {
        "schema_version": 1,
        "safety_margin_m": ENVIRONMENT.d_safe,
        "total": len(rows),
        "counts": dict(Counter(row["category"] for row in rows)),
        "by_source": {key: dict(value) for key, value in sorted(by_source.items())},
        "scenes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in ("total", "counts", "by_source")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
