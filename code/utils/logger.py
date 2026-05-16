"""
logger.py
---------
Training data logger for physics-informed SAC.

Writes one CSV row per environment step and manages timestamped
checkpoint directories for paper-quality experiment logging.
"""

import csv
import math
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Centralised reward component registry
# Add a new component once here — prints, CSV columns, and accumulators
# all pick it up automatically.
#   (info_key,   csv_column,    print_label, width)
# ---------------------------------------------------------------------------
REWARD_COMPONENTS = [
    ("r_track",    "r_track",     "r_trk",   9),
    ("r_obs",      "r_obs",       "r_obs",   9),
    ("r_null",     "r_null",      "r_null",  8),
    ("r_apf",      "r_apf",       "r_apf",   8),
    ("r_manip",    "r_manip",     "r_manip", 8),
    ("r_energy",   "r_energy",    "r_en",    7),
    ("r_collision","r_collision", "r_coll",  8),
    ("r_action",   "r_action",    "r_act",   8),
]

REWARD_CSV_COLS = [csv_col for _, csv_col, _, _ in REWARD_COMPONENTS]
REWARD_HEADER    = "  ".join(f"{label:^{width}}" for _, _, label, width in REWARD_COMPONENTS)
REWARD_FORMAT    = "  ".join(f"{{r_{i}:>{w}.4f}}" for i, (_, _, _, w) in enumerate(REWARD_COMPONENTS))

CSV_COLUMNS = [
    "global_step", "episode", "ep_step",
    "reward", "d_obs", "w",
] + REWARD_CSV_COLS + [
    "collision_penalty",
    "critic_loss", "actor_rl_loss", "physics_loss", "actor_loss", "alpha",
    "success", "ever_collided",
]


def reward_accumulators():
    """Return {csv_col: []} for per-episode reward tracking."""
    return {csv_col: [] for _, csv_col, _, _ in REWARD_COMPONENTS}


def accumulate_rewards(info: dict, acc: dict) -> None:
    """Accumulate reward components from an info dict into accumulator dicts."""
    for info_key, csv_col, _, _ in REWARD_COMPONENTS:
        acc[csv_col].append(info.get(info_key, 0.0))


def avg_rewards(acc: dict) -> dict:
    """Compute per-component averages from accumulator dicts."""
    return {
        col: (sum(vals) / len(vals)) if vals else 0.0
        for col, vals in acc.items()
    }


def reward_values_from_info(info: dict) -> dict:
    """Extract reward component values from info dict, keyed by csv_column."""
    return {csv_col: info.get(info_key, "") for info_key, csv_col, _, _ in REWARD_COMPONENTS}


def reward_print_values(avg_dict: dict) -> dict:
    """
    Build dict suitable for str.format(REWARD_FORMAT).
    avg_dict keys are csv_column names.
    """
    return {f"r_{i}": avg_dict.get(csv_col, 0.0) for i, (_, csv_col, _, _) in enumerate(REWARD_COMPONENTS)}


class TrainingLogger:
    """
    Logs per-step metrics to CSV and manages checkpoint paths.

    Usage
    -----
    logger = TrainingLogger(run_dir, hyperparams)

    # inside env step loop:
    logger.log_step(global_step, episode, ep_step, reward, info)
    logger.log_update(losses)   # call after agent.update()

    # after episode ends:
    ep_summary = logger.end_episode(episode, total_steps)

    # get checkpoint path:
    path = logger.checkpoint_path("best")

    logger.close()
    """

    def __init__(self, run_dir: str, hyperparams: dict):
        self.run_dir = run_dir
        self.hyperparams = hyperparams
        self.best_reward = -math.inf

        os.makedirs(run_dir, exist_ok=True)

        self.csv_path = os.path.join(run_dir, "training_log.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
        self._csv_writer.writeheader()

        # Validation log
        self.val_csv_path = os.path.join(run_dir, "validation_log.csv")
        self._val_csv_file = open(self.val_csv_path, "w", newline="")
        self._val_csv_writer = csv.DictWriter(
            self._val_csv_file,
            fieldnames=["episode", "success_rate", "avg_reward", "avg_tracking_error",
                       "avg_min_distance", "collision_rate"]
        )
        self._val_csv_writer.writeheader()

        self._last_losses: dict | None = None
        self._ep_losses: list[dict] = []
        self._ep_rewards: list[float] = []
        self._ep_d_obs: list[float] = []
        self._ep_w: list[float] = []
        self._ep_success: list[bool] = []

    # ---- reward helpers (class methods for convenience) ----
    @staticmethod
    def reward_accumulators():
        return reward_accumulators()

    @staticmethod
    def accumulate_rewards(info, acc):
        accumulate_rewards(info, acc)

    @staticmethod
    def avg_rewards(acc):
        return avg_rewards(acc)

    @staticmethod
    def reward_print_values(avg_dict):
        return reward_print_values(avg_dict)

    # ---- step-level logging ----
    def log_step(self, step: int, episode: int, ep_step: int,
                 reward: float, info: dict) -> None:
        """Call once per env step, immediately after env.step()."""
        self._ep_rewards.append(reward)
        self._ep_d_obs.append(info.get("d_obs", float("nan")))
        self._ep_w.append(info.get("w", float("nan")))
        self._ep_success.append(info.get("success", False))

        row: dict = {
            "global_step":      step,
            "episode":          episode,
            "ep_step":          ep_step,
            "reward":           reward,
            "d_obs":            info.get("d_obs", ""),
            "w":                info.get("w", ""),
            **reward_values_from_info(info),
            "collision_penalty": info.get("collision_penalty", ""),
            "success":          int(info.get("success", False)),
        }

        self._fill_losses(row)
        self._csv_writer.writerow(row)

    def _fill_losses(self, row: dict) -> None:
        if self._last_losses is not None:
            row.update({
                "critic_loss":    self._last_losses.get("critic_loss", ""),
                "actor_rl_loss":  self._last_losses.get("actor_rl_loss", ""),
                "physics_loss":   self._last_losses.get("physics_loss", ""),
                "actor_loss":     self._last_losses.get("actor_loss", ""),
                "alpha":          self._last_losses.get("alpha", ""),
            })
        else:
            row.update({k: "" for k in
                        ["critic_loss", "actor_rl_loss", "physics_loss", "actor_loss", "alpha"]})

    def log_update(self, losses: dict) -> None:
        """Call once per agent.update(), passing the returned losses dict."""
        self._last_losses = losses
        self._ep_losses.append(losses)

    def end_episode(self, episode: int, total_steps: int) -> dict:
        """
        Compute episode summary, flush CSV, reset accumulators.

        Returns
        -------
        dict with keys: total_reward, episode_length, avg_critic_loss,
        avg_actor_rl_loss, avg_physics_loss, avg_actor_loss, avg_alpha,
        min_d_obs, avg_manipulability, success
        """
        def _mean(lst, key):
            vals = [d[key] for d in lst if key in d]
            return sum(vals) / len(vals) if vals else float("nan")

        min_d_obs = min(self._ep_d_obs) if self._ep_d_obs else float("nan")

        summary = {
            "total_reward":      sum(self._ep_rewards),
            "episode_length":    len(self._ep_rewards),
            "avg_critic_loss":   _mean(self._ep_losses, "critic_loss"),
            "avg_actor_rl_loss": _mean(self._ep_losses, "actor_rl_loss"),
            "avg_physics_loss":  _mean(self._ep_losses, "physics_loss"),
            "avg_actor_loss":    _mean(self._ep_losses, "actor_loss"),
            "avg_alpha":         _mean(self._ep_losses, "alpha"),
            "min_d_obs":         min_d_obs,
            "avg_manipulability": (sum(self._ep_w) / len(self._ep_w)
                                   if self._ep_w else float("nan")),
            "success":           any(self._ep_success),
        }

        self._csv_file.flush()

        self._ep_losses.clear()
        self._ep_rewards.clear()
        self._ep_d_obs.clear()
        self._ep_w.clear()
        self._ep_success.clear()

        return summary

    def checkpoint_path(self, tag: str) -> str:
        """Return full path for a checkpoint file, e.g. tag='best' or 'ep00050'."""
        return os.path.join(self.run_dir, f"ckpt_{tag}.pt")

    def log_validation(self, episode: int, val_results: dict) -> None:
        """Log validation results to separate CSV."""
        row = {
            "episode": episode,
            "success_rate": val_results["success_rate"],
            "avg_reward": val_results["avg_reward"],
            "avg_tracking_error": val_results["avg_tracking_error"],
            "avg_min_distance": val_results["avg_min_distance"],
            "collision_rate": val_results["collision_rate"],
        }
        self._val_csv_writer.writerow(row)
        self._val_csv_file.flush()

    def log_episode_summary(self, step: int, episode: int, total_reward: float,
                             min_d_obs: float, avg_actor_loss: float,
                             avg_physics_loss: float, ep_step: int = None,
                             alpha: float = None,
                             avg_critic_loss: float = None,
                             avg_actor_total_loss: float = None,
                             avg_w: float = None,
                             success: bool = None,
                             ever_collided: bool = None,
                             **avg_reward_kwargs) -> None:
        """Write a single episode-summary row to the training CSV.

        avg_reward_kwargs keys must match csv_column names in REWARD_COMPONENTS.
        """
        row = {
            "global_step":      step,
            "episode":          episode,
            "reward":           total_reward,
            "d_obs":            min_d_obs,
            "actor_rl_loss":    avg_actor_loss,
            "physics_loss":     avg_physics_loss,
        }
        if ep_step is not None:
            row["ep_step"] = ep_step
        if avg_critic_loss is not None:
            row["critic_loss"] = avg_critic_loss
        if avg_actor_total_loss is not None:
            row["actor_loss"] = avg_actor_total_loss
        if avg_w is not None:
            row["w"] = avg_w
        if alpha is not None:
            row["alpha"] = alpha

        for _, csv_col, _, _ in REWARD_COMPONENTS:
            if csv_col in avg_reward_kwargs and avg_reward_kwargs[csv_col] is not None:
                row[csv_col] = avg_reward_kwargs[csv_col]

        collision_penalty = avg_reward_kwargs.get("collision_penalty")
        if collision_penalty is not None:
            row["collision_penalty"] = collision_penalty
        if success is not None:
            row["success"] = int(success)
        if ever_collided is not None:
            row["ever_collided"] = int(ever_collided)

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        """Flush and close the CSV files."""
        self._csv_file.flush()
        self._csv_file.close()
        self._val_csv_file.flush()
        self._val_csv_file.close()
        self._csv_file.close()
