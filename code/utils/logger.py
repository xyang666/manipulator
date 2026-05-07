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


CSV_COLUMNS = [
    "global_step", "episode", "ep_step",
    "reward", "d_obs", "w",
    "r_track", "r_obs", "r_manip", "r_energy", "r_collision", "collision_penalty",
    "critic_loss", "actor_rl_loss", "physics_loss", "actor_loss", "alpha",
]


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

    def log_step(self, step: int, episode: int, ep_step: int,
                 reward: float, info: dict) -> None:
        """Call once per env step, immediately after env.step()."""
        self._ep_rewards.append(reward)
        self._ep_d_obs.append(info.get("d_obs", float("nan")))
        self._ep_w.append(info.get("w", float("nan")))

        row: dict = {
            "global_step":      step,
            "episode":          episode,
            "ep_step":          ep_step,
            "reward":           reward,
            "d_obs":            info.get("d_obs", ""),
            "w":                info.get("w", ""),
            "r_track":          info.get("r_track", ""),
            "r_obs":            info.get("r_obs", ""),
            "r_manip":          info.get("r_manip", ""),
            "r_energy":         info.get("r_energy", ""),
            "r_collision":      info.get("r_collision", ""),
            "collision_penalty": info.get("collision_penalty", ""),
        }

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

        self._csv_writer.writerow(row)

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
            "success":           min_d_obs >= 0.02,
        }

        self._csv_file.flush()

        self._ep_losses.clear()
        self._ep_rewards.clear()
        self._ep_d_obs.clear()
        self._ep_w.clear()

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
                             avg_physics_loss: float, alpha: float = None,
                             avg_critic_loss: float = None,
                             avg_actor_total_loss: float = None,
                             avg_w: float = None,
                             avg_r_track: float = None,
                             avg_r_obs: float = None,
                             avg_r_manip: float = None,
                             avg_r_energy: float = None,
                             avg_r_collision: float = None) -> None:
        """Write a single episode-summary row to the training CSV.

        Used by the parallel training path (no per-step CSV logging).
        """
        row = {
            "global_step":      step,
            "episode":          episode,
            "reward":           total_reward,
            "d_obs":            min_d_obs,
            "actor_rl_loss":    avg_actor_loss,
            "physics_loss":     avg_physics_loss,
        }
        if avg_critic_loss is not None:
            row["critic_loss"] = avg_critic_loss
        if avg_actor_total_loss is not None:
            row["actor_loss"] = avg_actor_total_loss
        if avg_w is not None:
            row["w"] = avg_w
        if alpha is not None:
            row["alpha"] = alpha
        if avg_r_track is not None:
            row["r_track"] = avg_r_track
        if avg_r_obs is not None:
            row["r_obs"] = avg_r_obs
        if avg_r_manip is not None:
            row["r_manip"] = avg_r_manip
        if avg_r_energy is not None:
            row["r_energy"] = avg_r_energy
        if avg_r_collision is not None:
            row["r_collision"] = avg_r_collision
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        """Flush and close the CSV files."""
        self._csv_file.flush()
        self._csv_file.close()
        self._val_csv_file.flush()
        self._val_csv_file.close()
        self._csv_file.close()
