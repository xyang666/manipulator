"""
visualizer.py
-------------
Visualize generated trajectory scenes from JSON file.

Displays:
  - 3D workspace with start/goal points
  - Straight-line trajectory path
  - Spherical obstacles
  - Manipulability statistics

Usage:
    python -m trajectory.visualizer --input trajectories.json --scene_id 0
    python -m trajectory.visualizer --input trajectories.json --random 5
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict


class SceneVisualizer:
    """
    Visualize trajectory scenes with obstacles.
    """

    def __init__(self, json_path: str):
        """
        Parameters
        ----------
        json_path : path to trajectories JSON file
        """
        self.json_path = json_path
        self.scenes = self._load_scenes()

        print(f"[Visualizer] Loaded {len(self.scenes)} scenes from {json_path}")

    def _load_scenes(self) -> List[Dict]:
        """Load scenes from JSON file."""
        with open(self.json_path, 'r') as f:
            scenes = json.load(f)
        return scenes

    def get_scene(self, scene_id: int) -> Dict:
        """Get scene by ID."""
        for scene in self.scenes:
            if scene["scene_id"] == scene_id:
                return scene
        raise ValueError(f"Scene {scene_id} not found")

    def visualize_scene(self, scene_id: int, save_path: str = None):
        """
        Visualize a single scene in 3D.

        Parameters
        ----------
        scene_id  : scene identifier
        save_path : optional path to save figure
        """
        scene = self.get_scene(scene_id)

        start = np.array(scene["start"])
        goal = np.array(scene["goal"])
        obstacles = [np.array(obs) for obs in scene["obstacles"]]
        manip = scene["manipulability_mean"]

        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory line
        ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]],
                'b-', linewidth=3, label='Trajectory', zorder=10)

        # Plot start and goal points
        ax.scatter(*start, color='green', s=200, marker='o',
                  edgecolors='black', linewidths=2, label='Start', zorder=15)
        ax.scatter(*goal, color='red', s=200, marker='*',
                  edgecolors='black', linewidths=2, label='Goal', zorder=15)

        # Plot obstacles as spheres
        for i, obs in enumerate(obstacles):
            x, y, z, r = obs
            self._plot_sphere(ax, [x, y, z], r, color='orange', alpha=0.6)

        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')

        distance = np.linalg.norm(goal - start)
        ax.set_title(f'Scene {scene_id}: Trajectory with Obstacles\n'
                    f'Distance: {distance:.3f}m | Manipulability: {manip:.4f} | '
                    f'Obstacles: {len(obstacles)}',
                    fontsize=14, fontweight='bold', pad=20)

        # Set equal aspect ratio
        self._set_axes_equal(ax)

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)

        # Add statistics box
        stats_text = (
            f"Start: [{start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}]\n"
            f"Goal:  [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}]\n"
            f"Distance: {distance:.3f} m\n"
            f"Manipulability: {manip:.4f}\n"
            f"Obstacles: {len(obstacles)}"
        )
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 family='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_multiple(self, scene_ids: List[int], save_path: str = None):
        """
        Visualize multiple scenes in a grid.

        Parameters
        ----------
        scene_ids : list of scene IDs to visualize
        save_path : optional path to save figure
        """
        n_scenes = len(scene_ids)
        n_cols = min(3, n_scenes)
        n_rows = (n_scenes + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

        for idx, scene_id in enumerate(scene_ids):
            scene = self.get_scene(scene_id)
            start = np.array(scene["start"])
            goal = np.array(scene["goal"])
            obstacles = [np.array(obs) for obs in scene["obstacles"]]
            manip = scene["manipulability_mean"]

            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

            # Plot trajectory
            ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]],
                   'b-', linewidth=2, zorder=10)

            # Plot points
            ax.scatter(*start, color='green', s=100, marker='o', zorder=15)
            ax.scatter(*goal, color='red', s=100, marker='*', zorder=15)

            # Plot obstacles
            for obs in obstacles:
                x, y, z, r = obs
                self._plot_sphere(ax, [x, y, z], r, color='orange', alpha=0.5)

            # Labels
            ax.set_xlabel('X', fontsize=9)
            ax.set_ylabel('Y', fontsize=9)
            ax.set_zlabel('Z', fontsize=9)

            distance = np.linalg.norm(goal - start)
            ax.set_title(f'Scene {scene_id}\nDist: {distance:.2f}m, Manip: {manip:.3f}',
                        fontsize=10, fontweight='bold')

            self._set_axes_equal(ax)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_statistics(self, save_path: str = None):
        """
        Plot dataset statistics.

        Parameters
        ----------
        save_path : optional path to save figure
        """
        # Extract statistics
        manips = [s["manipulability_mean"] for s in self.scenes]
        distances = [np.linalg.norm(np.array(s["goal"]) - np.array(s["start"]))
                    for s in self.scenes]
        n_obstacles = [len(s["obstacles"]) for s in self.scenes]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Dataset Statistics ({len(self.scenes)} scenes)',
                    fontsize=14, fontweight='bold')

        # Manipulability histogram
        axes[0, 0].hist(manips, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(manips), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(manips):.4f}')
        axes[0, 0].set_xlabel('Manipulability', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Manipulability Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Distance histogram
        axes[0, 1].hist(distances, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(distances), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(distances):.3f}m')
        axes[0, 1].set_xlabel('Trajectory Distance (m)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Distance Distribution', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Scatter: Distance vs Manipulability
        axes[1, 0].scatter(distances, manips, alpha=0.6, s=50, c=n_obstacles,
                          cmap='viridis', edgecolors='black', linewidths=0.5)
        axes[1, 0].set_xlabel('Distance (m)', fontsize=11)
        axes[1, 0].set_ylabel('Manipulability', fontsize=11)
        axes[1, 0].set_title('Distance vs Manipulability', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
        cbar.set_label('# Obstacles', fontsize=10)

        # Statistics table
        axes[1, 1].axis('off')
        stats_text = f"""
Dataset Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total scenes:        {len(self.scenes)}

Manipulability:
  Mean:              {np.mean(manips):.4f}
  Std:               {np.std(manips):.4f}
  Min:               {np.min(manips):.4f}
  Max:               {np.max(manips):.4f}

Distance (m):
  Mean:              {np.mean(distances):.3f}
  Std:               {np.std(distances):.3f}
  Min:               {np.min(distances):.3f}
  Max:               {np.max(distances):.3f}

Obstacles per scene:
  Mean:              {np.mean(n_obstacles):.1f}
  Mode:              {max(set(n_obstacles), key=n_obstacles.count)}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def _plot_sphere(self, ax, center, radius, color='orange', alpha=0.6):
        """Plot a sphere in 3D."""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')

    def _set_axes_equal(self, ax):
        """Set equal aspect ratio for 3D plot."""
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def main():
    parser = argparse.ArgumentParser(description='Visualize trajectory scenes from JSON')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to trajectories JSON file')
    parser.add_argument('--scene_id', type=int, default=None,
                       help='Specific scene ID to visualize')
    parser.add_argument('--random', type=int, default=None,
                       help='Visualize N random scenes')
    parser.add_argument('--stats', action='store_true',
                       help='Plot dataset statistics')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path to save figure')

    args = parser.parse_args()

    # Resolve input path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        input_path = project_root / args.input
        if not input_path.exists():
            print(f"Error: File not found at {input_path}")
            sys.exit(1)

    # Create visualizer
    viz = SceneVisualizer(str(input_path))

    if args.stats:
        # Plot statistics
        output = args.output or str(input_path.parent / "dataset_statistics.png")
        viz.plot_statistics(save_path=output)

    elif args.scene_id is not None:
        # Visualize single scene
        output = args.output or str(input_path.parent / f"scene_{args.scene_id}.png")
        viz.visualize_scene(args.scene_id, save_path=output)

    elif args.random is not None:
        # Visualize random scenes
        n_scenes = min(args.random, len(viz.scenes))
        scene_ids = np.random.choice(len(viz.scenes), n_scenes, replace=False)
        output = args.output or str(input_path.parent / f"scenes_random_{n_scenes}.png")
        viz.visualize_multiple(scene_ids.tolist(), save_path=output)

    else:
        # Default: show first scene
        print("No visualization option specified. Use --scene_id, --random, or --stats")
        print(f"Available scenes: 0-{len(viz.scenes)-1}")
        sys.exit(1)


if __name__ == "__main__":
    main()
