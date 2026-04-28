"""
reachability.py
---------------
Monte Carlo sampling for workspace reachability analysis.

Generates a 3D heatmap showing the reachable workspace of a 7-DOF manipulator
by randomly sampling joint configurations and recording end-effector positions.

Usage:
    python -m trajectory.reachability --samples 100000 --resolution 50
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.kinematics import ManipulatorKinematics


class ReachabilityAnalyzer:
    """
    Analyzes manipulator workspace using Monte Carlo sampling.
    """

    def __init__(self, urdf_path: str, n_joints: int = 7):
        """
        Parameters
        ----------
        urdf_path : path to URDF file
        n_joints  : number of joints (default: 7)
        """
        self.kin = ManipulatorKinematics(urdf_path=urdf_path, n_joints=n_joints)
        self.n = n_joints

        # Joint limits (Panda default)
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        # Workspace bounds (will be computed from samples)
        self.workspace_min = None
        self.workspace_max = None

    def sample_configurations(self, n_samples: int = 100000, seed: int = 42) -> np.ndarray:
        """
        Randomly sample joint configurations within joint limits.

        Parameters
        ----------
        n_samples : number of samples
        seed      : random seed for reproducibility

        Returns
        -------
        positions : (n_samples, 3) array of end-effector positions
        """
        np.random.seed(seed)
        positions = []

        print(f"Sampling {n_samples} configurations...")
        for i in range(n_samples):
            if (i + 1) % 10000 == 0:
                print(f"  Progress: {i + 1}/{n_samples}")

            # Uniform random sampling within joint limits
            q = np.random.uniform(self.q_min, self.q_max)

            # Forward kinematics
            pos, _ = self.kin.forward_kinematics(q)
            positions.append(pos)

        positions = np.array(positions)
        print(f"Sampling complete. Generated {len(positions)} positions.")

        # Compute workspace bounds
        self.workspace_min = positions.min(axis=0)
        self.workspace_max = positions.max(axis=0)
        print(f"Workspace bounds:")
        print(f"  X: [{self.workspace_min[0]:.3f}, {self.workspace_max[0]:.3f}]")
        print(f"  Y: [{self.workspace_min[1]:.3f}, {self.workspace_max[1]:.3f}]")
        print(f"  Z: [{self.workspace_min[2]:.3f}, {self.workspace_max[2]:.3f}]")

        return positions

    def create_voxel_grid(self, positions: np.ndarray, resolution: int = 50) -> np.ndarray:
        """
        Create 3D voxel grid from sampled positions.

        Parameters
        ----------
        positions  : (n_samples, 3) array of positions
        resolution : grid resolution per axis

        Returns
        -------
        grid : (resolution, resolution, resolution) array with density counts
        """
        print(f"Creating voxel grid with resolution {resolution}^3...")

        # Create 3D histogram
        grid, edges = np.histogramdd(
            positions,
            bins=[resolution, resolution, resolution],
            range=[[self.workspace_min[0], self.workspace_max[0]],
                   [self.workspace_min[1], self.workspace_max[1]],
                   [self.workspace_min[2], self.workspace_max[2]]]
        )

        print(f"Voxel grid created. Max density: {grid.max():.0f} samples/voxel")
        return grid

    def plot_heatmap_slices(self, grid: np.ndarray, output_path: str = "reachability_heatmap.png"):
        """
        Plot 2D heatmap slices (XY, XZ, YZ planes) of the 3D voxel grid.

        Parameters
        ----------
        grid        : (resolution, resolution, resolution) voxel grid
        output_path : path to save figure
        """
        print(f"Generating heatmap slices...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Workspace Reachability Analysis (Monte Carlo Sampling)', fontsize=14, fontweight='bold')

        # Custom colormap: white -> blue -> red
        colors = ['white', 'lightblue', 'blue', 'orange', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('reachability', colors, N=n_bins)

        # XY plane (top view, Z = middle)
        z_mid = grid.shape[2] // 2
        xy_slice = grid[:, :, z_mid].T
        im1 = axes[0, 0].imshow(xy_slice, cmap=cmap, origin='lower', aspect='auto',
                                extent=[self.workspace_min[0], self.workspace_max[0],
                                       self.workspace_min[1], self.workspace_max[1]])
        axes[0, 0].set_title(f'XY Plane (Z = {self.workspace_min[2] + (self.workspace_max[2] - self.workspace_min[2]) / 2:.2f}m)')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        plt.colorbar(im1, ax=axes[0, 0], label='Density')

        # XZ plane (side view, Y = middle)
        y_mid = grid.shape[1] // 2
        xz_slice = grid[:, y_mid, :].T
        im2 = axes[0, 1].imshow(xz_slice, cmap=cmap, origin='lower', aspect='auto',
                                extent=[self.workspace_min[0], self.workspace_max[0],
                                       self.workspace_min[2], self.workspace_max[2]])
        axes[0, 1].set_title(f'XZ Plane (Y = {self.workspace_min[1] + (self.workspace_max[1] - self.workspace_min[1]) / 2:.2f}m)')
        axes[0, 1].set_xlabel('X (m)')
        axes[0, 1].set_ylabel('Z (m)')
        plt.colorbar(im2, ax=axes[0, 1], label='Density')

        # YZ plane (front view, X = middle)
        x_mid = grid.shape[0] // 2
        yz_slice = grid[x_mid, :, :].T
        im3 = axes[1, 0].imshow(yz_slice, cmap=cmap, origin='lower', aspect='auto',
                                extent=[self.workspace_min[1], self.workspace_max[1],
                                       self.workspace_min[2], self.workspace_max[2]])
        axes[1, 0].set_title(f'YZ Plane (X = {self.workspace_min[0] + (self.workspace_max[0] - self.workspace_min[0]) / 2:.2f}m)')
        axes[1, 0].set_xlabel('Y (m)')
        axes[1, 0].set_ylabel('Z (m)')
        plt.colorbar(im3, ax=axes[1, 0], label='Density')

        # Statistics panel
        axes[1, 1].axis('off')
        stats_text = f"""
Workspace Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
X range: [{self.workspace_min[0]:.3f}, {self.workspace_max[0]:.3f}] m
Y range: [{self.workspace_min[1]:.3f}, {self.workspace_max[1]:.3f}] m
Z range: [{self.workspace_min[2]:.3f}, {self.workspace_max[2]:.3f}] m

Volume: {np.prod(self.workspace_max - self.workspace_min):.3f} m³
Max density: {grid.max():.0f} samples/voxel
Non-empty voxels: {np.count_nonzero(grid)} / {grid.size}
Reachability: {100 * np.count_nonzero(grid) / grid.size:.1f}%
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Workspace reachability analysis using Monte Carlo sampling')
    parser.add_argument('--urdf', type=str, default='panda_description/urdf/panda.urdf',
                       help='Path to URDF file')
    parser.add_argument('--samples', type=int, default=100000,
                       help='Number of Monte Carlo samples')
    parser.add_argument('--resolution', type=int, default=50,
                       help='Voxel grid resolution per axis')
    parser.add_argument('--output', type=str, default='reachability_heatmap.png',
                       help='Output figure path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Resolve URDF path
    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        urdf_path = project_root / args.urdf
        if not urdf_path.exists():
            print(f"Error: URDF file not found at {urdf_path}")
            sys.exit(1)

    print("=" * 60)
    print("Workspace Reachability Analysis")
    print("=" * 60)
    print(f"URDF: {urdf_path}")
    print(f"Samples: {args.samples}")
    print(f"Resolution: {args.resolution}^3")
    print("=" * 60)

    # Run analysis
    analyzer = ReachabilityAnalyzer(urdf_path=str(urdf_path))
    positions = analyzer.sample_configurations(n_samples=args.samples, seed=args.seed)
    grid = analyzer.create_voxel_grid(positions, resolution=args.resolution)
    analyzer.plot_heatmap_slices(grid, output_path=args.output)

    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
