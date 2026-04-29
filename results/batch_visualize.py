"""
batch_visualize.py
------------------
Batch visualize all scenes from JSON file in 3x3 grids using visualizer.py

Usage:
    python batch_visualize.py
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from trajectory.visualizer import SceneVisualizer

# Load scenes
json_path = Path(__file__).parent / "trajectories.json"
viz = SceneVisualizer(str(json_path))

# Create output directory
output_dir = Path(__file__).parent / "visualizations"
output_dir.mkdir(exist_ok=True)

print(f"Loaded {len(viz.scenes)} scenes")
print(f"Generating visualizations in 3x3 grids...")

# Generate visualizations in batches of 9
num_scenes = len(viz.scenes)
batch_size = 9

# Use dataset indices instead of scene IDs (handles gaps from rejected scenes)
scene_ids = [s["scene_id"] for s in viz.scenes]

for batch_idx in range(0, num_scenes, batch_size):
    end_idx = min(batch_idx + batch_size, num_scenes)
    batch = scene_ids[batch_idx:end_idx]

    output_path = output_dir / f"scenes_{batch[0]:03d}_{batch[-1]:03d}.png"

    print(f"  Batch {batch_idx//batch_size + 1}: scenes {batch[0]}-{batch[-1]}")
    viz.visualize_multiple(batch, save_path=str(output_path))

print(f"\nAll visualizations saved to: {output_dir}")
print(f"Total batches: {(num_scenes + batch_size - 1) // batch_size}")
