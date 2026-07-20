"""Animated 2D heatmap example with fixed color normalization across frames."""

from pathlib import Path

import matplotlib.pyplot as plt
from brainglobe_atlasapi import BrainGlobeAtlas
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import brainglobe_heatmap as bgh

# Heat values used throughout the animation.
values = {
    "TH": 1,
    "RSP": 0.2,
    "AI": 0.4,
    "SS": -3,
    "MO": 2.6,
    "PVZ": -4,
    "LZ": -3,
    "VIS": 2,
    "AUD": 0.3,
    "RHP": -0.2,
    "STR": 0.5,
    "CB": 0.5,
    "FRP": -1.7,
    "HIP": 3,
    "PA": -4,
}

# Example settings.
atlas_name = "allen_mouse_25um"
orientation = "frontal"
cmap = "Reds"
step_um = 500
fps = 3

vmin = min(values.values())
vmax = max(values.values())

axis_idx = {"frontal": 0, "horizontal": 1, "sagittal": 2}[orientation]
atlas = BrainGlobeAtlas(atlas_name)

# Build the slice positions for the selected axis.
max_pos_um = atlas.reference.shape[axis_idx] * atlas.resolution[axis_idx]

positions = list(range(0, int(max_pos_um) + 1, step_um))

# Set up the figure and a shared colorbar.
fig, ax = plt.subplots(figsize=(10, 4))

# Keep the same color scale across frames.
norm = Normalize(vmin=vmin, vmax=vmax)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Value")


def update_frames(frame_idx):
    """Draw one animation frame for the current slice position."""
    ax.clear()
    pos = positions[frame_idx]

    # Recreate the heatmap for this slice position.
    heatmap = bgh.Heatmap(
        values,
        position=pos,
        orientation=orientation,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        format="2D",
        atlas_name=atlas_name,
    )
    heatmap.plot_subplot(fig, ax, show_cbar=False)
    ax.set_title(f"Position: {pos} um")
    # Show the current frame number.
    ax.text(
        0.02,
        0.98,
        f"Frame {frame_idx + 1}/{len(positions)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )


# Build the animation by calling the update function for each frame.
ani = FuncAnimation(
    fig,
    update_frames,
    frames=len(positions),
    interval=500,
    repeat=False,
)

# Save the animation next to this example script.
output_path = Path(__file__).with_name("brain_animation.gif")
ani.save(output_path, writer="pillow", fps=fps)

plt.show()
