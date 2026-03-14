"""Animated 2D heatmap example with fixed color normalization across frames."""

from pathlib import Path
import brainglobe_heatmap as bgh
import matplotlib.pyplot as plt
from brainglobe_atlasapi import BrainGlobeAtlas
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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

# Keep normalization fixed for valid between-frame comparisons.
vmin = min(values.values())
vmax = max(values.values())

atlas_name = "allen_mouse_25um"
orientation = "frontal"
cmap = "Reds"
step_um = 500
fps = 3

axis_idx = {"frontal": 0, "horizontal": 1, "sagittal": 2}[orientation]
atlas = BrainGlobeAtlas(atlas_name)
# Calculate the full range of slice positions along the selected axis.
max_pos_um = atlas.reference.shape[axis_idx] * atlas.resolution[axis_idx]

positions = list(range(0, int(max_pos_um) + 1, step_um))

fig, ax = plt.subplots(figsize=(10, 4))

# Fixed colour normalization for fair comparison across frames.
norm = Normalize(vmin=vmin, vmax=vmax)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Value")


def update_frames(frame_idx):
    ax.clear()
    pos = positions[frame_idx]

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


ani = FuncAnimation(
    fig,
    update_frames,
    frames=len(positions),
    interval=500,
    repeat=False,
)

output_path = Path(__file__).with_name("brain_animation.gif")
ani.save(output_path, writer="pillow", fps=fps)

plt.show()