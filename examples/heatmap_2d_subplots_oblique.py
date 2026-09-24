"""
This example shows named and oblique orientations side by side in subplots.
"""

import matplotlib.pyplot as plt

import brainglobe_heatmap as bgh

values = dict(  # scalar values for each region
    TH=1,
    RSP=0.2,
    AI=0.4,
    SS=-3,
    MO=2.6,
    PVZ=-4,
    LZ=-3,
    VIS=2,
    AUD=0.3,
    RHP=-0.2,
    STR=0.5,
    CB=0.5,
    FRP=-1.7,
    HIP=3,
    PA=-4,
)

orientations_title = [
    ("frontal", "frontal"),
    ("horizontal", "horizontal"),
    ("sagittal", "sagittal"),
    ((1, 0, 0), "frontal (1, 0, 0)"),
    ((0, 1, 0), "horizontal (0, 1, 0)"),
    ((0, 0, 1), "sagittal (0, 0, 1)"),
    ((1, 0, 0.3), "front tilted LR (1, 0, 0.3)"),
    ((0, 1, 0.3), "horiz tilted LR (0, 1, 0.3)"),
    ((0, 0.3, 1), "sagit tilted DV (0, 0.3 ,1)"),
    ((1, 0.3, 0), "front tilted DV (1, 0.3, 0)"),
    ((0.3, 1, 0), "horiz tilted AP (0.3, 1, 0)"),
    ((0.3, 0, 1), "sagit tilted AP (0.3, 0, 1)"),
]

# Create all heatmap scenes first to avoid segmentation fault
scenes = []
for orient, title in orientations_title:
    scene = bgh.Heatmap(
        values,
        position=(8000, 5000, 5000),
        orientation=orient,  # type: ignore[arg-type]
        title=title,  # type: ignore[arg-type]
        vmin=-5,
        vmax=3,
        format="2D",
    )
    scenes.append(scene)

fig, axs = plt.subplots(4, 3, figsize=(20, 28))
for scene, ax in zip(scenes, axs.flatten(), strict=False):
    scene.plot_subplot(fig=fig, ax=ax, show_cbar=True, hide_axes=False)

plt.tight_layout()
plt.show()
