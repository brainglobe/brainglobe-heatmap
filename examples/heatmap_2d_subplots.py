import matplotlib.pyplot as plt

import brainglobe_heatmap as bgh

data_dict = {
    "VISpm": 0.0,
    "VISp": 0.14285714285714285,
    "VISl": 0.2857142857142857,
    "VISli": 0.42857142857142855,
    "VISal": 0.5714285714285714,
    "VISrl": 0.7142857142857142,
    "SSp-bfd": 0.8571428571428571,
    "VISam": 1.0,
}

# Create a list of scenes to plot
# Note: it's important to keep reference to the scenes to avoid a
# segmentation fault
scenes = []
for distance in range(7500, 10500, 500):
    scene = bgh.Heatmap(
        data_dict,
        position=distance,
        orientation="frontal",
        thickness=10,
        format="2D",
        cmap="Reds",
        vmin=0,
        vmax=1,
        label_regions=False,
    )
    scenes.append(scene)

# Create a figure with 6 subplots and plot the scenes
fig, axs = plt.subplots(3, 2, figsize=(18, 12))
for scene, ax in zip(scenes, axs.flatten(), strict=False):
    scene.plot_subplot(fig=fig, ax=ax, show_cbar=True, hide_axes=False)

plt.tight_layout()
plt.show()
