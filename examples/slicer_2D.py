import bgheatmaps as bgh

from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects

from bg_atlasapi import BrainGlobeAtlas


"""
    This example shows how to use bgheatmaps to get the contours of 'sliced'
    brain regions to overlay them over histological data.
"""

atlas_name = "mpin_zfish_1um"  # name of the atlas

dors_vent_slice_pos = 130  # position of the slice in um

# Cut coordinates from meshes:
coords_dict = bgh.get_structures_slice_coords(
    ["tectum", "midbrain"],
    orientation="horizontal",
    atlas_name="mpin_zfish_1um",
    position=(0, dors_vent_slice_pos, 0),
)
_ = coords_dict.pop("root")  # we won't plot the root mask

# Get atlas slice:
atlas = BrainGlobeAtlas(atlas_name)
ref_slice_idx = int(dors_vent_slice_pos / atlas.resolution[0])
confocal_slice = atlas.reference[:, ref_slice_idx, :]

# plot:
f, ax = plt.subplots(figsize=(6, 9))
plt.imshow(confocal_slice, cmap="gray_r")
for struct_name, contours in coords_dict.items():
    for cont in contours:
        plt.fill(cont[:, 0], -cont[:, 1], lw=1, fc="none", ec="k")

        # write outlined text to label each brain region
        txt = plt.text(
            cont[:, 0].mean(),
            -cont[:, 1].mean(),
            struct_name[:4] + ".",
            ha="center",
            va="center",
            fontsize=12,
        )
        txt.set_path_effects(
            [path_effects.withStroke(linewidth=6, foreground="white",)]
        )


ax.set(xlabel="Left - Right pos (µm)", ylabel="Post. - Ant. pos (µm)")
plt.show()
