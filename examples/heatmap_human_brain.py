"""
This example shows how to generate a heatmap with the `allen_human_500um`
human brain atlas.

N.B. the physical scales are very different to e.g. the small animal atlases.
"""

import brainglobe_heatmap as bgh

values = dict(SFG=1, PrCG=2, Ca=4, Pu=10)  # scalar values for each region


scene = bgh.Heatmap(
    values,
    position=(100000, 100000, 100000),
    thickness=1000,
    atlas_name="allen_human_500um",
    format="2D",
).show()
