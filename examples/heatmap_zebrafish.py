import bgheatmaps as bgh

"""
    This example shows how to use visualize a heatmap in 2D
"""

# from brainrender import Scene

# scene = Scene(atlas_name='mpin_zfish_1um')
# print(scene.atlas.lookup_df.iloc[40:60])

values = {
    "facial motor nucleus": 3.5,
    "anterior trigeminal motor region": 0.5,
    "locus coeruleus": -2,
    "dorsal thalamus": -1,
    "ventral habenula": -4,
    "caudal hypothalamus": 2.5,
    "pituitary": 4,
    "posterior tuberculum": -3,
    "preoptic region": 3,
    "olfactory bulb": 5,
    "medial tegmentum": -2,
    "tectal neuropil": -4,
}

f = bgh.heatmap(
    values,
    position=None,
    orientation="sagittal",  # 'frontal' or 'sagittal', or 'horizontal' or a tuple (x,y,z)
    thickness=1000,
    atlas_name="mpin_zfish_1um",
    format="2D",
    title="zebra fish heatmap",
).show(xlabel="AP (μm)", ylabel="DV (μm)")
