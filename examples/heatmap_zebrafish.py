import bgheatmaps as bgh


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
    "oculomotor nucleus": 1,
    "tegmentum": -0.5,
    "inferior olive": -0.75,
    "inferior medulla oblongata": 1.5,
    "cerebellum": 0.5,
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
