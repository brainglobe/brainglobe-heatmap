import brainglobe_heatmap as bgh

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

# Region annotation:
# - False or True to annotate all regions with their names
annotate_regions = True
# - List[str]: annotate only specified regions with their names,
# ['olfactory bulb', 'dorsal thalamus']
annotate_regions_specific = ["olfactory bulb"]
# - Dict[str, Union[str, int, float]]: annotate regions with custom text,
# dict(TH='Thalamus', 'RSP'=0.2)
annotate_regions_custom = values

annotate_text_options = dict(
    fontweight="normal",
    fontsize=8,
    rotation="horizontal",  # float or {'vertical', 'horizontal'}
    color="black",
    alpha=1,  # float in range 0-1
)

f = bgh.Heatmap(
    values,
    position=175,
    # 'frontal' or 'sagittal', or 'horizontal' or a tuple (x,y,z)
    orientation="horizontal",
    thickness=250,
    atlas_name="mpin_zfish_1um",
    format="2D",
    title="zebra fish heatmap",
    annotate_regions=False,
    annotate_text_options=annotate_text_options,
).show(xlabel="AP (μm)", ylabel="DV (μm)")
