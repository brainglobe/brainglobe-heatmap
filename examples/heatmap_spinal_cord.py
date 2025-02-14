import brainglobe_heatmap as bgh

values = {
    "2Ssp": 3.5,
    "3Sp": 0.5,
    "IB": -2,
    "5Sp": -1,
    "7Sp": -4,
    "LDCom": 2.5,
    "10Sp": 4,
    "D": -3,
    "ICl": -4,
    "6Sp": 4,
    "gr": 8,
    "vf": -1,
    "rs": -4,
    "LSp": 7,
    "dcs": -5,
}

# Region annotation:
# - False or True to annotate all regions with their names
annotate_regions = True
# - List[str]: annotate only specified regions with their names,
# ['gr', 'vf']
annotate_regions_specific = ["gr"]
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
    position=1000,
    # 'frontal' or 'sagittal', or 'horizontal' or a tuple (x,y,z)
    orientation="frontal",
    thickness=1000,
    atlas_name="allen_cord_20um",
    annotate_regions=False,
    annotate_text_options=annotate_text_options,
    format="2D",
).show()
