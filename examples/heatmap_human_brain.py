"""
This example shows how to generate a heatmap with the `allen_human_500um`
human brain atlas.

N.B. the physical scales are very different to e.g. the small animal atlases.
"""

import brainglobe_heatmap as bgh

values = dict(SFG=1, PrCG=2, Ca=4, Pu=10)  # scalar values for each region

# Region annotation:
# - False or True to annotate all regions with their names
annotate_regions = True
# - List[str]: annotate only specified regions with their names,
# ['Ca', 'Pu']
annotate_regions_specific = ["Pu"]
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

scene = bgh.Heatmap(
    values,
    position=(100000, 100000, 100000),
    thickness=1000,
    atlas_name="allen_human_500um",
    format="2D",
    annotate_regions=False,
    annotate_text_options=annotate_text_options,
).show(filename="2")
