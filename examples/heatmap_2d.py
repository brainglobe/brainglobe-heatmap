"""
This example shows how to use visualize a heatmap in 2D
"""

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

# Region annotation:
# - False or True to annotate all regions with their names
annotate_regions = True
# - List[str]: annotate only specified regions with their names,
# ['TH', 'RSP']
annotate_regions_specific = ["HIP"]
# - Dict[str, Union[str, int, float]]: annotate regions with custom text,
# dict(TH='Thalamus', 'RSP'=0.2)
annotate_regions_custom = values

annotate_text_options = dict(
    fontweight="normal",
    fontsize=10,
    rotation="horizontal",  # float or {'vertical', 'horizontal'}
    color="black",
    alpha=1,  # float in range 0-1
)

f = bgh.Heatmap(
    values,
    # when using a named orientation, you can pass a single value!
    position=5000,
    # 'frontal' or 'sagittal,' or 'horizontal' or a tuple (x,y,z)
    orientation="frontal",
    title="horizontal view",
    vmin=-5,
    vmax=3,
    format="2D",
    annotate_regions=True,
    annotate_text_options=annotate_text_options,
).show()
