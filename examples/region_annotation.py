"""
This example shows how generate a heatmap
in 2D and 3D with annotations on brain regions
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

annotate_regions = True
annotate_text_options_2d = dict(
    fontweight="normal",
    fontsize=10,
    rotation="horizontal",  # float or 'vertical' or 'horizontal'
    color="black",
    alpha=1,  # float in range 0-1
)
# ----------------------------------------
# Annotate specified regions,
# annotate_regions = ["TH", "MOs"]
#
# Annotate regions with custom values or text,
# annotate_regions = dict(MO=0.4, SS="Text")

f = bgh.Heatmap(
    values,
    # when using a named orientation, you can pass a single value!
    position=9000,
    # 'frontal' or 'sagittal,' or 'horizontal' or a tuple (x,y,z)
    orientation="frontal",
    thickness=3000,
    title="frontal view",
    vmin=-5,
    vmax=3,
    annotate_regions=annotate_regions,
    annotate_text_options_2d=annotate_text_options_2d,
    format="2D",  # or 3D
).show()
