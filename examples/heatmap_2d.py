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
).show()
