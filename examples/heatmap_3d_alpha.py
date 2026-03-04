"""
This example shows how to control the transparency
of brain regions in a 3D heatmap.

The alpha parameter accepts a float (applied to all regions)
or a dict mapping region acronyms to individual alpha values.
"""

import brainglobe_heatmap as bgh

values = dict(
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

# Apply per-region alpha: HIP at 30% opacity, TH at 80% opacity
scene = bgh.Heatmap(
    values,
    position=(8000, 5000, 5000),
    orientation="frontal",
    thickness=1000,
    title="frontal",
    vmin=-5,
    vmax=3,
    format="3D",
    alpha={"HIP": 0.3, "TH": 0.8},
).show()
