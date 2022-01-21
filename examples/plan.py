import bgheatmaps as bgh

"""
    This example shows how to use the `plan` functionality to visualize the 
    position of the planes used to 'slice' the brain for heatmap visualization.
"""

regions = [
    "TH",
    "RSP",
    "AI",
    "SS",
    "MO",
    "PVZ",
    "LZ",
    "VIS",
    "AUD",
    "RHP",
    "STR",
    "CB",
    "FRP",
    "HIP",
    "PA",
]


planner = bgh.plan(
    regions,
    position=(8000, 5000, 5000,),
    orientation="frontal",  # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
    thickness=2000,  # thickness of the slices used for rendering (in microns)
    arrow_scale=750,
)
planner.show()
