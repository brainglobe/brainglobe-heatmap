import bgheatmaps as bgh

"""
    This example shows how to get the coordinates 
    of selected brain regions in the slicing plane
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


coordinates = bgh.get_structures_slice_coords(
    regions,
    position=(8000, 5000, 5000,),
    orientation="frontal",  # 'frontal' or 'sagittal', or 'horizontal' or a tuple (x,y,z)
)

print(coordinates)
