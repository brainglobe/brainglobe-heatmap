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


coordinates = bgh.get_plane_coordinates(
    regions,
    position=(
        8000,
        5000,
        5000,
    ),  # displacement along the AP axis relative to midpoint
    orientation="frontal",  # 'frontal' or 'sagittal', or 'top' or a tuple (x,y,z)
)

print(coordinates)
