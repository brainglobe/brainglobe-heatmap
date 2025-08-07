"""
This example shows how to go from cell counts per region
(e.g. as outputted by cellfinder) to a plot showing the
density (count/volume) of cells in each brain region
"""

from pathlib import Path

import pandas as pd
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from brainrender._io import load_mesh_from_file

import brainglobe_heatmap as bgh

# get the number of cells for each region
cells_summary = pd.read_csv(Path(__file__).parent / "summary.csv")

# get regions two levels up the hierarchy
atlas = BrainGlobeAtlas("allen_mouse_25um")
structures_df = atlas.lookup_df

# Merge the cells summary with the structures CSV to get structure
# acronyms and IDs
cells_summary = pd.merge(
    cells_summary, structures_df, left_on="structure_name", right_on="name"
)

# Fetch the 2nd order parent region for each region to accumulate
# cell counts at that level
cells_summary["parent_region"] = [
    (
        atlas.get_structure_ancestors(r)[-2]
        if (len(atlas.get_structure_ancestors(r)) > 1)
        else None
    )
    for r in cells_summary["id"]
]
# Filter out rows where parent_region is NaN
cells_summary = cells_summary[cells_summary["parent_region"].notna()]

# Accumulate cell counts for each parent region
cells_summary = cells_summary.groupby("parent_region", as_index=False).sum(
    numeric_only=True
)
cells_summary.set_index("parent_region", inplace=True)

# Get regions' volume
volumes = []
for region in cells_summary.index:
    obj_file = str(atlas.meshfile_from_structure(region))
    mesh = load_mesh_from_file(obj_file)
    volumes.append(mesh.volume())

# Calculate the density using atlas volume for each region
cells_summary["volume_mm3"] = volumes
cells_summary["cells_per_mm3"] = (
    cells_summary["total_cells"] / cells_summary["volume_mm3"]
)

# Filter out regions with very low cell density
cells_summary = cells_summary.loc[cells_summary["cells_per_mm3"] > 5 * 1e-9]

print(cells_summary[["volume_mm3", "cells_per_mm3"]])

f = bgh.Heatmap(
    cells_summary["cells_per_mm3"].to_dict(),
    position=(
        8000,
        5000,
        5000,
    ),
    orientation="frontal",  # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
    title="cell density",
    format="2D",
    cmap="Reds",
).show()
