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
cells_path = Path(__file__).parent / "cell_counts_example.h5"
cells_summary = pd.read_csv(Path(__file__).parent / "summary.csv")

data = pd.read_hdf(cells_path)
cell_counts = data.groupby("region").count()

# get regions two levels up the hierarchy
atlas = BrainGlobeAtlas("allen_mouse_25um")
structures_csv = pd.read_csv(atlas.root_dir / "structures.csv")

cells_summary = cells_summary[cells_summary["total_cells"] > 0]
merged_df = pd.merge(
    cells_summary, structures_csv, left_on="structure_name", right_on="name"
)
mask = [
    len(atlas.get_structure_descendants(atl_id)) == 0
    for atl_id in merged_df["id"]
]
filtered_merged_df = merged_df[mask]

par_regions = set(
    atlas.get_structure_ancestors(r)[-2] for r in filtered_merged_df["id"]
)
filtered_par_merged_df = merged_df[merged_df["acronym"].isin(par_regions)]

parent_regions = [
    atlas.get_structure_ancestors(r)[-2] for r in cell_counts.index
]

cell_counts["parent_region"] = parent_regions
cell_counts = cell_counts.groupby("parent_region").sum()


# get regions' volume
volumes = []
for region in cell_counts.index:
    obj_file = str(atlas.meshfile_from_structure(region))
    mesh = load_mesh_from_file(obj_file)
    volumes.append(mesh.volume())

# get the density (num cells / volume)
cell_counts["volume"] = volumes
cell_counts["density"] = cell_counts["x"] / cell_counts["volume"]
cell_counts = cell_counts.loc[cell_counts.density > 5 * 1e-9]

print(cell_counts)


f = bgh.Heatmap(
    cell_counts.density.to_dict(),
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
