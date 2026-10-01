"""
Tests output of `get_structures_slice_coords` VS annotation coords
"""

import numpy as np
import pytest
from brainglobe_atlasapi import BrainGlobeAtlas

import brainglobe_heatmap as bgh

ATLAS_NAME = "allen_mouse_25um"
REGIONS = ["HIP"]
POSITION = (8000, 5000, 5000)
THRESHOLD_UM = 75  # 3 voxels at 25µm resolution
OBLIQUE_ORIENTATIONS = [
    (1, 0.3, 0),
    (1, 0, 0.3),
    (0.3, 1, 0),
    (0, 1, 0.3),
    (0.3, 0, 1),
    (0, 0.3, 1),
    (1, 1, 0),
]


@pytest.fixture(scope="module")
def atlas():
    return BrainGlobeAtlas(ATLAS_NAME, check_latest=False)


# Expects that mesh coords match annotation coords
@pytest.mark.parametrize("orientation", ["frontal", "horizontal", "sagittal"])
@pytest.mark.parametrize("region", REGIONS)
def test_mesh_coords_match_annotation(atlas, orientation, region):
    ORIENTATION_TO_AXIS = {"frontal": 0, "horizontal": 1, "sagittal": 2}

    atlas_resolution = atlas.resolution[0]

    coords = bgh.get_structures_slice_coords(
        [region], position=POSITION, orientation=orientation
    )

    # mesh - um
    mesh_pts = np.vstack(coords[region])

    # annotation - voxels
    axis_index = ORIENTATION_TO_AXIS[orientation]
    idx = int(POSITION[axis_index] / atlas_resolution)
    mask_3d = atlas.get_structure_mask(region)
    if axis_index == 0:  # frontal
        mask_2d = mask_3d[idx, :, :] > 0
    elif axis_index == 1:  # horizontal
        mask_2d = mask_3d[:, idx, :] > 0
    else:  # sagital
        mask_2d = mask_3d[:, :, idx] > 0

    rows, cols = np.where(mask_2d)

    # voxel to µm
    # sagittal x=rows, y=cols
    if orientation == "sagittal":
        ann_pts = np.column_stack(
            [rows * atlas_resolution, cols * atlas_resolution]
        )
    else:
        ann_pts = np.column_stack(
            [cols * atlas_resolution, rows * atlas_resolution]
        )

    # for each mesh point, distance to nearest annotation
    diffs = mesh_pts[:, None, :] - ann_pts[None, :, :]
    distances = np.linalg.norm(diffs, axis=2).min(axis=1)

    p95 = np.percentile(distances, 95)

    assert p95 < THRESHOLD_UM, (
        f"{orientation}/{region}: "
        f"mesh contour points too far from annotation voxels. "
        f"p95={p95:.0f} µm, threshold={THRESHOLD_UM} µm"
    )


# Expects that Tuple orientation (1,0,0) produce same coords as 'frontal', etc.
@pytest.mark.parametrize("orientation", ["frontal", "horizontal", "sagittal"])
@pytest.mark.parametrize("region", REGIONS)
def test_tuple_orientation_matches_named(orientation, region):
    ORIENTATION_TO_TUPLE = {
        "frontal": (1, 0, 0),
        "horizontal": (0, 1, 0),
        "sagittal": (0, 0, 1),
    }

    tuple_coords = bgh.get_structures_slice_coords(
        [region],
        position=POSITION,
        orientation=ORIENTATION_TO_TUPLE[orientation],
    )

    named_coords = bgh.get_structures_slice_coords(
        [region], position=POSITION, orientation=orientation
    )

    tuple_pts = np.vstack(tuple_coords[region])
    named_pts = np.vstack(named_coords[region])

    # Points should be close (same contour, possibly different ordering)
    diffs = tuple_pts[None, :, :] - named_pts[:, None, :]
    distances = np.linalg.norm(diffs, axis=2).min(axis=1)
    max_dist = distances.max()

    assert max_dist < THRESHOLD_UM, (
        f"{orientation}/{region}: named vs tuple coords differ. "
        f"max_dist={max_dist:.0f} µm, threshold={THRESHOLD_UM} µm"
    )


# expects Z produce 2D spread on both axes by checking root.
# a root can't be this ${MIN_SPREAD_UM} thin
# can catch issues with norm
@pytest.mark.parametrize(
    "orientation", [(1, 1, 0), (1, 0, 1), (0, 1, 1), (0.3, 0.7, 1)]
)
def test_oblique_orientation_has_2d_spread(orientation):
    MIN_SPREAD_UM = 500

    coords = bgh.get_structures_slice_coords(
        REGIONS, position=POSITION, orientation=orientation
    )

    root_pts = np.vstack(coords["root"])
    spread = root_pts.max(axis=0) - root_pts.min(axis=0)

    assert spread[0] > MIN_SPREAD_UM, (
        f"orientation={orientation}: "
        f"spread={spread[0]:.0f} µm "
        f"(min {MIN_SPREAD_UM} µm expected)"
    )
    assert spread[1] > MIN_SPREAD_UM, (
        f"orientation={orientation}: "
        f"spread={spread[1]:.0f} µm "
        f"(min {MIN_SPREAD_UM} µm expected)"
    )


# Expects oblique 2D coords to be real atlas coords: recovering the dropped
# axis from the plane equation must land the point on the region in the
# annotation volume
@pytest.mark.parametrize("orientation", OBLIQUE_ORIENTATIONS)
@pytest.mark.parametrize("region", REGIONS)
def test_oblique_coords_match_annotation(atlas, orientation, region):
    from scipy.spatial import cKDTree

    # 2D (x, y) -> atlas axes (AP=0, DV=1, LR=2), by dominant axis
    PLANE_AXES = {0: (2, 1), 1: (2, 0), 2: (0, 1)}

    atlas_resolution = atlas.resolution[0]

    coords = bgh.get_structures_slice_coords(
        [region], position=POSITION, orientation=orientation
    )
    pts_2d = np.vstack(coords[region])

    # orientation is given in brainrender space (LR flipped)
    normal = np.array(orientation, dtype=float)
    normal[2] = -normal[2]
    dropped = int(np.argmax(np.abs(orientation)))
    x_axis, y_axis = PLANE_AXES[dropped]

    pts_3d = np.zeros((len(pts_2d), 3))
    pts_3d[:, x_axis] = pts_2d[:, 0]
    pts_3d[:, y_axis] = pts_2d[:, 1]
    pts_3d[:, dropped] = (
        normal @ np.array(POSITION, dtype=float)
        - normal[x_axis] * pts_2d[:, 0]
        - normal[y_axis] * pts_2d[:, 1]
    ) / normal[dropped]

    mask_voxels = np.argwhere(atlas.get_structure_mask(region) > 0)
    distances, _ = cKDTree(mask_voxels * atlas_resolution).query(pts_3d)
    p95 = np.percentile(distances, 95)

    assert p95 < THRESHOLD_UM, (
        f"{orientation}/{region}: "
        f"2D coords are not at the region's 3D atlas position. "
        f"p95={p95:.0f} µm, threshold={THRESHOLD_UM} µm"
    )


# Expects oblique coords to stay within the atlas, i.e. no negative axes
@pytest.mark.parametrize("orientation", OBLIQUE_ORIENTATIONS)
def test_oblique_coords_within_atlas_bounds(atlas, orientation):
    PLANE_AXES = {0: (2, 1), 1: (2, 0), 2: (0, 1)}

    coords = bgh.get_structures_slice_coords(
        REGIONS, position=POSITION, orientation=orientation
    )
    root_pts = np.vstack(coords["root"])

    extent = np.array(atlas.shape) * atlas.resolution[0]
    x_axis, y_axis = PLANE_AXES[int(np.argmax(np.abs(orientation)))]
    upper = extent[[x_axis, y_axis]]

    # root mesh can overshoot the volume by less than a voxel
    tolerance = atlas.resolution[0]
    assert (
        root_pts >= -tolerance
    ).all(), f"orientation={orientation}: min={root_pts.min(axis=0)}"
    assert (root_pts <= upper + tolerance).all(), (
        f"orientation={orientation}: max={root_pts.max(axis=0)}, "
        f"atlas extent={upper}"
    )
