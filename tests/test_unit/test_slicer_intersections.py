"""Tests Slicer.get_structures_slice_coords( ) VS Plane.get_projections( )"""

import numpy as np
import vedo as vd
from brainrender.actor import Actor

from brainglobe_heatmap.slicer import Slicer


def _get_root(res):
    sphere = vd.Sphere(r=1, res=res)
    cells = sphere.cells
    np.random.RandomState(66).shuffle(cells)
    mesh = vd.Mesh([sphere.vertices, cells])
    actor = Actor(mesh, name="root", br_class="brain region")
    actor._mesh = mesh  # normally set by brainrender's Scene on render
    return actor


# asserts that the Slicer contours go through the same intersection
# logic as Plane.get_projections( ), e.g. Plane._join_reset( ), so
# that large contours are not truncated
def test_slicer_contours_match_plane_projections():
    root = _get_root(res=1000)
    slicer = Slicer((0, 0, 0), "sagittal", 1, root)

    projected, _ = slicer.get_structures_slice_coords([], root)
    plane_projected = slicer.plane0.get_projections([root])

    assert projected.keys() == plane_projected.keys()
    for key, points in projected.items():
        assert points.shape == plane_projected[key].shape, key
