"""Tests for Plane._join_reset( )"""

import numpy as np
import vedo as vd

from brainglobe_heatmap.plane import Plane


def _get_piece(res):
    sphere = vd.Sphere(r=1, res=res)
    cells = sphere.cells
    np.random.RandomState(66).shuffle(cells)
    mesh = vd.Mesh([sphere.vertices, cells])

    plane = Plane(
        origin=np.array([0.0, 0.0, 0.0]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
    )
    intersection = plane.intersect_with(mesh)
    return intersection.split()[0]


# asserts that _join_reset VS join(reset=True)
# produces same result on simple cases
def test_join_reset_matches_vedo_join():
    piece = _get_piece(res=100)

    _join_reset = Plane._join_reset(piece).vertices
    vedos = piece.join(reset=True).vertices

    np.testing.assert_array_equal(_join_reset, vedos)


# asserts that vedo's join(reset=True) loses data
# due to its default SetMaximumLength
def test_join_reset_handles_kim_root():
    piece = _get_piece(res=1000)

    _join_reset = Plane._join_reset(piece).vertices
    vedos = piece.join(reset=True).vertices

    # If the test fails maybe
    # there is no need to increase MaximumLength anymore
    assert _join_reset.shape[0] > vedos.shape[0]
