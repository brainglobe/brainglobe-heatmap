"""Tests for Plane._join_reset( )"""

import numpy as np
import pytest
from brainrender.scene import Scene

from brainglobe_heatmap.plane import Plane
from brainglobe_heatmap.slicer import Slicer


@pytest.fixture(scope="module")
def allen_scene():
    return Scene(check_latest=False)


@pytest.fixture(scope="module")
def kim_scene():
    return Scene(atlas_name="kim_mouse_10um", check_latest=False)


# Current root broken position for kim
# ex: 7100, 9100, 9200
POSITION = 9100


def _get_piece(scene):
    slicer = Slicer(
        position=POSITION, orientation="frontal", thickness=10, root=scene.root
    )
    intersection = slicer.plane0.intersect_with(scene.root._mesh)
    return intersection.split()[0]


# asserts that _join_reset VS join(reset=True)
# produces same result on simple cases
def test_join_reset_matches_vedo_join(allen_scene):
    piece = _get_piece(allen_scene)

    ours = Plane._join_reset(piece)
    vedos = piece.join(reset=True).vertices

    np.testing.assert_array_equal(ours, vedos)


# asserts that vedo's join(reset=True) loses data
# due to its default SetMaximumLength
# particularly testing kim_10um ${POSITION}
def test_join_reset_handles_kim_root(kim_scene):
    piece = _get_piece(kim_scene)

    ours = Plane._join_reset(piece)
    vedos = piece.join(reset=True).vertices

    # If the test fails maybe would need to
    # hard test a different position
    # or no need to increase MaximumLength anymore
    assert ours.shape[0] > vedos.shape[0]
