"""
Integration tests for hemisphere-specific heatmap values (issue #58).

Users can now pass a dict with "left"/"right" keys for any region
to visualise different values in each hemisphere.
"""

import matplotlib as mpl
import pytest
from brainrender import settings

import brainglobe_heatmap as bgh

settings.INTERACTIVE = False
settings.OFFSCREEN = True
mpl.use("Agg")


@pytest.fixture
def heatmap_both_hemispheres():
    h = bgh.Heatmap(
        values={"TH": {"left": 0.8, "right": 0.2}},
        position=7000,
        orientation="frontal",
        format="2D",
        check_latest=False,
    )
    yield h
    h.scene.close()


@pytest.fixture
def heatmap_left_only():
    h = bgh.Heatmap(
        values={"TH": {"left": 0.5}},
        position=7000,
        orientation="frontal",
        format="2D",
        check_latest=False,
    )
    yield h
    h.scene.close()


@pytest.fixture
def heatmap_mixed():
    h = bgh.Heatmap(
        values={
            "TH": 1.0,
            "VISp": {"left": 0.8, "right": 0.2},
        },
        position=7000,
        orientation="frontal",
        format="2D",
        check_latest=False,
    )
    yield h
    h.scene.close()


def test_both_hemispheres_produces_two_actors(heatmap_both_hemispheres):
    """Two actors should be created, one per hemisphere."""
    th_actors = [
        a for a in heatmap_both_hemispheres.regions_meshes if "TH" in a.name
    ]
    assert len(th_actors) == 2
    names = {a.name for a in th_actors}
    assert names == {"TH__left", "TH__right"}


def test_both_hemispheres_different_colors(heatmap_both_hemispheres):
    """Left and right actors should have different colors."""
    color_by_name = {
        a.name: c for a, c in heatmap_both_hemispheres.actor_colors.items()
    }
    assert color_by_name["TH__left"] != color_by_name["TH__right"]
    # left=0.8 -> darker (lower red channel in Reds cmap)
    # right=0.2 -> lighter (higher red channel)
    assert color_by_name["TH__left"][0] < color_by_name["TH__right"][0]


def test_left_only_produces_one_actor(heatmap_left_only):
    """Only one actor should be created when only left is specified."""
    th_actors = [a for a in heatmap_left_only.regions_meshes if "TH" in a.name]
    assert len(th_actors) == 1
    assert th_actors[0].name == "TH__left"


def test_mixed_bilateral_and_per_hemisphere(heatmap_mixed):
    """Bilateral regions keep plain names; per-hemisphere get __side suffix."""
    names = {a.name for a in heatmap_mixed.regions_meshes}
    assert "TH" in names
    assert "VISp__left" in names
    assert "VISp__right" in names
    assert "TH__left" not in names
    assert "TH__right" not in names


def test_vmin_vmax_spans_all_values(heatmap_mixed):
    """vmin/vmax should be computed across bilateral and per-hemisphere."""
    assert heatmap_mixed.vmin == 0.2
    assert heatmap_mixed.vmax == 1.0


def test_2d_projection_has_unique_segment_keys(heatmap_both_hemispheres):
    """Projected segments should have unique keys for each hemisphere."""
    projected, _ = heatmap_both_hemispheres.slicer.get_structures_slice_coords(
        heatmap_both_hemispheres.regions_meshes,
        heatmap_both_hemispheres.scene.root,
    )
    th_keys = [k for k in projected.keys() if "TH" in k]
    left_keys = [k for k in th_keys if "TH__left" in k]
    right_keys = [k for k in th_keys if "TH__right" in k]
    assert len(left_keys) >= 1
    assert len(right_keys) >= 1


def test_backwards_compatible_scalar_values():
    """Scalar-only values should work exactly as before."""
    h = bgh.Heatmap(
        values={"TH": 1.0, "VISp": 0.5},
        position=7000,
        orientation="frontal",
        format="2D",
        check_latest=False,
    )
    names = {a.name for a in h.regions_meshes}
    assert "TH" in names
    assert "VISp" in names
    assert all("__" not in n for n in names)
    h.scene.close()
