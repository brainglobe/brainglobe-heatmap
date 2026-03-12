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
    # In Reds cmap the R channel is NOT monotonic with value —
    # darker reds (high value) have lower R.
    # left=0.8 -> R=0.40, right=0.2 -> R=1.0
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


# ---------------------------------------------------------------------------
# hemisphere="left" / "right" for bilateral scalar regions
# (covers bilateral hemisphere cutting via brainrender hemisphere= param)
# ---------------------------------------------------------------------------


@pytest.fixture
def heatmap_bilateral_left_hemisphere():
    h = bgh.Heatmap(
        values={"TH": 1.0, "VISp": 0.5},
        position=7000,
        orientation="frontal",
        hemisphere="left",
        format="2D",
        check_latest=False,
    )
    yield h
    h.scene.close()


@pytest.fixture
def heatmap_bilateral_right_hemisphere():
    h = bgh.Heatmap(
        values={"TH": 1.0},
        position=7000,
        orientation="frontal",
        hemisphere="right",
        format="2D",
        check_latest=False,
    )
    yield h
    h.scene.close()


def test_bilateral_hemisphere_left_cuts_mesh(
    heatmap_bilateral_left_hemisphere,
):
    """hemisphere='left' should cut bilateral actors to the left side only."""
    h = heatmap_bilateral_left_hemisphere
    bounds = h.scene.root._mesh.bounds()
    import numpy as np

    mid_z = np.array(bounds).reshape(3, 2).mean(axis=1)[2]
    for actor in h.regions_meshes:
        verts = actor._mesh.vertices
        # Left hemisphere: z-axis points Right in ASR space,
        # so left = z <= mid_z
        assert (
            verts[:, 2].max() <= mid_z + 1e-3
        ), f"Actor {actor.name} has vertices on the right side after left cut"


def test_bilateral_hemisphere_right_cuts_mesh(
    heatmap_bilateral_right_hemisphere,
):
    """hemisphere='right' should cut bilateral actors to the right
    side only."""
    h = heatmap_bilateral_right_hemisphere
    bounds = h.scene.root._mesh.bounds()
    import numpy as np

    mid_z = np.array(bounds).reshape(3, 2).mean(axis=1)[2]
    for actor in h.regions_meshes:
        verts = actor._mesh.vertices
        # Right hemisphere: z-axis points Right in ASR space,
        # so right = z >= mid_z
        assert (
            verts[:, 2].min() >= mid_z - 1e-3
        ), f"Actor {actor.name} has vertices on the left side after right cut"


def test_bilateral_left_hemisphere_colors_assigned(
    heatmap_bilateral_left_hemisphere,
):
    """actor_colors should be populated correctly for
    hemisphere='left' bilateral."""
    h = heatmap_bilateral_left_hemisphere
    assert len(h.actor_colors) == len(h.regions_meshes)
    for actor, color in h.actor_colors.items():
        assert color is not None
        assert "__" not in actor.name


# ---------------------------------------------------------------------------
# plot() standalone method and plot_subplot() branches
# (covers lines 415-430, 520, 534, 536, 546-550, 552)
# ---------------------------------------------------------------------------


@pytest.fixture
def heatmap_for_plot():
    h = bgh.Heatmap(
        values={"TH": 1.0, "VISp": 0.5},
        position=7000,
        orientation="frontal",
        format="2D",
        check_latest=False,
    )
    yield h
    h.scene.close()


def test_plot_returns_figure(heatmap_for_plot):
    """plot() should return a matplotlib Figure."""
    import matplotlib.pyplot as plt

    fig = heatmap_for_plot.plot(show_cbar=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_saves_file(heatmap_for_plot, tmp_path):
    """plot() with filename= should write a file to disk."""
    import matplotlib.pyplot as plt

    out = tmp_path / "heatmap.png"
    heatmap_for_plot.plot(filename=str(out), show_cbar=False)
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close("all")


def test_plot_subplot_label_regions():
    """label_regions=True should produce a colorbar with correct tick count."""
    import matplotlib.pyplot as plt

    h = bgh.Heatmap(
        values={"TH": 1.0, "VISp": 0.5},
        position=7000,
        orientation="frontal",
        format="2D",
        label_regions=True,
        check_latest=False,
    )
    fig, ax = plt.subplots()
    fig, ax = h.plot_subplot(fig=fig, ax=ax, show_cbar=True)
    # Two regions -> two ticks on the colorbar
    cbar_ax = fig.axes[-1]
    labels = [t.get_text() for t in cbar_ax.get_yticklabels() if t.get_text()]
    assert labels == ["TH", "VISp"]
    plt.close("all")
    h.scene.close()


def test_plot_subplot_label_regions_per_hemisphere():
    """label_regions=True with per-hemisphere values uses left:/right: keys."""
    import matplotlib.pyplot as plt

    h = bgh.Heatmap(
        values={"TH": {"left": 0.8, "right": 0.2}},
        position=7000,
        orientation="frontal",
        format="2D",
        label_regions=True,
        check_latest=False,
    )
    fig, ax = plt.subplots()
    fig, ax = h.plot_subplot(fig=fig, ax=ax, show_cbar=True)
    cbar_ax = fig.axes[-1]
    # Two entries: left:TH and right:TH
    labels = [t.get_text() for t in cbar_ax.get_yticklabels() if t.get_text()]
    assert labels == ["left:TH", "right:TH"]
    plt.close("all")
    h.scene.close()


def test_plot_subplot_cbar_label(heatmap_for_plot):
    """cbar_label= should set the colorbar label text."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig, ax = heatmap_for_plot.plot_subplot(
        fig=fig, ax=ax, show_cbar=True, cbar_label="my label"
    )
    cbar_ax = fig.axes[-1]
    assert cbar_ax.get_ylabel() == "my label"
    plt.close("all")


def test_plot_subplot_hide_axes(heatmap_for_plot):
    """hide_axes=True should remove axis ticks and labels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig, ax = heatmap_for_plot.plot_subplot(
        fig=fig, ax=ax, show_cbar=False, hide_axes=True
    )
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_xticks().size == 0
    assert ax.get_yticks().size == 0
    plt.close("all")


def test_plot_subplot_show_legend(heatmap_for_plot):
    """show_legend=True should attach a legend to the axes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig, ax = heatmap_for_plot.plot_subplot(
        fig=fig, ax=ax, show_cbar=False, show_legend=True
    )
    assert ax.get_legend() is not None
    plt.close("all")


# ---------------------------------------------------------------------------
# check_values error paths (lines 70, 74, 81, 88)
# ---------------------------------------------------------------------------


def test_check_values_unrecognized_region():
    """Passing an unknown region acronym should raise ValueError."""
    with pytest.raises(ValueError, match="not recognized"):
        bgh.Heatmap(
            values={"NOTAREGION": 1.0},
            position=7000,
            orientation="frontal",
            format="2D",
            check_latest=False,
        )


def test_check_values_wrong_type_bilateral():
    """Passing a string as a bilateral value should raise ValueError."""
    with pytest.raises(ValueError, match="should be floats"):
        bgh.Heatmap(
            values={"TH": "high"},
            position=7000,
            orientation="frontal",
            format="2D",
            check_latest=False,
        )


def test_check_values_wrong_type_per_hemisphere():
    """Passing a string inside a per-hemisphere dict should raise
    ValueError."""
    with pytest.raises(ValueError, match="should be floats"):
        bgh.Heatmap(
            values={"TH": {"left": "high", "right": 0.2}},
            position=7000,
            orientation="frontal",
            format="2D",
            check_latest=False,
        )


def test_check_values_all_nan():
    """All-NaN values should produce vmin=vmax=NaN without crashing."""
    import math

    h = bgh.Heatmap(
        values={"TH": float("nan")},
        position=7000,
        orientation="frontal",
        format="2D",
        check_latest=False,
    )
    assert math.isnan(h.vmin)
    assert math.isnan(h.vmax)
    h.scene.close()


# ---------------------------------------------------------------------------
# show() 2D path (line 369)
# ---------------------------------------------------------------------------


def test_show_2d_returns_figure(heatmap_for_plot):
    """show() in 2D format should return a matplotlib Figure."""
    import matplotlib.pyplot as plt

    result = heatmap_for_plot.show()
    assert isinstance(result, plt.Figure)
    plt.close("all")
