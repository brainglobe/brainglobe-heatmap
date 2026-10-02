"""Hemisphere-specific heatmap values (issue #58)."""

import matplotlib.pyplot as plt
import pytest
from brainrender import settings

import brainglobe_heatmap as bgh

settings.INTERACTIVE = False
settings.OFFSCREEN = True

VALUES = {"TH": 1.0, "VISp": {"left": 0.8, "right": 0.2}}


@pytest.fixture
def heatmap():
    h = bgh.Heatmap(
        values=VALUES,
        position=9000,
        orientation="frontal",
        format="2D",
        label_regions=True,
        check_latest=False,
    )
    yield h
    h.scene.close()
    plt.close("all")


def test_one_actor_per_side(heatmap):
    names = {a.name for a in heatmap.regions_meshes}
    assert names == {"TH", "VISp__left", "VISp__right"}


def test_sides_match_atlas_hemisphere(heatmap):
    # brainrender renders with z negated; undo it to query the atlas
    for actor in heatmap.regions_meshes:
        if "__" not in actor.name:
            continue
        com = actor._mesh.vertices.mean(axis=0) * (1, 1, -1)
        side = heatmap.scene.atlas.hemisphere_from_coords(
            com, microns=True, as_string=True
        )
        assert actor.name.endswith(f"__{side}")


def test_colors_and_range(heatmap):
    assert heatmap.colors["VISp__left"] != heatmap.colors["VISp__right"]
    assert (heatmap.vmin, heatmap.vmax) == (0.2, 1.0)


def test_2d_plot_segments_and_colorbar_labels(heatmap):
    fig, ax = plt.subplots()
    heatmap.plot_subplot(fig=fig, ax=ax, show_legend=True)

    legend = {t.get_text() for t in ax.get_legend().get_texts()}
    assert {"VISp (left)", "VISp (right)"} <= legend

    labels = {t.get_text() for t in fig.axes[-1].get_yticklabels()}
    assert {"VISp (left)", "VISp (right)"} <= labels


def test_label_regions_list_matches_acronym(heatmap):
    heatmap.label_regions = ["VISp"]
    fig, ax = plt.subplots()
    heatmap.plot_subplot(fig=fig, ax=ax)
    labels = {t.get_text() for t in fig.axes[-1].get_yticklabels()}
    assert labels == {"VISp (left)", "VISp (right)"}


def test_3d_render_applies_alpha_to_both_sides():
    h = bgh.Heatmap(
        values=VALUES,
        position=9000,
        format="3D",
        alpha={"VISp": 0.4},
        interactive=False,
        check_latest=False,
    )
    h.show()
    actors = {a.name: a for a in h.regions_meshes}
    assert actors["VISp__left"].alpha() == pytest.approx(0.4)
    assert actors["VISp__right"].alpha() == pytest.approx(0.4)
    h.scene.close()
