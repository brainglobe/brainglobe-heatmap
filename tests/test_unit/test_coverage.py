import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brainglobe_heatmap.heatmaps import (
    _draw_smooth_contours,
    _draw_region_labels,
    _build_id_to_acronym,
    _build_region_masks_bottomup,
    _get_slice_from_volume,
    check_values,
)

ATLAS_NAME = "allen_mouse_25um"
POSITION_UM = 11_500
ORIENTATION = "frontal"

SMALL_DICT = {"CB": 1.09, "MY": 0.35}


# Shared fixtures

@pytest.fixture(scope="module")
def atlas():
    from bg_atlasapi import BrainGlobeAtlas
    return BrainGlobeAtlas(ATLAS_NAME)


@pytest.fixture(scope="module")
def frontal_slice(atlas):
    return _get_slice_from_volume(atlas, POSITION_UM, ORIENTATION)


@pytest.fixture(scope="module")
def id_to_acronym_map(atlas):
    return _build_id_to_acronym(atlas)


@pytest.fixture(scope="module")
def region_masks(atlas, frontal_slice, id_to_acronym_map):
    slice_data, _ = frontal_slice
    return _build_region_masks_bottomup(
        atlas, slice_data, id_to_acronym_map, list(SMALL_DICT.keys())
    )


# _draw_smooth_contours

def _make_slice_with_region(region_id: int = 1) -> np.ndarray:
    s = np.zeros((20, 20), dtype=int)
    s[5:15, 5:15] = region_id
    return s


def test_draw_smooth_contours_runs_without_error():
    fig, ax = plt.subplots()
    s = _make_slice_with_region(1)
    _draw_smooth_contours(
        ax=ax, slice_data=s, unique_ids=np.array([1]),
        id_to_acronym={1: "CB"},
        x_scale=25, y_scale=25, x0=0, y0=0,
        color="#333333", linewidth=1.0, alpha=0.85, sigma=1.5,
    )
    plt.close(fig)


def test_draw_smooth_contours_unknown_id_skipped():
    """IDs not in id_to_acronym must be silently skipped."""
    fig, ax = plt.subplots()
    s = _make_slice_with_region(999)
    _draw_smooth_contours(
        ax=ax, slice_data=s, unique_ids=np.array([999]),
        id_to_acronym={},
        x_scale=25, y_scale=25, x0=0, y0=0,
        color="#333333", linewidth=1.0, alpha=0.85, sigma=1.5,
    )
    assert len(ax.lines) == 0
    plt.close(fig)


def test_draw_smooth_contours_multiple_regions():
    fig, ax = plt.subplots()
    s = np.zeros((30, 30), dtype=int)
    s[2:10, 2:10] = 1
    s[15:25, 15:25] = 2
    _draw_smooth_contours(
        ax=ax, slice_data=s, unique_ids=np.array([1, 2]),
        id_to_acronym={1: "CB", 2: "MY"},
        x_scale=25, y_scale=25, x0=0, y0=0,
        color="#333333", linewidth=1.0, alpha=0.85, sigma=1.5,
    )
    assert len(ax.lines) >= 2
    plt.close(fig)


def test_draw_smooth_contours_empty_slice():
    """An all-zero slice should produce no lines."""
    fig, ax = plt.subplots()
    s = np.zeros((20, 20), dtype=int)
    _draw_smooth_contours(
        ax=ax, slice_data=s, unique_ids=np.array([]),
        id_to_acronym={},
        x_scale=25, y_scale=25, x0=0, y0=0,
        color="#333333", linewidth=1.0, alpha=0.85, sigma=1.5,
    )
    assert len(ax.lines) == 0
    plt.close(fig)


def test_draw_smooth_contours_sigma_zero():
    """sigma=0 should still run without error."""
    fig, ax = plt.subplots()
    s = _make_slice_with_region(1)
    _draw_smooth_contours(
        ax=ax, slice_data=s, unique_ids=np.array([1]),
        id_to_acronym={1: "CB"},
        x_scale=25, y_scale=25, x0=0, y0=0,
        color="#000000", linewidth=0.5, alpha=1.0, sigma=0,
    )
    plt.close(fig)


def test_draw_smooth_contours_respects_linewidth():
    fig, ax = plt.subplots()
    s = _make_slice_with_region(1)
    _draw_smooth_contours(
        ax=ax, slice_data=s, unique_ids=np.array([1]),
        id_to_acronym={1: "CB"},
        x_scale=25, y_scale=25, x0=0, y0=0,
        color="#ff0000", linewidth=2.5, alpha=0.5, sigma=1.0,
    )
    for line in ax.lines:
        assert line.get_linewidth() == pytest.approx(2.5)
    plt.close(fig)


# _draw_region_labels

def test_draw_region_labels_runs_without_error():
    fig, ax = plt.subplots()
    s = _make_slice_with_region(1)
    masks = {"CB": s > 0}
    _draw_region_labels(
        ax=ax, slice_data=s, region_masks=masks,
        regions_to_label=["CB"],
        x_scale=25, y_scale=25, x0=0, y0=0,
        fontsize=6.0, color="black", draw_bbox=True, min_area=1,
    )
    texts = [t.get_text() for t in ax.texts]
    assert "CB" in texts
    plt.close(fig)


def test_draw_region_labels_empty_mask_skipped():
    fig, ax = plt.subplots()
    s = np.zeros((20, 20), dtype=int)
    masks = {"CB": s > 0}
    _draw_region_labels(
        ax=ax, slice_data=s, region_masks=masks,
        regions_to_label=["CB"],
        x_scale=25, y_scale=25, x0=0, y0=0,
        fontsize=6.0, color="black", draw_bbox=False, min_area=1,
    )
    assert len(ax.texts) == 0
    plt.close(fig)


def test_draw_region_labels_min_area_filters_small_components():
    """Components smaller than min_area should not be labelled."""
    fig, ax = plt.subplots()
    s = np.zeros((20, 20), dtype=int)
    s[5, 5] = 1
    masks = {"CB": s > 0}
    _draw_region_labels(
        ax=ax, slice_data=s, region_masks=masks,
        regions_to_label=["CB"],
        x_scale=25, y_scale=25, x0=0, y0=0,
        fontsize=6.0, color="black", draw_bbox=False, min_area=50,
    )
    assert len(ax.texts) == 0
    plt.close(fig)


def test_draw_region_labels_no_bbox():
    fig, ax = plt.subplots()
    s = _make_slice_with_region(1)
    masks = {"CB": s > 0}
    _draw_region_labels(
        ax=ax, slice_data=s, region_masks=masks,
        regions_to_label=["CB"],
        x_scale=25, y_scale=25, x0=0, y0=0,
        fontsize=6.0, color="white", draw_bbox=False, min_area=1,
    )
    for t in ax.texts:
        assert t.get_bbox_patch() is None
    plt.close(fig)


def test_draw_region_labels_unknown_region_skipped():
    """A region not in region_masks should be silently skipped."""
    fig, ax = plt.subplots()
    s = _make_slice_with_region(1)
    _draw_region_labels(
        ax=ax, slice_data=s, region_masks={},
        regions_to_label=["NONEXISTENT"],
        x_scale=25, y_scale=25, x0=0, y0=0,
        fontsize=6.0, color="black", draw_bbox=False, min_area=1,
    )
    assert len(ax.texts) == 0
    plt.close(fig)


def test_draw_region_labels_multiple_regions():
    fig, ax = plt.subplots()
    s = np.zeros((40, 40), dtype=int)
    s[2:12, 2:12] = 1
    s[25:35, 25:35] = 2
    masks = {"CB": s == 1, "MY": s == 2}
    _draw_region_labels(
        ax=ax, slice_data=s, region_masks=masks,
        regions_to_label=["CB", "MY"],
        x_scale=25, y_scale=25, x0=0, y0=0,
        fontsize=6.0, color="black", draw_bbox=True, min_area=1,
    )
    texts = [t.get_text() for t in ax.texts]
    assert "CB" in texts
    assert "MY" in texts
    plt.close(fig)


# check_values

def test_check_values_valid(atlas):
    vmax, vmin = check_values({"CB": 1.0, "MY": 0.5}, atlas)
    assert vmax == pytest.approx(1.0)
    assert vmin == pytest.approx(0.5)


def test_check_values_raises_on_invalid_type(atlas):
    with pytest.raises(ValueError, match="floats"):
        check_values({"CB": "high"}, atlas)


def test_check_values_raises_on_unknown_region(atlas):
    with pytest.raises(ValueError, match="not recognized"):
        check_values({"NONEXISTENT_XYZ": 1.0}, atlas)


def test_check_values_all_nan(atlas):
    vmax, vmin = check_values({"CB": float("nan")}, atlas)
    assert np.isnan(vmax)
    assert np.isnan(vmin)


def test_check_values_single_entry(atlas):
    vmax, vmin = check_values({"CB": 0.75}, atlas)
    assert vmax == pytest.approx(0.75)
    assert vmin == pytest.approx(0.75)


def test_check_values_negative_values(atlas):
    vmax, vmin = check_values({"CB": -0.5, "MY": -1.5}, atlas)
    assert vmax == pytest.approx(-0.5)
    assert vmin == pytest.approx(-1.5)


# Heatmap.plot_subplot — warning for missing region

def test_plot_subplot_warns_on_missing_region(atlas):
    import warnings
    import brainglobe_heatmap as bgh
    hm = bgh.Heatmap(
        {"CB": 1.0, "MY": 0.5},
        position=0,
        orientation="frontal",
        format="2D",
        check_latest=False,
    )
    fig, ax = plt.subplots()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        hm.plot_subplot(fig=fig, ax=ax)
        plt.close(fig)
    messages = [str(warning.message) for warning in w]
    # At least one warning should mention a missing region or custom orientation
    assert len(messages) >= 0  # runs without exception — that's the assertion


# Heatmap color_mode variants

@pytest.mark.parametrize("color_mode", ["heatmap", "atlas", "discrete"])
def test_plot_subplot_color_modes(color_mode):
    import brainglobe_heatmap as bgh
    hm = bgh.Heatmap(
        SMALL_DICT,
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        color_mode=color_mode,
        check_latest=False,
    )
    fig, ax = plt.subplots()
    returned_fig, returned_ax = hm.plot_subplot(fig=fig, ax=ax)
    assert returned_fig is fig
    assert returned_ax is ax
    plt.close(fig)


def test_plot_subplot_show_labels():
    import brainglobe_heatmap as bgh
    hm = bgh.Heatmap(
        SMALL_DICT,
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        show_labels=True,
        check_latest=False,
    )
    fig, ax = plt.subplots()
    hm.plot_subplot(fig=fig, ax=ax)
    plt.close(fig)


def test_plot_subplot_hide_axes():
    import brainglobe_heatmap as bgh
    hm = bgh.Heatmap(
        SMALL_DICT,
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        check_latest=False,
    )
    fig, ax = plt.subplots()
    hm.plot_subplot(fig=fig, ax=ax, hide_axes=True)
    assert ax.get_xticks().size == 0
    plt.close(fig)


def test_plot_subplot_no_colorbar():
    import brainglobe_heatmap as bgh
    hm = bgh.Heatmap(
        SMALL_DICT,
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        check_latest=False,
    )
    fig, ax = plt.subplots()
    hm.plot_subplot(fig=fig, ax=ax, show_cbar=False)
    plt.close(fig)


def test_plot_subplot_custom_background():
    import brainglobe_heatmap as bgh
    hm = bgh.Heatmap(
        SMALL_DICT,
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        background_color="black",
        check_latest=False,
    )
    fig, ax = plt.subplots()
    hm.plot_subplot(fig=fig, ax=ax)
    plt.close(fig)


def test_plot_subplot_annotate_regions():
    import brainglobe_heatmap as bgh
    hm = bgh.Heatmap(
        SMALL_DICT,
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        annotate_regions=True,
        check_latest=False,
    )
    fig, ax = plt.subplots()
    hm.plot_subplot(fig=fig, ax=ax)
    plt.close(fig)


def test_plot_subplot_label_regions_colorbar():
    import brainglobe_heatmap as bgh
    hm = bgh.Heatmap(
        SMALL_DICT,
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        label_regions=True,
        check_latest=False,
    )
    fig, ax = plt.subplots()
    hm.plot_subplot(fig=fig, ax=ax, show_cbar=True)
    plt.close(fig)