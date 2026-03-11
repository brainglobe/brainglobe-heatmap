from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from brainrender import settings

import brainglobe_heatmap as bgh

settings.INTERACTIVE = False
settings.OFFSCREEN = True

VALUES = {
    "TH": 1,
    "HIP": 3,
    "VIS": 2,
    "PA": -4,
}


@pytest.fixture
def mock_projected():
    """Fixture providing mock projected slice data with TH, HIP, and VIS visible."""
    return {
        "TH_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        "TH_segment_1": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        "HIP_segment_1": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        "VIS_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        "VIS_segment_1": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        "root_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    }


@pytest.fixture
def heatmap_2d():
    """Fixture for a standard 2D heatmap."""
    heatmap = bgh.Heatmap(
        VALUES,
        format="2D",
        position=1000,
        orientation="frontal",
        vmin=-5,
        vmax=3,
        check_latest=False,
        interactive=False,
    )
    yield heatmap
    heatmap.scene.close()


def _show_with_mocks(heatmap, mock_projected, **show_kwargs):
    """Helper to call heatmap.show() with slicer and matplotlib mocked."""
    mock_colorbar = MagicMock()
    with (
        patch.object(
            heatmap.slicer,
            "get_structures_slice_coords",
            return_value=(mock_projected, None),
        ),
        patch(
            "matplotlib.figure.Figure.colorbar", return_value=mock_colorbar
        ) as mock_colorbar_fn,
        patch("matplotlib.pyplot.show"),
    ):
        heatmap.show(**show_kwargs)
    return mock_colorbar_fn, mock_colorbar


# --- Colorbar visibility tests ---


def test_colorbar_hidden_when_show_cbar_false(heatmap_2d, mock_projected):
    """Colorbar must not be created when show_cbar=False."""
    mock_colorbar_fn, _ = _show_with_mocks(
        heatmap_2d, mock_projected, show_cbar=False
    )
    assert mock_colorbar_fn.call_count == 0


def test_colorbar_shown_when_show_cbar_true(heatmap_2d, mock_projected):
    """Colorbar must be created when show_cbar=True."""
    mock_colorbar_fn, _ = _show_with_mocks(
        heatmap_2d, mock_projected, show_cbar=True
    )
    assert mock_colorbar_fn.call_count == 1


def test_show_cbar_false_overrides_label_regions(heatmap_2d, mock_projected):
    """show_cbar=False must suppress colorbar even when label_regions=True."""
    heatmap_2d.label_regions = True
    mock_colorbar_fn, mock_colorbar = _show_with_mocks(
        heatmap_2d, mock_projected, show_cbar=False
    )
    assert mock_colorbar_fn.call_count == 0
    assert mock_colorbar.set_ticks.call_count == 0


# --- Region label tests ---


def test_no_region_labels_when_label_regions_false(heatmap_2d, mock_projected):
    """set_ticks must not be called when label_regions=False."""
    heatmap_2d.label_regions = False
    _, mock_colorbar = _show_with_mocks(
        heatmap_2d, mock_projected, show_cbar=True
    )
    assert mock_colorbar.set_ticks.call_count == 0


def test_region_labels_shown_when_label_regions_true(
    heatmap_2d, mock_projected
):
    """set_ticks must be called once when label_regions=True and show_cbar=True."""
    heatmap_2d.label_regions = True
    _, mock_colorbar = _show_with_mocks(
        heatmap_2d, mock_projected, show_cbar=True
    )
    assert mock_colorbar.set_ticks.call_count == 1


def test_region_labels_match_visible_regions(heatmap_2d, mock_projected):
    """Colorbar labels must only include visible regions within vmin/vmax range."""
    heatmap_2d.label_regions = True
    _, mock_colorbar = _show_with_mocks(
        heatmap_2d, mock_projected, show_cbar=True
    )
    _, kwargs = mock_colorbar.set_ticks.call_args
    region_labels = kwargs["labels"]
    tick_values = kwargs["ticks"]

    visible_regions = {"TH", "HIP", "VIS"}  # from mock_projected, excluding root
    expected_regions = {
        r for r in visible_regions
        if heatmap_2d.vmin <= VALUES[r] <= heatmap_2d.vmax
    }

    assert set(region_labels) == expected_regions
    assert "root" not in region_labels
    assert len(tick_values) == len(expected_regions)


def test_region_tick_values_match_region_values(heatmap_2d, mock_projected):
    """Each colorbar tick value must match the corresponding region's value."""
    heatmap_2d.label_regions = True
    _, mock_colorbar = _show_with_mocks(
        heatmap_2d, mock_projected, show_cbar=True
    )
    _, kwargs = mock_colorbar.set_ticks.call_args
    region_labels = kwargs["labels"]
    tick_values = kwargs["ticks"]

    for i, region in enumerate(region_labels):
        assert tick_values[i] == VALUES[region]


def test_region_tick_values_within_vmin_vmax(heatmap_2d, mock_projected):
    """All colorbar tick values must be within vmin/vmax range."""
    heatmap_2d.label_regions = True
    _, mock_colorbar = _show_with_mocks(
        heatmap_2d, mock_projected, show_cbar=True
    )
    _, kwargs = mock_colorbar.set_ticks.call_args
    tick_values = kwargs["ticks"]

    for val in tick_values:
        assert heatmap_2d.vmin <= val <= heatmap_2d.vmax


# --- cbar_label tests ---


def test_cbar_label_set_when_provided(heatmap_2d, mock_projected):
    """set_label must be called when cbar_label is provided."""
    _, mock_colorbar = _show_with_mocks(
        heatmap_2d, mock_projected, show_cbar=True, cbar_label="Test Label"
    )
    assert mock_colorbar.set_label.call_count == 1


def test_cbar_label_not_set_when_not_provided(heatmap_2d, mock_projected):
    """set_label must not be called when cbar_label is not provided."""
    _, mock_colorbar = _show_with_mocks(
        heatmap_2d, mock_projected, show_cbar=True
    )
    assert mock_colorbar.set_label.call_count == 0


def test_cbar_label_hidden_when_show_cbar_false(heatmap_2d, mock_projected):
    """set_label must not be called when show_cbar=False even with cbar_label set."""
    _, mock_colorbar = _show_with_mocks(
        heatmap_2d,
        mock_projected,
        show_cbar=False,
        cbar_label="Test Label",
    )
    assert mock_colorbar.set_label.call_count == 0


# --- Edge case tests ---


def test_colorbar_empty_when_no_visible_regions(heatmap_2d):
    """Colorbar ticks must be empty when no regions are visible."""
    heatmap_2d.label_regions = True
    _, mock_colorbar = _show_with_mocks(heatmap_2d, {}, show_cbar=True)
    _, kwargs = mock_colorbar.set_ticks.call_args
    assert len(kwargs["ticks"]) == 0
    assert len(kwargs["labels"]) == 0


def test_colorbar_empty_when_only_root_visible(heatmap_2d):
    """Colorbar ticks must be empty when only root segment is visible."""
    heatmap_2d.label_regions = True
    projected = {
        "root_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    }
    _, mock_colorbar = _show_with_mocks(heatmap_2d, projected, show_cbar=True)
    _, kwargs = mock_colorbar.set_ticks.call_args
    assert len(kwargs["ticks"]) == 0
    assert len(kwargs["labels"]) == 0


def test_colorbar_single_region(heatmap_2d):
    """Colorbar must show exactly one tick when one region is visible."""
    heatmap_2d.label_regions = True
    heatmap_2d.values = {"TH": 1.0}
    projected = {
        "TH_segment_1": np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    }
    _, mock_colorbar = _show_with_mocks(heatmap_2d, projected, show_cbar=True)
    _, kwargs = mock_colorbar.set_ticks.call_args
    assert len(kwargs["ticks"]) == 1
    assert len(kwargs["labels"]) == 1
    assert kwargs["labels"][0] == "TH"
    assert kwargs["ticks"][0] == 1.0


def test_colorbar_empty_when_all_values_outside_range(heatmap_2d):
    """Colorbar ticks must be empty when all region values are outside vmin/vmax."""
    heatmap_2d.label_regions = True
    heatmap_2d.values = {"TH": -6.2, "VIS": 4.1}
    projected = {
        "TH_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        "VIS_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    }
    _, mock_colorbar = _show_with_mocks(heatmap_2d, projected, show_cbar=True)
    _, kwargs = mock_colorbar.set_ticks.call_args
    assert len(kwargs["ticks"]) == 0
    assert len(kwargs["labels"]) == 0


def test_colorbar_only_in_range_regions_shown(heatmap_2d):
    """Colorbar must only show regions whose values are within vmin/vmax."""
    heatmap_2d.label_regions = True
    heatmap_2d.values = {"TH": -6.2, "VIS": 1.1, "HIP": 5}
    projected = {
        "TH_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        "HIP_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        "VIS_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    }
    _, mock_colorbar = _show_with_mocks(heatmap_2d, projected, show_cbar=True)
    _, kwargs = mock_colorbar.set_ticks.call_args
    assert kwargs["labels"] == ["VIS"]
    assert kwargs["ticks"][0] == 1.1
