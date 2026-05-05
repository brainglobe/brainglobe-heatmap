import warnings
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from brainrender import settings

import brainglobe_heatmap as bgh

settings.INTERACTIVE = False
settings.OFFSCREEN = True

mpl.use("Agg")


# Constants

POSITION_UM = 11_500
ORIENTATION = "frontal"

EXAMPLE_VALUES = {
    "TH": 1,
    "RSP": 0.2,
    "AI": 0.4,
    "SS": -3,
    "MO": 2.6,
    "PVZ": -4,
    "LZ": -3,
    "VIS": 2,
    "AUD": 0.3,
    "RHP": -0.2,
    "STR": 0.5,
    "CB": 0.5,
    "FRP": -1.7,
    "HIP": 3,
    "PA": -4,
}

# Regions confirmed to have pixels in the frontal slice at POSITION_UM.
# Verified empirically: only CB is present at this position among the
# regions in EXAMPLE_VALUES.
EXPECTED_VISIBLE_REGIONS = ["CB"]

EXPECTED_ABSENT_REGIONS = ["root"] + [
    r for r in EXAMPLE_VALUES if r not in EXPECTED_VISIBLE_REGIONS
]

EXAMPLE_TEXT_OPTIONS = {
    "fontweight": "normal",
    "fontsize": 10,
    "rotation": "horizontal",
    "color": "black",
    "alpha": 1,
}

COMMON_PARAMS = {
    "position": POSITION_UM,
    "orientation": ORIENTATION,
    "thickness": 3000,
    "title": "frontal view",
    "vmin": -5,
    "vmax": 3,
    "annotate_text_options_2d": EXAMPLE_TEXT_OPTIONS,
    "check_latest": False,
    "interactive": False,
}


# Fixtures


@pytest.fixture
def heatmap_2d():
    heatmap = bgh.Heatmap(EXAMPLE_VALUES, format="2D", **COMMON_PARAMS)
    yield heatmap
    heatmap.scene.close()


@pytest.fixture
def heatmap_3d():
    heatmap = bgh.Heatmap(EXAMPLE_VALUES, format="3D", **COMMON_PARAMS)
    yield heatmap
    heatmap.scene.close()


# plot_subplot: image output


@pytest.mark.parametrize(
    "color_mode",
    ["heatmap", "atlas", "discrete"],
    ids=["heatmap", "atlas", "discrete"],
)
def test_plot_subplot_draws_image(heatmap_2d, color_mode):
    heatmap_2d.color_mode = color_mode
    fig, ax = plt.subplots()
    heatmap_2d.plot_subplot(fig, ax)
    assert len(ax.images) > 0
    plt.close("all")


def test_plot_subplot_returns_fig_and_ax(heatmap_2d):
    fig, ax = plt.subplots()
    result = heatmap_2d.plot_subplot(fig, ax)
    assert isinstance(result, tuple) and len(result) == 2
    plt.close("all")


# plot_subplot: colorbar


def test_colorbar_present_in_heatmap_mode(heatmap_2d):
    heatmap_2d.color_mode = "heatmap"
    fig, ax = plt.subplots()
    heatmap_2d.plot_subplot(fig, ax, show_cbar=True)
    assert len(fig.axes) > 1
    plt.close("all")


def test_no_colorbar_in_atlas_mode(heatmap_2d):
    heatmap_2d.color_mode = "atlas"
    fig, ax = plt.subplots()
    heatmap_2d.plot_subplot(fig, ax, show_cbar=True)
    # colorbar is suppressed for atlas/discrete modes
    assert len(fig.axes) == 1
    plt.close("all")


# plot_subplot: legend


@pytest.mark.parametrize(
    "color_mode,expect_legend",
    [
        ("atlas", True),
        ("discrete", True),
        ("heatmap", False),
    ],
    ids=["atlas", "discrete", "heatmap"],
)
def test_legend_presence_by_color_mode(heatmap_2d, color_mode, expect_legend):
    heatmap_2d.color_mode = color_mode
    fig, ax = plt.subplots()
    heatmap_2d.plot_subplot(fig, ax, show_cbar=False)
    has_legend = ax.get_legend() is not None
    assert has_legend == expect_legend
    plt.close("all")


# plot_subplot: axes styling


def test_hide_axes_removes_ticks(heatmap_2d):
    fig, ax = plt.subplots()
    heatmap_2d.plot_subplot(fig, ax, hide_axes=True)
    assert ax.get_xticks().size == 0
    assert ax.get_yticks().size == 0
    plt.close("all")


def test_background_colour_applied(heatmap_2d):
    import matplotlib.colors as mc

    heatmap_2d.background_color = "#123456"
    fig, ax = plt.subplots()
    heatmap_2d.plot_subplot(fig, ax)
    assert mc.to_hex(ax.get_facecolor()) == "#123456"
    plt.close("all")


# plot_subplot: region labels


def test_show_labels_adds_text_artists():
    hm = bgh.Heatmap(
        {"CB": 1.0, "MY": 0.5},
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        show_labels=True,
        label_min_area=1,
        check_latest=False,
    )
    fig, ax = plt.subplots()
    hm.plot_subplot(fig, ax)
    assert len(ax.texts) > 0, "No region labels were drawn."
    plt.close("all")
    hm.scene.close()


# plot_subplot: warnings


def test_no_warning_for_valid_regions(heatmap_2d):
    fig, ax = plt.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        heatmap_2d.plot_subplot(fig, ax)
    assert not any("not recognized" in str(w.message) for w in caught)
    plt.close("all")


def test_warning_for_region_absent_from_slice():
    """A region absent from the chosen slice should emit a UserWarning."""
    hm = bgh.Heatmap(
        {"CB": 1.0},
        position=1_000,  # far anterior — cerebellum absent at this position
        orientation=ORIENTATION,
        format="2D",
        check_latest=False,
    )
    fig, ax = plt.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hm.plot_subplot(fig, ax)
    assert any("no pixels in slice" in str(w.message) for w in caught)
    plt.close("all")
    hm.scene.close()


# annotate_regions — 2D


@pytest.mark.parametrize(
    "annotate_regions,expected_regions,unexpected_regions",
    [
        (True, EXPECTED_VISIBLE_REGIONS, EXPECTED_ABSENT_REGIONS),
        (False, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        (None, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        ([], [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        ({}, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        (["CB"], ["CB"], EXPECTED_ABSENT_REGIONS),
        (
            {"CB": "Cerebellum", "MY": 25.5},
            ["Cerebellum"],
            ["CB", "25.5", "MY"] + EXPECTED_ABSENT_REGIONS,
        ),
    ],
    ids=[
        "true",
        "false",
        "empty_none",
        "empty_list",
        "empty_dict",
        "list_on_slice",
        "dict_custom",
    ],
)
def test_annotate_regions_2d(
    heatmap_2d, annotate_regions, expected_regions, unexpected_regions
):
    with patch("matplotlib.axes.Axes.annotate") as mock_annotate:
        heatmap_2d.annotate_regions = annotate_regions
        fig, ax = plt.subplots()
        heatmap_2d.plot_subplot(fig, ax)

        annotated = [call.args[0] for call in mock_annotate.call_args_list]

        if not expected_regions:
            assert (
                len(annotated) == 0
            ), f"Expected no annotations, got {annotated}"

        for region in expected_regions:
            assert any(
                region == a for a in annotated
            ), f"Expected '{region}' in annotations, got {annotated}"

        for region in unexpected_regions:
            assert (
                region not in annotated
            ), f"Unexpected region '{region}' found in annotations"

        if mock_annotate.called:
            for call in mock_annotate.call_args_list:
                for key, value in EXAMPLE_TEXT_OPTIONS.items():
                    assert call.kwargs[key] == value

        plt.close("all")


# annotate_regions — 3D


@pytest.mark.parametrize(
    "annotate_regions,expected_regions,unexpected_regions",
    [
        (True, EXPECTED_VISIBLE_REGIONS, EXPECTED_ABSENT_REGIONS),
        (False, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        (None, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        ([], [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        ({}, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        (["CB"], ["CB"], EXPECTED_ABSENT_REGIONS),
        (
            {"CB": "Cerebellum", "MY": 25.5},
            ["Cerebellum"],
            ["CB", "25.5", "MY"] + EXPECTED_ABSENT_REGIONS,
        ),
    ],
    ids=[
        "true",
        "false",
        "empty_none",
        "empty_list",
        "empty_dict",
        "list_on_slice",
        "dict_custom",
    ],
)
def test_annotate_regions_3d(
    heatmap_3d, annotate_regions, expected_regions, unexpected_regions
):
    with patch("brainrender.scene.Scene.add_label") as mock_add_label:
        heatmap_3d.annotate_regions = annotate_regions
        heatmap_3d.show()

        annotated = [
            call.kwargs["label"] for call in mock_add_label.call_args_list
        ]

        if not expected_regions:
            assert (
                len(annotated) == 0
            ), f"Expected no annotations, got {annotated}"

        for region in expected_regions:
            assert any(
                region == a for a in annotated
            ), f"Expected '{region}' in annotations, got {annotated}"

        for region in unexpected_regions:
            assert (
                region not in annotated
            ), f"Unexpected region '{region}' found in annotations"
