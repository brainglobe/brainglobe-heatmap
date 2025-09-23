from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from brainrender import settings

import brainglobe_heatmap as bgh

settings.INTERACTIVE = False
settings.OFFSCREEN = True

# Regions visible in a frontal slice at position 9000
EXPECTED_VISIBLE_REGIONS = ["TH", "RSP", "VIS", "CB", "HIP", "RHP"]

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

EXPECTED_ABSENT_REGIONS = ["root"] + [
    region
    for region in EXAMPLE_VALUES.keys()
    if region not in EXPECTED_VISIBLE_REGIONS
]

EXAMPLE_TEXT_OPTIONS = {
    "fontweight": "normal",
    "fontsize": 10,
    "rotation": "horizontal",
    "color": "black",
    "alpha": 1,
}

COMMON_PARAMS = {
    "position": 9000,
    "orientation": "frontal",
    "thickness": 3000,
    "title": "frontal view",
    "vmin": -5,
    "vmax": 3,
    "annotate_text_options_2d": EXAMPLE_TEXT_OPTIONS,
    "check_latest": False,
    "interactive": False,
}

mpl.use("Agg")  # Use a non-interactive backend for testing


@pytest.fixture
def heatmap_2d():
    """Fixture for 2D heatmap"""
    heatmap = bgh.Heatmap(EXAMPLE_VALUES, format="2D", **COMMON_PARAMS)
    yield heatmap
    heatmap.scene.close()


@pytest.fixture
def heatmap_3d():
    """Fixture for 3D heatmap"""
    heatmap = bgh.Heatmap(EXAMPLE_VALUES, format="3D", **COMMON_PARAMS)
    yield heatmap
    heatmap.scene.close()


@pytest.mark.parametrize(
    "annotate_regions,expected_regions,unexpected_regions",
    [
        (True, EXPECTED_VISIBLE_REGIONS, EXPECTED_ABSENT_REGIONS),
        (False, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        # empty
        (None, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        ([], [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        ({}, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        # specified regions
        (["TH", "SS"], ["TH"], ["SS"] + EXPECTED_ABSENT_REGIONS),
        # CB will be on the slice at position 9000
        (["CB"], ["CB"], EXPECTED_ABSENT_REGIONS),
        # Custom annotations
        (
            {"TH": "Thalamus", "CB": 25.5},
            ["Thalamus", "25.5"],
            ["TH", "CB"] + EXPECTED_ABSENT_REGIONS,
        ),
    ],
    ids=[
        "true",
        "false",
        "empty_none",
        "empty_list",
        "empty_dict",
        "list_specified",
        "list_specified_on_slice",
        "dict_custom_annotations",
    ],
)
def test_example_region_annotation_2d(
    heatmap_2d, annotate_regions, expected_regions, unexpected_regions
):
    with patch("matplotlib.axes.Axes.annotate") as mock_annotate:
        heatmap_2d.annotate_regions = annotate_regions
        fig, ax = plt.subplots()
        heatmap_2d.plot_subplot(fig, ax)

        # Get actual annotated regions
        annotated_regions = [
            call.args[0] for call in mock_annotate.call_args_list
        ]

        if not expected_regions:
            assert len(annotated_regions) == 0, (
                f"Expected no annotations."
                f"Found {len(annotated_regions)}: {annotated_regions}"
            )

        for region in expected_regions:
            assert any(
                region == annotation for annotation in annotated_regions
            ), f"Expected region {region} to be found in annotations"

        for region in unexpected_regions:
            assert (
                region not in annotated_regions
            ), f"Unexpected region {region} found in annotations"

        if mock_annotate.called:
            for call in mock_annotate.call_args_list:
                for key, value in EXAMPLE_TEXT_OPTIONS.items():
                    assert call.kwargs[key] == value


@pytest.mark.parametrize(
    "annotate_regions,expected_regions,unexpected_regions",
    [
        (True, EXPECTED_VISIBLE_REGIONS, EXPECTED_ABSENT_REGIONS),
        (False, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        # empty
        (None, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        ([], [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        ({}, [], EXPECTED_VISIBLE_REGIONS + EXPECTED_ABSENT_REGIONS),
        # specified regions
        (["TH", "SS"], ["TH"], ["SS"] + EXPECTED_ABSENT_REGIONS),
        # CB will be on the slice at position 9000
        (["CB"], ["CB"], EXPECTED_ABSENT_REGIONS),
        # Custom annotations
        (
            {"TH": "Thalamus", "CB": 25.5},
            ["Thalamus", "25.5"],
            ["TH", "CB"] + EXPECTED_ABSENT_REGIONS,
        ),
    ],
    ids=[
        "true",
        "false",
        "empty_none",
        "empty_list",
        "empty_dict",
        "list_specified",
        "list_specified_on_slice",
        "dict_custom_annotations",
    ],
)
def test_example_region_annotation_3d(
    heatmap_3d, annotate_regions, expected_regions, unexpected_regions
):
    with patch("brainrender.scene.Scene.add_label") as mock_add_label:
        heatmap_3d.annotate_regions = annotate_regions
        heatmap_3d.show()

        # Get actual annotated regions
        annotated_regions = [
            call.kwargs["label"] for call in mock_add_label.call_args_list
        ]

        if not expected_regions:
            assert len(annotated_regions) == 0, (
                f"Expected no annotations."
                f"Found {len(annotated_regions)}: {annotated_regions}"
            )

        for region in expected_regions:
            assert any(
                region == annotation for annotation in annotated_regions
            ), f"Expected {region} to be found in annotations"

        for region in unexpected_regions:
            assert (
                region not in annotated_regions
            ), f"Unexpected region {region} found in annotations"
