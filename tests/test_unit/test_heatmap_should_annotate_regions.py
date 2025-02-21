import pytest

import brainglobe_heatmap as bgh


@pytest.fixture
def heatmap():
    values = {"TH": 1, "RSP": 0.2}
    return bgh.Heatmap(
        values,
        position=5000,
        annotate_regions=False,
    )


@pytest.mark.parametrize(
    "annotate_regions, region, expected",
    [
        # annotate_regions=False (default)
        (False, "TH", None),
        # annotate_regions=True
        (True, "TH", "TH"),
        (True, "root", None),
        # annotate_regions as list
        (["TH"], "TH", "TH"),
        (["TH"], "RSP", None),
        # annotate_regions as dict
        ({"TH": "Thalamus", "RSP": 0.5}, "TH", "Thalamus"),
        ({"TH": "Thalamus", "RSP": 0.5}, "RSP", "0.5"),
        ({"TH": "Thalamus", "RSP": 0.5}, "AI", None),
        ({"TH": 123, "RSP": None, "VIS": True}, "TH", "123"),
        ({"TH": 123, "RSP": None, "VIS": True}, "RSP", "None"),
        ({"TH": 123, "RSP": None, "VIS": True}, "VIS", "True"),
        # Empty configurations
        ([], "TH", None),
        ({}, "TH", None),
        # Special region names
        (["3N"], "3N", "3N"),
        ({"3N": 41}, "3N", "41"),
    ],
)
def test_should_annotate_region(heatmap, annotate_regions, region, expected):
    heatmap.annotate_regions = annotate_regions

    result = heatmap.should_annotate_region(region)
    assert result == expected
