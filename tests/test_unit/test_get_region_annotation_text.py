import pytest

from brainglobe_heatmap import Heatmap


@pytest.mark.parametrize(
    "annotate_regions, region, expected",
    [
        pytest.param(False, "TH", None, id="annotate_regions_false"),
        pytest.param(True, "TH", "TH", id="annotate_regions_true"),
        pytest.param(True, "root", None, id="root-ignored"),
        # annotate_regions as list
        pytest.param(["TH"], "TH", "TH", id="list-included"),
        pytest.param(["TH"], "RSP", None, id="list-excluded"),
        # annotate_regions as dict
        pytest.param(
            {"TH": "Thalamus", "RSP": 0.5},
            "TH",
            "Thalamus",
            id="dict-text-value",
        ),
        pytest.param(
            {"TH": "Thalamus", "RSP": 0.5},
            "RSP",
            "0.5",
            id="dict-numeric-value",
        ),
        pytest.param(
            {"TH": "Thalamus", "RSP": 0.5},
            "AI",
            None,
            id="dict-missing-region",
        ),
        pytest.param(
            {"TH": 123, "RSP": None, "VIS": True},
            "TH",
            "123",
            id="dict-int-value",
        ),
        pytest.param(
            {"TH": 123, "RSP": None, "VIS": True},
            "RSP",
            "None",
            id="dict-none-value",
        ),
        pytest.param(
            {"TH": 123, "RSP": None, "VIS": True},
            "VIS",
            "True",
            id="dict-bool-value",
        ),
        # Empty configurations
        pytest.param([], "TH", None, id="empty-list"),
        pytest.param({}, "TH", None, id="empty-dict"),
    ],
)
def test_get_region_annotation_text(annotate_regions, region, expected):
    heatmap = type("Heatmap", (), {"annotate_regions": annotate_regions})()
    result = Heatmap.get_region_annotation_text(heatmap, region)
    assert result == expected, (
        f"Expected annotation '{expected}' for region '{region}',"
        f"got '{result}'"
    )
