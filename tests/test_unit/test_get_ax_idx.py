import pytest

from brainglobe_heatmap.slicer import get_ax_idx


# Tests for Orientation values in function get_ax_idx in slicer.py
@pytest.mark.parametrize(
    "input_str, out_idx",
    [
        ("frontal", 0),
        ("horizontal", 1),
        ("sagittal", 2),
    ],
)
def test_get_ax_idx(input_str, out_idx):
    assert get_ax_idx(input_str) == out_idx


def test_invalid_orientation_raises():
    with pytest.raises(ValueError, match="not recognized"):
        get_ax_idx("vertical")


def test_case_sensitive_raises():
    with pytest.raises(ValueError, match="not recognized"):
        get_ax_idx("Frontal")


def test_empty_value_raises():
    with pytest.raises(ValueError, match="not recognized"):
        get_ax_idx("")
