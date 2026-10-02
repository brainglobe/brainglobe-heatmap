import numpy as np
import pandas as pd
import pytest

from brainglobe_heatmap.heatmaps import check_values


# Mocking an atlas for the tests
@pytest.fixture
def mock_atlas():
    atlas = type("MockAtlas", (), {})()
    atlas.lookup_df = pd.DataFrame(
        {"acronym": ["TH", "RSP", "AI", "SS", "MO", "VIS", "HIP", "CB"]}
    )
    return atlas


# Tests for valid inputs in function check_values in heatmaps.py
class TestValidInput:
    def test_single_region(self, mock_atlas):
        values = {"TH": 0.9}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 0.9
        assert vmin == 0.9

    def test_multiple_regions(self, mock_atlas):
        values = {"TH": 0.9, "RSP": 1, "AI": 0.5, "SS": 0.3}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 1
        assert vmin == 0.3

    def test_integer_input(self, mock_atlas):
        values = {"TH": 1, "RSP": 0, "AI": -1}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 1
        assert vmin == -1

    def test_int_and_float_input(self, mock_atlas):
        values = {"TH": 1, "RSP": 0.5, "AI": 1}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 1
        assert vmin == 0.5

    def test_same_values(self, mock_atlas):
        values = {"TH": 0.5, "RSP": 0.5, "AI": 0.5}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 0.5
        assert vmin == 0.5

    def test_zero_values(self, mock_atlas):
        values = {"TH": 0, "RSP": 0.0, "AI": 0}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 0
        assert vmin == 0


# Tests for NaN(Not a Number) in function check_values in heatmaps.py
class Test_NaN:
    def test_all_nan(self, mock_atlas):
        values = {"TH": np.nan, "RSP": np.nan}
        vmax, vmin = check_values(values, mock_atlas)
        assert np.isnan(vmax)
        assert np.isnan(vmin)

    def test_some_nan(self, mock_atlas):
        values = {"TH": np.nan, "RSP": 0.9, "AI": np.nan}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 0.9
        assert vmin == 0.9

    def test_single_nan(self, mock_atlas):
        values = {"TH": 0.6, "RSP": 0.0, "AI": np.nan}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 0.6
        assert vmin == 0.0


# Tests for Invalid input in function check_values in heatmaps.py
class Test_InvalidInput:
    def test_empty_input(self, mock_atlas):
        values = {}
        vmax, vmin = check_values(values, mock_atlas)
        assert np.isnan(vmax)
        assert np.isnan(vmin)

    def test_none_input_raises(self, mock_atlas):
        values = {"RSP": None}
        with pytest.raises(
            ValueError, match="Heatmap values should be floats"
        ):
            check_values(values, mock_atlas)

    def test_string_input_raises(self, mock_atlas):
        values = {"TH": "one"}
        with pytest.raises(
            ValueError, match="Heatmap values should be floats"
        ):
            check_values(values, mock_atlas)

    def test_list_input_raises(self, mock_atlas):
        values = {"TH": [0, 1, 0.9]}
        with pytest.raises(
            ValueError, match="Heatmap values should be floats"
        ):
            check_values(values, mock_atlas)

    def test_unknown_region_raises(self, mock_atlas):
        values = {"FAKE_REGION": 1}
        with pytest.raises(ValueError, match="not recognized"):
            check_values(values, mock_atlas)

    def test_unknown_region_with_valid_region_raises(self, mock_atlas):
        values = {"TH": 1, "UNKNOWN": 1}
        with pytest.raises(ValueError, match="not recognized"):
            check_values(values, mock_atlas)


# Tests for per-hemisphere dict values
class TestPerHemisphere:
    def test_range_spans_scalar_and_per_hemisphere(self, mock_atlas):
        values = {"TH": 1.0, "RSP": {"left": 0.8, "right": 0.2}}
        assert check_values(values, mock_atlas) == (1.0, 0.2)

    def test_single_side(self, mock_atlas):
        assert check_values({"TH": {"right": 0.3}}, mock_atlas) == (0.3, 0.3)

    @pytest.mark.parametrize("bad", [{}, {"left": 1, "center": 2}])
    def test_bad_sides_raise(self, mock_atlas, bad):
        with pytest.raises(ValueError, match="must have"):
            check_values({"TH": bad}, mock_atlas)

    def test_non_numeric_side_raises(self, mock_atlas):
        with pytest.raises(ValueError, match=r"TH\[left\]"):
            check_values({"TH": {"left": "high"}}, mock_atlas)
