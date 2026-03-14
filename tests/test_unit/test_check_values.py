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
    def test_singleRegion(self, mock_atlas):
        values = {"TH": 0.9}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 0.9
        assert vmin == 0.9

    def test_multipleRegions(self, mock_atlas):
        values = {"TH": 0.9, "RSP": 1, "AI": 0.5, "SS": 0.3}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 1
        assert vmin == 0.3

    def test_integetInput(self, mock_atlas):
        values = {"TH": 1, "RSP": 0, "AI": -1}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 1
        assert vmin == -1

    def test_int_and_floatInput(self, mock_atlas):
        values = {"TH": 1, "RSP": 0.5, "AI": 1}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 1
        assert vmin == 0.5

    def test_sameValues(self, mock_atlas):
        values = {"TH": 0.5, "RSP": 0.5, "AI": 0.5}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 0.5
        assert vmin == 0.5

    def test_zeroValues(self, mock_atlas):
        values = {"TH": 0, "RSP": 0.0, "AI": 0}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 0
        assert vmin == 0


# Tests for NaN(Not a Number) in function check_values in heatmaps.py
class Test_NaN:
    def test_allNaN(self, mock_atlas):
        values = {"TH": np.nan, "RSP": np.nan}
        vmax, vmin = check_values(values, mock_atlas)
        assert np.isnan(vmax)
        assert np.isnan(vmin)

    def test_someNaN(self, mock_atlas):
        values = {"TH": np.nan, "RSP": 0.9, "AI": np.nan}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 0.9
        assert vmin == 0.9

    def test_single_Nan(self, mock_atlas):
        values = {"TH": 0.6, "RSP": 0.0, "AI": np.nan}
        vmax, vmin = check_values(values, mock_atlas)
        assert vmax == 0.6
        assert vmin == 0.0


# Tests for Invalid input in function check_values in heatmaps.py
class Test_InvalidInput:
    def test_emptyInput_raises(self, mock_atlas):
        values = {}
        vmax, vmin = check_values(values, mock_atlas)
        assert np.isnan(vmax)
        assert np.isnan(vmin)

    def test_NoneInput_raises(self, mock_atlas):
        values = {"RSP": None}
        with pytest.raises(
            ValueError, match="Heatmap values should be floats"
        ):
            check_values(values, mock_atlas)

    def test_stringInput_raises(self, mock_atlas):
        values = {"TH": "one"}
        with pytest.raises(
            ValueError, match="Heatmap values should be floats"
        ):
            check_values(values, mock_atlas)

    def test_ListInput_raises(self, mock_atlas):
        values = {"TH": [0, 1, 0.9]}
        with pytest.raises(
            ValueError, match="Heatmap values should be floats"
        ):
            check_values(values, mock_atlas)

    def test_UnknownRegion_raises(self, mock_atlas):
        values = {"FAKE_REGION": 1}
        with pytest.raises(ValueError, match="not recognized"):
            check_values(values, mock_atlas)

    def test_UnknownRegion_with_validRegion_raises(self, mock_atlas):
        values = {"TH": 1, "UNKNOWN": 1}
        with pytest.raises(ValueError, match="not recognized"):
            check_values(values, mock_atlas)
