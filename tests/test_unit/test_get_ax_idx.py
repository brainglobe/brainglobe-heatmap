import pytest

from brainglobe_heatmap.slicer import get_ax_idx


# Tests for Orientation values in function get_ax_idx in slicer.py
class Test_GetAxIndex:
    def test_frontal(self):
        assert get_ax_idx("frontal") == 0

    def test_sagittal(self):
        assert get_ax_idx("sagittal") == 2

    def test_horizontal(self):
        assert get_ax_idx("horizontal") == 1

    def test_invalidOrientation(self):
        with pytest.raises(ValueError, match="not recognized"):
            get_ax_idx("vertical")

    def test_CaseSensitive_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            get_ax_idx("Frontal")

    def test_emptyValue_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            get_ax_idx("")
