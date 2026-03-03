import pytest
import numpy as np
from brainglobe_heatmap.slicer import get_ax_idx
from brainglobe_heatmap.slicer import Slicer

def test_get_ax_idx_frontal():
    assert get_ax_idx("frontal") == 0

def test_get_ax_idx_sagittal():
    assert get_ax_idx("sagittal") == 2

def test_get_ax_idx_horizontal():
    assert get_ax_idx("horizontal") == 1

def test_get_ax_idx_invalid():
    with pytest.raises(ValueError) as excinfo:
        get_ax_idx("invalid")

    assert 'not recognized' in str(excinfo.value)

def test_get_ax_idx_case_sensitive():
    with pytest.raises(ValueError):
        get_ax_idx("Frontal")

#def test_slicer_invalid_orientation():
#    with pytest.raises(ValueError):
#        Slicer(position=0, orientation="invalid", thickness=10)

class DummyRoot:
    def center_of_mass(self):
        return np.array([0, 0, 0])
    
def test_position_float_with_vector_orientation_raises():
    dummy_root = DummyRoot()

    with pytest.raises(ValueError):
        Slicer(
            position=10,
            orientation=np.array([1, 0, 0]),
            thickness=5,
            root=dummy_root,
        )

'''def test_position_none_uses_center_of_mass():
    dummy_root = DummyRoot()

    slicer = Slicer(
        position=None,
        orientation="frontal",
        thickness=5,
        root=dummy_root,
    )

    assert slicer is not None'''