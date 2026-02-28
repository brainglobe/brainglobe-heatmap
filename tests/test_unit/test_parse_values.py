import pytest
from brainglobe_heatmap.heatmaps import parse_values


def test_all_bilateral():
    """Scalar-only input is fully backwards compatible."""
    bilateral, per_hemi = parse_values({"TH": 1.0, "VISp": 0.5})
    assert bilateral == {"TH": 1.0, "VISp": 0.5}
    assert per_hemi == {}


def test_all_per_hemisphere():
    """All regions with hemisphere-specific values."""
    bilateral, per_hemi = parse_values({
        "VISp": {"left": 0.8, "right": 0.2},
    })
    assert bilateral == {}
    assert per_hemi == {"VISp": {"left": 0.8, "right": 0.2}}


def test_mixed_bilateral_and_per_hemisphere():
    """Mix of scalar and per-hemisphere values splits correctly."""
    bilateral, per_hemi = parse_values({
        "TH": 1.0,
        "VISp": {"left": 0.8, "right": 0.2},
        "MOp": {"left": 0.5},
    })
    assert bilateral == {"TH": 1.0}
    assert per_hemi == {
        "VISp": {"left": 0.8, "right": 0.2},
        "MOp": {"left": 0.5},
    }


def test_left_only():
    """Only left hemisphere value specified."""
    bilateral, per_hemi = parse_values({"VISp": {"left": 0.8}})
    assert per_hemi == {"VISp": {"left": 0.8}}
    assert bilateral == {}


def test_right_only():
    """Only right hemisphere value specified."""
    bilateral, per_hemi = parse_values({"VISp": {"right": 0.3}})
    assert per_hemi == {"VISp": {"right": 0.3}}
    assert bilateral == {}


def test_invalid_hemisphere_key_raises():
    """Invalid hemisphere key (not left/right) raises ValueError."""
    with pytest.raises(ValueError, match='may only contain'):
        parse_values({"VISp": {"left": 0.8, "center": 0.2}})


def test_empty_hemisphere_dict_raises():
    """Empty per-hemisphere dict raises ValueError."""
    with pytest.raises(ValueError, match="is empty"):
        parse_values({"VISp": {}})


def test_empty_input():
    """Empty input returns two empty dicts."""
    bilateral, per_hemi = parse_values({})
    assert bilateral == {}
    assert per_hemi == {}
