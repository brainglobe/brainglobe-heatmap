import numpy as np
import pytest

import brainglobe_heatmap as bgh
from brainglobe_heatmap.heatmaps import (
    ATLAS_REGION_COLORS,
    _build_id_to_acronym,
    _build_region_masks_bottomup,
    _get_orientation_axis,
    _get_slice_from_volume,
    _smooth_contour_path,
)

ATLAS_NAME = "allen_mouse_25um"
POSITION_UM = 11_500
ORIENTATION = "frontal"

SMALL_DICT = {
    "CB": 1.09,
    "MY": 0.35,
    "NOD": 0.42,
    "IO": 0.50,
}


# Shared fixtures


@pytest.fixture(scope="module")
def atlas():
    from bg_atlasapi import BrainGlobeAtlas

    return BrainGlobeAtlas(ATLAS_NAME)


@pytest.fixture(scope="module")
def frontal_slice(atlas):
    return _get_slice_from_volume(atlas, POSITION_UM, ORIENTATION)


@pytest.fixture(scope="module")
def id_to_acronym_map(atlas):
    return _build_id_to_acronym(atlas)


@pytest.fixture(scope="module")
def region_masks(atlas, frontal_slice, id_to_acronym_map):
    slice_data, _ = frontal_slice
    return _build_region_masks_bottomup(
        atlas, slice_data, id_to_acronym_map, list(SMALL_DICT.keys())
    )


# _get_orientation_axis


@pytest.mark.parametrize(
    "orientation,expected_axis",
    [
        ("frontal", 0),
        ("horizontal", 1),
        ("sagittal", 2),
    ],
    ids=["frontal", "horizontal", "sagittal"],
)
def test_get_orientation_axis(orientation, expected_axis):
    assert _get_orientation_axis(orientation) == expected_axis


def test_get_orientation_axis_unknown_raises():
    with pytest.raises(KeyError):
        _get_orientation_axis("coronal")


# _get_slice_from_volume


def test_get_slice_from_volume_returns_2d_array(atlas):
    s, _ = _get_slice_from_volume(atlas, POSITION_UM, ORIENTATION)
    assert s.ndim == 2


@pytest.mark.parametrize(
    "orientation,axis",
    [
        ("frontal", 0),
        ("horizontal", 1),
        ("sagittal", 2),
    ],
    ids=["frontal", "horizontal", "sagittal"],
)
def test_get_slice_from_volume_index_conversion(atlas, orientation, axis):
    """Position in µm should be converted to the correct slice index."""
    position = 5_000
    _, idx = _get_slice_from_volume(atlas, position, orientation)
    expected = int(round(position / atlas.resolution[axis]))
    assert idx == expected


def test_get_slice_from_volume_position_zero_gives_index_zero(atlas):
    _, idx = _get_slice_from_volume(atlas, 0, ORIENTATION)
    assert idx == 0


def test_get_slice_from_volume_clamped_to_max(atlas):
    """An out-of-bounds position must be clamped to the last valid index."""
    _, idx = _get_slice_from_volume(atlas, 1_000_000_000, ORIENTATION)
    assert idx == atlas.annotation.shape[0] - 1


def test_get_slice_from_volume_custom_orientation_returns_none(atlas):
    """A custom normal vector should return (None, None) to trigger fallback."""
    s, idx = _get_slice_from_volume(atlas, POSITION_UM, (1, 0, 0))
    assert s is None
    assert idx is None


def test_get_slice_from_volume_3d_position_uses_correct_axis(atlas):
    """A 3-element position array should use the axis matching the orientation."""
    pos_3d = [POSITION_UM, 0, 0]
    _, idx_3d = _get_slice_from_volume(atlas, pos_3d, ORIENTATION)
    _, idx_scalar = _get_slice_from_volume(atlas, POSITION_UM, ORIENTATION)
    assert idx_3d == idx_scalar


def test_get_slice_from_volume_contains_annotated_pixels(atlas):
    s, _ = _get_slice_from_volume(atlas, POSITION_UM, ORIENTATION)
    assert (s > 0).any()


# _build_id_to_acronym


def test_build_id_to_acronym_returns_dict(id_to_acronym_map):
    assert isinstance(id_to_acronym_map, dict)


def test_build_id_to_acronym_keys_are_integers(id_to_acronym_map):
    assert all(isinstance(k, int) for k in id_to_acronym_map)


def test_build_id_to_acronym_values_are_strings(id_to_acronym_map):
    assert all(isinstance(v, str) for v in id_to_acronym_map.values())


def test_build_id_to_acronym_not_empty(id_to_acronym_map):
    assert len(id_to_acronym_map) > 0


def test_build_id_to_acronym_known_region(atlas, id_to_acronym_map):
    cb_id = atlas.structures["CB"]["id"]
    assert cb_id in id_to_acronym_map
    # Only assert the value is a non-empty string — the exact acronym string
    # returned may vary across bg_atlasapi versions.
    assert isinstance(id_to_acronym_map[cb_id], str)
    assert len(id_to_acronym_map[cb_id]) > 0


def test_build_id_to_acronym_no_duplicates(id_to_acronym_map):
    acronyms = list(id_to_acronym_map.values())
    assert len(acronyms) == len(set(acronyms))


# _build_region_masks_bottomup


def test_build_region_masks_bottomup_has_key_for_every_target(region_masks):
    for rname in SMALL_DICT:
        assert rname in region_masks


def test_build_region_masks_bottomup_masks_are_boolean(region_masks):
    for mask in region_masks.values():
        assert mask.dtype == bool


def test_build_region_masks_bottomup_shape_matches_slice(
    region_masks, frontal_slice
):
    slice_data, _ = frontal_slice
    for mask in region_masks.values():
        assert mask.shape == slice_data.shape


def test_build_region_masks_bottomup_cb_not_empty(region_masks):
    """Regression test for the colour bug.

    Before the fix, CB's mask was always empty because its pixels in the
    annotation volume are carried by its leaf sub-regions, not by CB's own
    ID.  The bottom-up ancestor traversal must assign those pixels to CB.
    """
    assert region_masks[
        "CB"
    ].any(), "CB mask is empty — the bottom-up ancestor fix is not working."


def test_build_region_masks_bottomup_distinct_regions_differ(region_masks):
    regions = [r for r in region_masks if region_masks[r].any()]
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            r1, r2 = regions[i], regions[j]
            assert not np.array_equal(
                region_masks[r1], region_masks[r2]
            ), f"Identical masks for '{r1}' and '{r2}'"


def test_build_region_masks_bottomup_empty_target_list(
    atlas, frontal_slice, id_to_acronym_map
):
    slice_data, _ = frontal_slice
    masks = _build_region_masks_bottomup(
        atlas, slice_data, id_to_acronym_map, []
    )
    assert masks == {}


def test_build_region_masks_bottomup_unknown_region_gives_empty_mask(
    atlas, frontal_slice, id_to_acronym_map
):
    slice_data, _ = frontal_slice
    masks = _build_region_masks_bottomup(
        atlas, slice_data, id_to_acronym_map, ["NONEXISTENT_XYZ"]
    )
    assert "NONEXISTENT_XYZ" in masks
    assert not masks["NONEXISTENT_XYZ"].any()


def test_build_region_masks_bottomup_cb_coverage(region_masks, frontal_slice):
    """CB should cover at least 1 % of annotated pixels in the slice."""
    slice_data, _ = frontal_slice
    total = (slice_data > 0).sum()
    ratio = region_masks["CB"].sum() / total
    assert ratio > 0.01, f"CB covers only {ratio:.1%} of brain pixels."


# _smooth_contour_path


def _circle_coords(n: int = 40) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([np.sin(t) * 10 + 20, np.cos(t) * 10 + 20])


def test_smooth_contour_path_output_shape_unchanged():
    coords = _circle_coords()
    assert _smooth_contour_path(coords, sigma=1.5).shape == coords.shape


def test_smooth_contour_path_short_path_returned_as_is():
    """Contours with fewer than 6 points must be returned unchanged."""
    coords = np.array([[0, 0], [1, 1], [2, 2]])
    np.testing.assert_array_equal(
        _smooth_contour_path(coords, sigma=1.5), coords
    )


def test_smooth_contour_path_reduces_variance():
    rng = np.random.default_rng(42)
    noisy = _circle_coords() + rng.normal(0, 3, (40, 2))
    smoothed = _smooth_contour_path(noisy, sigma=2.0)
    assert smoothed[:, 0].var() < noisy[:, 0].var()
    assert smoothed[:, 1].var() < noisy[:, 1].var()


def test_smooth_contour_path_tiny_sigma_preserves_shape():
    coords = _circle_coords()
    np.testing.assert_allclose(
        _smooth_contour_path(coords, sigma=0.01), coords, atol=1.0
    )


# ATLAS_REGION_COLORS palette


def test_atlas_region_colors_minimum_length():
    assert len(ATLAS_REGION_COLORS) >= 20


def test_atlas_region_colors_all_valid():
    import matplotlib.colors as mc

    for color in ATLAS_REGION_COLORS:
        assert mc.is_color_like(color), f"Invalid color: {color}"


def test_atlas_region_colors_all_distinct():
    assert len(ATLAS_REGION_COLORS) == len(set(ATLAS_REGION_COLORS))


# Heatmap.get_region_annotation_text


@pytest.mark.parametrize(
    "annotate_regions,region,expected",
    [
        pytest.param(False, "TH", None, id="annotate_regions_false"),
        pytest.param(True, "TH", "TH", id="annotate_regions_true"),
        pytest.param(True, "root", None, id="root_ignored"),
        pytest.param(["TH"], "TH", "TH", id="list_included"),
        pytest.param(["TH"], "RSP", None, id="list_excluded"),
        pytest.param(
            {"TH": "Thalamus", "RSP": 0.5},
            "TH",
            "Thalamus",
            id="dict_text_value",
        ),
        pytest.param(
            {"TH": "Thalamus", "RSP": 0.5},
            "RSP",
            "0.5",
            id="dict_numeric_value",
        ),
        pytest.param(
            {"TH": "Thalamus", "RSP": 0.5},
            "AI",
            None,
            id="dict_missing_region",
        ),
        pytest.param(
            {"TH": 123, "RSP": None, "VIS": True},
            "TH",
            "123",
            id="dict_int_value",
        ),
        pytest.param(
            {"TH": 123, "RSP": None, "VIS": True},
            "RSP",
            "None",
            id="dict_none_value",
        ),
        pytest.param(
            {"TH": 123, "RSP": None, "VIS": True},
            "VIS",
            "True",
            id="dict_bool_value",
        ),
        pytest.param([], "TH", None, id="empty_list"),
        pytest.param({}, "TH", None, id="empty_dict"),
    ],
)
def test_get_region_annotation_text(annotate_regions, region, expected):
    heatmap = type("Heatmap", (), {"annotate_regions": annotate_regions})()
    result = bgh.Heatmap.get_region_annotation_text(heatmap, region)
    assert (
        result == expected
    ), f"Expected '{expected}' for region '{region}', got '{result}'"


# vmin / vmax


@pytest.mark.parametrize(
    "vmin,vmax",
    [
        (0.0, 2.0),
        (-1.0, 1.0),
        (0, 0.001),
    ],
    ids=["positive_range", "symmetric_range", "tiny_range"],
)
def test_custom_vmin_vmax_stored(vmin, vmax):
    hm = bgh.Heatmap(
        SMALL_DICT,
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        vmin=vmin,
        vmax=vmax,
        check_latest=False,
    )
    assert hm.vmin == vmin
    assert hm.vmax == vmax


def test_auto_vmin_vmax_from_values():
    hm = bgh.Heatmap(
        SMALL_DICT,
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        check_latest=False,
    )
    assert hm.vmin == pytest.approx(min(SMALL_DICT.values()))
    assert hm.vmax == pytest.approx(max(SMALL_DICT.values()))


def test_equal_values_do_not_produce_degenerate_range():
    """When all values are equal, vmin must be adjusted to avoid a
    degenerate colourmap range."""
    uniform = {"CB": 0.5, "MY": 0.5}
    hm = bgh.Heatmap(
        uniform,
        position=POSITION_UM,
        orientation=ORIENTATION,
        format="2D",
        check_latest=False,
    )
    assert hm.vmin != hm.vmax


# Left-right symmetry regression


def test_slice_left_right_pixel_ratio(atlas):
    """The left and right halves of a frontal slice should contain a similar
    number of annotated pixels (ratio > 0.85).

    This is a regression test for bug #103, where the medio-lateral axis
    was inverted by the legacy mesh-projection pipeline.
    """
    s, _ = _get_slice_from_volume(atlas, POSITION_UM, "frontal")
    mid = s.shape[1] // 2
    left = (s[:, :mid] > 0).sum()
    right = (s[:, mid:] > 0).sum()
    ratio = min(left, right) / max(left, right)
    assert ratio > 0.85, (
        f"Left-right asymmetry detected (ratio={ratio:.3f}). "
        "Possible regression of bug #103."
    )


def test_slice_structure_id_overlap(atlas):
    """Structure IDs present in the left and right halves should overlap
    substantially (Jaccard index > 0.5)."""
    s, _ = _get_slice_from_volume(atlas, POSITION_UM, "frontal")
    mid = s.shape[1] // 2
    ids_left = set(np.unique(s[:, :mid])) - {0}
    ids_right = set(np.unique(s[:, mid:])) - {0}
    jaccard = len(ids_left & ids_right) / len(ids_left | ids_right)
    assert jaccard > 0.5, (
        f"Too few shared IDs between left and right halves "
        f"(Jaccard={jaccard:.3f}). Possible ML-axis inversion."
    )
