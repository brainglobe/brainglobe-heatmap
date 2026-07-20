import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from brainglobe_heatmap.heatmaps import find_annotation_position_inside_polygon


@pytest.mark.parametrize(
    "vertices",
    [
        np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),  # non-closed
        np.array(
            [
                [0, 0],
                [0, 2],
                [2, 2],
                [2, 1.5],
                [0.5, 1.5],
                [0.5, 0.5],
                [2, 0.5],
                [2, 0],
            ]
        ),  # concave
        np.array([[0, 0], [0, 0.0001], [0.0001, 0.0001], [0.0001, 0]]),  # tiny
        np.array(
            [
                [0, 0],
                [0, 2],
                [0.2, 2],
                [0.2, 0.2],
                [1.8, 0.2],
                [1.8, 1.8],
                [0.2, 1.8],
                [0.2, 0],
                [0, 0],
            ]
        ),  # narrow c-shape
        np.array([[0, 0], [2, 2], [2, 0], [0, 2]]),  # cross
        np.array(
            [[0, 0], [1, 1], [0, 2], [2, 2], [1, 1], [2, 0]]
        ),  # bowtie self-touch
    ],
    ids=[
        "non_closed_rectangle",
        "concave",
        "tiny",
        "narrow-c_shape",
        "intersecting_pts_cross",
        "intersecting_pts_bowtie",
    ],
)
def test_find_annotation_position_inside_polygon(vertices):
    result = find_annotation_position_inside_polygon(vertices)

    polygon = Polygon(vertices.tolist())
    point_obj_result = Point(result)
    assert point_obj_result.within(polygon)


@pytest.mark.parametrize(
    "invalid_vertices",
    [np.array([[0, 0], [1, 1], [2, 2]]), np.array([])],
    ids=["insufficient_vertices", "empty"],
)
def test_handles_invalid_polygons(invalid_vertices):
    result = find_annotation_position_inside_polygon(invalid_vertices)

    assert result is None, f"Expected None, but got {result!r}"


def test_too_few_vertices_returns_none():
    """Polygon with fewer than 4 vertices should return None."""
    vertices = np.array([[0, 0], [1, 0], [0, 1]])

    result = find_annotation_position_inside_polygon(vertices)

    assert result is None


def test_polylabel_square_center():
    """Polylabel should return a point near the center of a square."""
    vertices = np.array(
        [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [0, 0],
        ]
    )

    result = find_annotation_position_inside_polygon(vertices)

    assert result is not None
    x, y = result

    assert pytest.approx(x, abs=0.2) == 5
    assert pytest.approx(y, abs=0.2) == 5


def test_multipolygon_selects_largest_region():
    """
    Self-intersecting polygon should be repaired and
    return a point inside the resulting geometry.
    """
    vertices = np.array(
        [
            [0, 0],
            [0, 10],
            [15, 0],
            [15, 10],
            [0, 0],
        ]
    )

    result = find_annotation_position_inside_polygon(vertices)

    assert result is not None
    x, y = result

    repaired_polygon = Polygon(vertices).buffer(0)

    assert Point(x, y).within(repaired_polygon)
