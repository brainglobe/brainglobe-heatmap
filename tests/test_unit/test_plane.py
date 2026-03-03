import numpy as np
import pytest
from brainglobe_heatmap.plane import Plane

def test_plane_normal_computation():
    origin = np.array([0, 0, 0])
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])

    plane = Plane(origin, u, v)

    assert np.allclose(plane.normal, np.array([0, 0, 1]))

def test_plane_non_orthogonal_vectors_raise():
    origin = np.array([0, 0, 0])
    u = np.array([1, 0, 0])
    v = np.array([1, 0, 0])

    with pytest.raises(AssertionError):
        Plane(origin, u, v)

def test_from_norm_creates_valid_plane():
    origin = np.array([0, 0, 0])
    normal = np.array([0, 0, 1])

    plane = Plane.from_norm(origin, normal)

    assert np.allclose(
        np.abs(plane.normal),
        np.array([0, 0, 1])
    )

def test_p3_to_p2_projection():
    origin = np.array([0, 0, 0])
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])

    plane = Plane(origin, u, v)
    
    points_3d = np.array([
        [1, 2, 0],
        [3, 4, 0]
    ])

    projected = plane.p3_to_p2(points_3d)

    assert np.allclose(projected, np.array([
        [1, 2],
        [3, 4]
    ]))

def test_center_of_mass_returns_origin():
    origin = np.array([5, 6, 7])
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])

    plane = Plane(origin, u, v)

    assert np.allclose(plane.center_of_mass(), origin)