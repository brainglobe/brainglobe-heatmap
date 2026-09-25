from unittest.mock import MagicMock, patch

import numpy as np

from brainglobe_heatmap.planner import plan


@patch("brainglobe_heatmap.heatmaps.check_values")
def test_plan_init_with_list_regions(mock_check):
    mock_check.return_value = (1, 0)

    regions = ["A", "B"]
    position = [0, 0, 0]

    p = plan(regions, position)

    assert p.values == {"A": 1, "B": 1}


@patch("brainglobe_heatmap.heatmaps.check_values")
@patch("brainglobe_heatmap.planner.print_plane")
def test_print_plane_called(mock_print, mock_check):
    mock_check.return_value = (1, 0)

    plan({"A": 1}, [0, 0, 0])

    assert mock_print.call_count == 2


@patch("brainglobe_heatmap.heatmaps.check_values")
@patch("brainglobe_heatmap.planner.Sphere")
@patch("brainglobe_heatmap.planner.Arrow")
def test_show_calls_render_and_add(mock_arrow, mock_sphere, mock_check):
    mock_check.return_value = (1, 0)

    p = plan({"A": 1}, [0, 0, 0])

    # Mock scene
    p.scene = MagicMock()
    p.scene.root = MagicMock()
    p.scene.root._mesh = MagicMock()

    # Mock slicer
    p.slicer = MagicMock()

    # Mock planes
    mock_plane = MagicMock()
    mock_plane.center = np.array([0, 0, 0])
    mock_plane.normal = np.array([1, 0, 0])
    mock_plane.u = np.array([0, 1, 0])
    mock_plane.v = np.array([0, 0, 1])

    mock_mesh = MagicMock()
    mock_mesh.center = [0, 0, 0]
    mock_mesh.width = 10
    mock_plane.to_mesh.return_value = mock_mesh

    p.slicer.plane0 = mock_plane
    p.slicer.plane1 = mock_plane

    p.regions_meshes = []

    # Run
    p.show()

    # Assertions
    p.slicer.show_plane_intersection.assert_called_once()
    assert p.scene.add.call_count > 0
    p.scene.render.assert_called_once()
