from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from brainglobe_heatmap.planner import plan, print_plane


class FakePlane:
    """Minimal stand-in for Plane with the attributes print_plane reads."""

    def __init__(self, center, normal, u, v):
        self.center = np.array(center, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.u = np.array(u, dtype=float)
        self.v = np.array(v, dtype=float)


@pytest.mark.parametrize(
    "center, normal, u, v",
    [
        pytest.param(
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            id="origin-axis-aligned",
        ),
        pytest.param(
            [5000.123, -3000.456, 7000.789],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            id="offset-center",
        ),
        pytest.param(
            [1.005, 2.005, 3.005],
            [0.577, 0.577, 0.577],
            [-0.707, 0.707, 0],
            [0.408, 0.408, -0.816],
            id="oblique-plane",
        ),
    ],
)
def test_print_plane_outputs_plane_attributes(
    center, normal, u, v, capsys
):
    plane = FakePlane(center, normal, u, v)
    print_plane("test plane", plane, "blue")

    captured = capsys.readouterr().out
    assert "test plane" in captured
    assert "center point" in captured
    assert "norm" in captured


def test_print_plane_rounds_center_to_two_decimals(capsys):
    plane = FakePlane(
        [1.23456, 7.89012, 3.45678],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    )
    print_plane("rounding", plane, "red")

    captured = capsys.readouterr().out
    assert "1.23" in captured
    assert "7.89" in captured
    assert "3.46" in captured
    # full precision should not appear
    assert "1.23456" not in captured


@patch("brainglobe_heatmap.planner.Heatmap.__init__", return_value=None)
@patch("brainglobe_heatmap.planner.print_plane")
def test_plan_converts_region_list_to_dict(mock_print, mock_init):
    mock_init.return_value = None
    p = plan.__new__(plan)
    p.slicer = MagicMock()

    regions_list = ["TH", "RSP", "AI"]
    plan.__init__(p, regions_list, position=(8000, 5000, 5000))

    call_args = mock_init.call_args
    passed_regions = call_args[0][0]
    assert passed_regions == {"TH": 1, "RSP": 1, "AI": 1}


@patch("brainglobe_heatmap.planner.Heatmap.__init__", return_value=None)
@patch("brainglobe_heatmap.planner.print_plane")
def test_plan_passes_dict_regions_unchanged(mock_print, mock_init):
    mock_init.return_value = None
    p = plan.__new__(plan)
    p.slicer = MagicMock()

    regions_dict = {"TH": 0.5, "RSP": -1.2}
    plan.__init__(p, regions_dict, position=(8000,))

    call_args = mock_init.call_args
    passed_regions = call_args[0][0]
    assert passed_regions == {"TH": 0.5, "RSP": -1.2}


@patch("brainglobe_heatmap.planner.Heatmap.__init__", return_value=None)
@patch("brainglobe_heatmap.planner.print_plane")
def test_plan_always_passes_3d_format(mock_print, mock_init):
    mock_init.return_value = None
    p = plan.__new__(plan)
    p.slicer = MagicMock()

    plan.__init__(p, {"TH": 1}, position=(5000,))

    call_kwargs = mock_init.call_args[1]
    assert call_kwargs["format"] == "3D"


@patch("brainglobe_heatmap.planner.Heatmap.__init__", return_value=None)
@patch("brainglobe_heatmap.planner.print_plane")
def test_plan_stores_arrow_scale(mock_print, mock_init):
    mock_init.return_value = None
    p = plan.__new__(plan)
    p.slicer = MagicMock()

    plan.__init__(
        p, {"TH": 1}, position=(5000,), arrow_scale=750
    )
    assert p.arrow_scale == 750


@patch("brainglobe_heatmap.planner.Heatmap.__init__", return_value=None)
@patch("brainglobe_heatmap.planner.print_plane")
def test_plan_default_arrow_scale(mock_print, mock_init):
    mock_init.return_value = None
    p = plan.__new__(plan)
    p.slicer = MagicMock()

    plan.__init__(p, {"TH": 1}, position=(5000,))
    assert p.arrow_scale == 10


@patch("brainglobe_heatmap.planner.Heatmap.__init__", return_value=None)
@patch("brainglobe_heatmap.planner.print_plane")
def test_plan_prints_both_slicer_planes(mock_print, mock_init):
    mock_init.return_value = None
    p = plan.__new__(plan)
    plane0 = MagicMock()
    plane1 = MagicMock()
    p.slicer = MagicMock(plane0=plane0, plane1=plane1)

    plan.__init__(p, {"TH": 1}, position=(5000,))

    assert mock_print.call_count == 2
    first_call = mock_print.call_args_list[0]
    second_call = mock_print.call_args_list[1]
    assert first_call[0][0] == "Plane 0"
    assert first_call[0][1] is plane0
    assert second_call[0][0] == "Plane 1"
    assert second_call[0][1] is plane1


@patch("brainglobe_heatmap.planner.Heatmap.__init__", return_value=None)
@patch("brainglobe_heatmap.planner.print_plane")
def test_plan_forwards_kwargs_to_heatmap(mock_print, mock_init):
    mock_init.return_value = None
    p = plan.__new__(plan)
    p.slicer = MagicMock()

    plan.__init__(
        p,
        {"TH": 1},
        position=(5000,),
        orientation="sagittal",
        thickness=2000,
    )

    call_kwargs = mock_init.call_args[1]
    assert call_kwargs["orientation"] == "sagittal"
    assert call_kwargs["thickness"] == 2000


def test_show_returns_scene():
    p = plan.__new__(plan)
    p.arrow_scale = 10
    p.interactive = False
    p.zoom = None

    mock_scene = MagicMock()
    mock_scene.root._mesh.alpha.return_value = None
    p.scene = mock_scene

    fake_plane = MagicMock()
    fake_plane.center = [0, 0, 0]
    fake_plane.normal = [1, 0, 0]
    fake_plane.u = [0, 1, 0]
    fake_plane.v = [0, 0, 1]

    plane_mesh = MagicMock()
    plane_mesh.alpha.return_value = plane_mesh
    plane_mesh.color.return_value = plane_mesh
    plane_mesh.center = [0, 0, 0]
    plane_mesh.width = 1000
    fake_plane.to_mesh.return_value = plane_mesh

    p.slicer = MagicMock(plane0=fake_plane, plane1=fake_plane)
    p.regions_meshes = []

    result = p.show()

    assert result is mock_scene
    mock_scene.render.assert_called_once_with(
        interactive=False, zoom=None
    )


def test_show_sets_root_alpha():
    p = plan.__new__(plan)
    p.arrow_scale = 10
    p.interactive = False
    p.zoom = None

    mock_scene = MagicMock()
    p.scene = mock_scene

    fake_plane = MagicMock()
    fake_plane.center = [0, 0, 0]
    fake_plane.normal = [1, 0, 0]
    fake_plane.u = [0, 1, 0]
    fake_plane.v = [0, 0, 1]

    plane_mesh = MagicMock()
    plane_mesh.alpha.return_value = plane_mesh
    plane_mesh.color.return_value = plane_mesh
    plane_mesh.center = [0, 0, 0]
    plane_mesh.width = 500
    fake_plane.to_mesh.return_value = plane_mesh

    p.slicer = MagicMock(plane0=fake_plane, plane1=fake_plane)
    p.regions_meshes = []

    p.show()

    mock_scene.root._mesh.alpha.assert_called_once_with(0.3)
