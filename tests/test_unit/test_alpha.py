from unittest.mock import MagicMock, patch

import matplotlib as mpl
import pytest
from brainrender import settings

import brainglobe_heatmap as bgh

settings.INTERACTIVE = False
settings.OFFSCREEN = True

mpl.use("Agg")


EXAMPLE_VALUES = {
    "TH": 1,
    "RSP": 0.2,
    "HIP": 3,
}

COMMON_PARAMS = {
    "position": 9000,
    "orientation": "frontal",
    "thickness": 3000,
    "vmin": -5,
    "vmax": 3,
    "check_latest": False,
    "interactive": False,
}


@pytest.mark.parametrize(
    "alpha",
    [
        pytest.param(None, id="none"),
        pytest.param(0.0, id="float_zero"),
        pytest.param(0.5, id="float_mid"),
        pytest.param(1.0, id="float_one"),
        pytest.param({"HIP": 0.3, "TH": 0.8}, id="dict_valid"),
        pytest.param({"HIP": 0.0, "TH": 1.0}, id="dict_boundary"),
        pytest.param({"HIP": 0.5}, id="dict_partial"),
    ],
)
def test_alpha_valid_values_accepted(alpha):
    """Valid alpha values must not raise during Heatmap init."""
    heatmap = bgh.Heatmap(
        EXAMPLE_VALUES, format="3D", alpha=alpha, **COMMON_PARAMS
    )
    assert heatmap.alpha == alpha
    heatmap.scene.close()


@pytest.mark.parametrize(
    "alpha, exc_type, match",
    [
        pytest.param(
            -0.1,
            ValueError,
            "`alpha` must be between 0.0 and 1.0",
            id="float_below_zero",
        ),
        pytest.param(
            1.1,
            ValueError,
            "`alpha` must be between 0.0 and 1.0",
            id="float_above_one",
        ),
        pytest.param(
            {"HIP": -0.1},
            ValueError,
            "must be between 0.0 and 1.0",
            id="dict_below_zero",
        ),
        pytest.param(
            {"HIP": 1.5},
            ValueError,
            "must be between 0.0 and 1.0",
            id="dict_above_one",
        ),
        pytest.param(
            "high",
            TypeError,
            "`alpha` must be a float or a dict",
            id="wrong_type_str",
        ),
        pytest.param(
            [0.5],
            TypeError,
            "`alpha` must be a float or a dict",
            id="wrong_type_list",
        ),
    ],
)
def test_alpha_invalid_values_raise(alpha, exc_type, match):
    """Invalid alpha values must raise the correct exception."""
    with pytest.raises(exc_type, match=match):
        bgh.Heatmap(EXAMPLE_VALUES, format="3D", alpha=alpha, **COMMON_PARAMS)


@pytest.fixture
def heatmap_3d_no_alpha():
    heatmap = bgh.Heatmap(EXAMPLE_VALUES, format="3D", **COMMON_PARAMS)
    yield heatmap
    heatmap.scene.close()


@pytest.fixture
def heatmap_3d_global_alpha():
    heatmap = bgh.Heatmap(
        EXAMPLE_VALUES, format="3D", alpha=0.4, **COMMON_PARAMS
    )
    yield heatmap
    heatmap.scene.close()


@pytest.fixture
def heatmap_3d_dict_alpha():
    heatmap = bgh.Heatmap(
        EXAMPLE_VALUES,
        format="3D",
        alpha={"HIP": 0.2, "TH": 0.9},
        **COMMON_PARAMS,
    )
    yield heatmap
    heatmap.scene.close()


def test_alpha_none_stored(heatmap_3d_no_alpha):
    """When alpha is not passed, self.alpha must be None."""
    assert heatmap_3d_no_alpha.alpha is None


def test_global_alpha_stored(heatmap_3d_global_alpha):
    """Global float alpha must be stored correctly on the instance."""
    assert heatmap_3d_global_alpha.alpha == 0.4


def test_dict_alpha_stored(heatmap_3d_dict_alpha):
    """Dict alpha must be stored correctly on the instance."""
    assert heatmap_3d_dict_alpha.alpha == {"HIP": 0.2, "TH": 0.9}


def test_alpha_none_does_not_call_actor_alpha(heatmap_3d_no_alpha):
    """When alpha is None, actor.alpha() must never be called."""
    with patch("brainrender.scene.Scene.get_actors") as mock_get_actors:
        mock_actor = MagicMock()
        mock_get_actors.return_value = [mock_actor]
        heatmap_3d_no_alpha.render()
        mock_actor.alpha.assert_not_called()


def test_dict_alpha_only_specified_regions(heatmap_3d_dict_alpha):
    """Dict alpha must only target regions present in the dict."""
    alpha_dict = heatmap_3d_dict_alpha.alpha
    assert "RSP" not in alpha_dict
    assert alpha_dict.get("HIP") == 0.2
    assert alpha_dict.get("TH") == 0.9


def test_alpha_has_no_effect_on_2d():
    """Passing alpha with format='2D' must not raise."""
    heatmap = bgh.Heatmap(
        EXAMPLE_VALUES,
        format="2D",
        alpha=0.5,
        **COMMON_PARAMS,
    )
    assert heatmap.alpha == 0.5
    heatmap.scene.close()


def test_global_alpha_applied_to_actors(heatmap_3d_global_alpha):
    """Global alpha must be applied to region actors after render()."""
    heatmap_3d_global_alpha.render()
    for region in EXAMPLE_VALUES:
        actors = heatmap_3d_global_alpha.scene.get_actors(
            br_class="brain region", name=region
        )
        if actors:
            assert actors[0].alpha() == pytest.approx(0.4, abs=0.05)


def test_dict_alpha_applied_to_specified_actors(heatmap_3d_dict_alpha):
    """Dict alpha must be applied to actors whose region is in the dict."""
    heatmap_3d_dict_alpha.render()
    for region, expected_alpha in {"HIP": 0.2, "TH": 0.9}.items():
        actors = heatmap_3d_dict_alpha.scene.get_actors(
            br_class="brain region", name=region
        )
        if actors:
            assert actors[0].alpha() == pytest.approx(expected_alpha, abs=0.05)


def test_global_alpha_render_applies_to_all_regions():
    """render() must apply alpha() to every region actor."""
    heatmap = bgh.Heatmap(
        EXAMPLE_VALUES, format="3D", alpha=0.6, **COMMON_PARAMS
    )
    heatmap.render()
    for region in EXAMPLE_VALUES:
        actors = heatmap.scene.get_actors(br_class="brain region", name=region)
        if actors:
            assert actors[0].alpha() == pytest.approx(0.6, abs=0.05)
    heatmap.scene.close()


def test_dict_alpha_render_applies_to_matching_regions():
    """render() must apply alpha() only to regions present in the dict."""
    heatmap = bgh.Heatmap(
        EXAMPLE_VALUES,
        format="3D",
        alpha={"HIP": 0.3},
        **COMMON_PARAMS,
    )
    heatmap.render()
    actors = heatmap.scene.get_actors(br_class="brain region", name="HIP")
    if actors:
        assert actors[0].alpha() == pytest.approx(0.3, abs=0.05)
    heatmap.scene.close()


def test_render_global_alpha_calls_actor_alpha(heatmap_3d_global_alpha):
    """render() must call actor.alpha() for every region
    when alpha is a float."""
    mock_actor = MagicMock()
    mock_actor.name = "mock_region"
    mock_actor._mesh.vertices = []

    with patch.object(
        heatmap_3d_global_alpha.scene,
        "get_actors",
        return_value=[mock_actor],
    ):
        heatmap_3d_global_alpha.render()

    mock_actor.alpha.assert_called_with(0.4)


def test_render_dict_alpha_calls_actor_alpha_for_matching_region():
    """render() must call actor.alpha() only for regions in the dict."""
    heatmap = bgh.Heatmap(
        EXAMPLE_VALUES,
        format="3D",
        alpha={"TH": 0.9},
        **COMMON_PARAMS,
    )

    mock_actors = {}
    for region in EXAMPLE_VALUES:
        actor = MagicMock()
        actor.name = region
        actor._mesh.vertices = []
        mock_actors[region] = actor

    def fake_get_actors(br_class=None, name=None):
        return [mock_actors[name]]

    with patch.object(
        heatmap.scene, "get_actors", side_effect=fake_get_actors
    ):
        heatmap.render()

    # TH is in the dict — alpha should be called
    mock_actors["TH"].alpha.assert_called_with(0.9)
    # HIP and RSP are not in the dict — alpha should NOT be called
    mock_actors["HIP"].alpha.assert_not_called()
    mock_actors["RSP"].alpha.assert_not_called()
    heatmap.scene.close()
