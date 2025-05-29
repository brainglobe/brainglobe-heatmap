from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from brainrender import settings

import brainglobe_heatmap as bgh

settings.INTERACTIVE = False
settings.OFFSCREEN = True

# mock projected data for get_structures_slice_coords
MOCK_PROJECTED = {
    "TH_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    "TH_segment_1": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    "HIP_segment_1": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    "VIS_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    "VIS_segment_1": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    "root_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
}

# regions expect to be on plot
MOCK_VISIBLE_REGIONS = set()
for segment in MOCK_PROJECTED:
    if not segment.startswith("root"):
        region, _ = segment.split("_segment_")
        MOCK_VISIBLE_REGIONS.add(region)

VALUES = {
    "TH": 1,
    "HIP": 3,
    "VIS": 2,
    "PA": -4,
}

# ensures tests validity
for region in MOCK_VISIBLE_REGIONS:
    assert (
        region in VALUES
    ), f"'{region}' in MOCK_VISIBLE_REGIONS must exist in VALUES dictionary"


@pytest.fixture
def heatmap_2d():
    """Fixture for 2D heatmap"""
    heatmap = bgh.Heatmap(
        VALUES,
        format="2D",
        position=1000,
        orientation="frontal",
        vmin=-5,
        vmax=3,
        check_latest=False,
        interactive=False,
    )
    yield heatmap
    heatmap.scene.close()


@pytest.mark.parametrize(
    "show_cbar,label_regions,cbar_label,expected_calls",
    [
        # show/hide colorbar
        (False, False, None, {"colorbar": 0, "set_ticks": 0, "set_label": 0}),
        (True, False, None, {"colorbar": 1, "set_ticks": 0, "set_label": 0}),
        # show_cbar=false overrides other settings
        (
            False,
            True,
            "Test Label",
            {"colorbar": 0, "set_ticks": 0, "set_label": 0},
        ),
        # test colormap region label
        (True, True, None, {"colorbar": 1, "set_ticks": 1, "set_label": 0}),
        # with cbar_label='str'
        (
            True,
            False,
            "Test Label",
            {"colorbar": 1, "set_ticks": 0, "set_label": 1},
        ),
        # all enabled
        (
            True,
            True,
            "Test Label",
            {"colorbar": 1, "set_ticks": 1, "set_label": 1},
        ),
    ],
)
def test_colorbar_functionality(
    heatmap_2d, show_cbar, label_regions, cbar_label, expected_calls
):
    """
    Tests colorbar functionality with different parameter combinations.

    Checks that:
    - colorbar display is controlled by show_cbar parameter.
    - colorbar tick values and labels match visible regions.
    - correct calls are made based on parameter combinations.
    """
    heatmap_2d.label_regions = label_regions

    mock_colorbar = MagicMock()

    with (
        patch.object(
            heatmap_2d.slicer,
            "get_structures_slice_coords",
            return_value=(MOCK_PROJECTED, None),
        ),
        patch(
            "matplotlib.figure.Figure.colorbar", return_value=mock_colorbar
        ) as mock_colorbar_fn,
        # prevent figure display
        patch("matplotlib.pyplot.show"),
    ):
        heatmap_2d.show(show_cbar=show_cbar, cbar_label=cbar_label)

        # verify call counts
        assert mock_colorbar_fn.call_count == expected_calls["colorbar"], (
            f"expecting colorbar"
            f"to be called {expected_calls['colorbar']} times"
        )
        assert (
            mock_colorbar.set_ticks.call_count == expected_calls["set_ticks"]
        ), (
            f"expecting colorbar"
            f"set_ticks to be called {expected_calls['set_ticks']} times"
        )
        assert (
            mock_colorbar.set_label.call_count == expected_calls["set_label"]
        ), (
            f"expecting colorbar"
            f"set_label to be called {expected_calls['set_label']} times"
        )

        # check region labels
        if label_regions and show_cbar:
            args, kwargs = mock_colorbar.set_ticks.call_args

            tick_values = kwargs.get("ticks")
            # expecting regions with values within (vmin vmax) range
            expected_regions = [
                r
                for r in MOCK_VISIBLE_REGIONS
                if heatmap_2d.vmin <= VALUES[r] <= heatmap_2d.vmax
            ]

            assert len(tick_values) == len(expected_regions), (
                "expecting colorbar "
                "tick values len() to match visible regions "
                "within (vmin vmax) range"
            )

            assert (
                "labels" in kwargs
            ), "expecting 'labels' parameter in colorbar set_ticks kwargs"
            region_labels = kwargs["labels"]
            assert set(region_labels) == set(expected_regions), (
                "expecting colorbar region labels to match "
                "visible regions within (vmin vmax) range"
            )
            assert "root" not in region_labels, (
                "expecting 'root' to be filtered out "
                "from colorbar region labels"
            )

            for i, region in enumerate(region_labels):
                assert tick_values[i] == VALUES[region], (
                    f"expecting colorbar tick value for"
                    f"'{region}' to match VALUES['{region}']"
                )
                assert heatmap_2d.vmin <= VALUES[region] <= heatmap_2d.vmax, (
                    f"expecting colorbar "
                    f"tick value {VALUES[region]} for region '{region}' "
                    f"to be within (vmin vmax) range "
                    f"[{heatmap_2d.vmin}, {heatmap_2d.vmax}]"
                )


@pytest.mark.parametrize(
    "test_case",
    [
        # no visible regions
        {
            "values": {"TH": 1.0, "VIS": 2.0},
            "projected": {},
            "label_regions": True,
            "description": "no visible regions",
        },
        # only root segment visible (empty colorbar)
        {
            "values": {"TH": 1.0, "VIS": 2.0},
            "projected": {
                "root_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            },
            "label_regions": True,
            "description": "only_root_segment",
        },
        # single value with segment_1
        {
            "values": {"TH": 1.0},
            "projected": {
                "TH_segment_1": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            },
            "label_regions": True,
            "description": "single value",
        },
        # values outside (vmin vmax) range
        {
            "values": {"TH": -6.2, "VIS": 4.1},  # <vmin && >vmax
            "projected": {
                "TH_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
                "VIS_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            },
            "label_regions": True,
            "description": "value outside range",
        },
        # mixed in/out of (vmin vmax) range
        {
            "values": {"TH": -6.2, "VIS": 1.1, "HIP": 5},
            "projected": {
                "TH_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
                "HIP_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
                "VIS_segment_0": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            },
            "label_regions": True,
            "description": "mixed range values",
        },
    ],
)
def test_colorbar_edge_cases(heatmap_2d, test_case):
    """
    Tests colorbar behavior in edge cases using parameterized test cases.
    - no visible regions.
    - only root segment visible.
    - single region value with segment_1.
    - value outside (vmin vmax) range.
    - mixed in/out of (vmin vmax) range.
    """
    values = test_case.get("values")
    heatmap_2d.label_regions = test_case.get("label_regions")
    heatmap_2d.values = values

    # verify regions exist in VALUES dictionary
    for region in values.keys():
        assert region in VALUES, (
            f"Test region '{region}' "
            f"must exist in VALUES dictionary for test validity"
        )

    # get projected data from test case
    mock_projected = test_case.get("projected", {})

    mock_colorbar = MagicMock()

    with (
        patch.object(
            heatmap_2d.slicer,
            "get_structures_slice_coords",
            return_value=(mock_projected, None),
        ),
        patch(
            "matplotlib.figure.Figure.colorbar", return_value=mock_colorbar
        ) as mock_colorbar_fn,
        # prevent figure display
        patch("matplotlib.pyplot.show"),
    ):
        heatmap_2d.show(show_cbar=True)

        if test_case.get("description") in [
            "no visible regions",
            "only_root_segment",
        ]:
            assert (
                mock_colorbar_fn.called
            ), "expecting colorbar to be created for empty cases"
            args, kwargs = mock_colorbar.set_ticks.call_args
            tick_values = kwargs.get("ticks")
            assert len(tick_values) == 0, (
                "expecting colorbar tick values to be empty"
                "for cases with no visible regions"
            )
            assert len(kwargs["labels"]) == 0, (
                "expecting colorbar tick labels to be empty"
                "for cases with no visible regions"
            )

        elif test_case.get("description") == "single value":
            assert mock_colorbar_fn.called, "expecting colorbar to be created"
            assert (
                mock_colorbar.set_ticks.called
            ), "expecting colorbar set_ticks to be called"
            args, kwargs = mock_colorbar.set_ticks.call_args
            tick_values = kwargs.get("ticks")
            region_labels = kwargs["labels"]

            assert len(tick_values) == 1, (
                "expecting colorbar tick values to be len() 1"
                "for single value cases"
            )
            assert len(region_labels) == 1, (
                "expecting colorbar tick labels to be len() 1"
                "for single value cases"
            )
            assert (
                region_labels[0] in values
            ), f"expecting '{region_labels[0]}' to be in test values"
            assert (
                tick_values[0] == values[region_labels[0]]
            ), "expecting tick value to match the test value of region"

        elif test_case.get("description") == "value outside range":
            assert mock_colorbar_fn.called, "expecting colorbar to be created"
            assert (
                mock_colorbar.set_ticks.called
            ), "expecting colorbar set_ticks to be called"
            args, kwargs = mock_colorbar.set_ticks.call_args
            tick_values = kwargs.get("ticks")
            region_labels = kwargs["labels"]

            assert len(tick_values) == 0, (
                "expecting no colorbar tick values "
                "when all values are outside (vmin vmax) range"
            )
            assert len(region_labels) == 0, (
                "expecting no colorbar tick labels "
                "when all values are outside (vmin vmax) range"
            )

        elif test_case.get("description") == "mixed range values":
            assert mock_colorbar_fn.called, "expecting colorbar to be created"
            assert (
                mock_colorbar.set_ticks.called
            ), "expecting colorbar set_ticks to be called"
            args, kwargs = mock_colorbar.set_ticks.call_args
            tick_values = kwargs.get("ticks")
            region_labels = kwargs["labels"]

            expected_regions = ["VIS"]  # Only in-range regions

            assert len(tick_values) == len(expected_regions), (
                "expecting colorbar tick values count to match "
                "only regions within (vmin vmax) range"
            )
            assert region_labels == expected_regions, (
                "expecting colorbar region labels to match "
                "only regions within (vmin vmax) range"
            )

            # Verify the values correspond to the right regions
            for i, region in enumerate(region_labels):
                assert tick_values[i] == values[region], (
                    f"expecting colorbar tick value for "
                    f"'{region}' to match VALUES['{region}']"
                )
                assert heatmap_2d.vmin <= VALUES[region] <= heatmap_2d.vmax, (
                    f"expecting colorbar "
                    f"tick value {VALUES[region]} for region '{region}' "
                    f"to be within (vmin vmax) "
                    f"range [{heatmap_2d.vmin}, {heatmap_2d.vmax}]"
                )
