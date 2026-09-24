import runpy
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
from brainrender import settings

matplotlib.use("Agg")
settings.INTERACTIVE = False
settings.OFFSCREEN = True

EXAMPLES_DIR = Path(__file__).parents[2] / "examples"

NOT_TESTED = [
    "heatmap_human_brain.py",  # non-default atlas
    "heatmap_spinal_cord.py",  # non-default atlas
    "heatmap_zebrafish.py",  # non-default atlas
]

EXAMPLE = [
    "heatmap_2d.py",
    "heatmap_2d_subplots.py",
    "slicer_2D.py",
    "region_annotation.py",
    "region_annotation_specified.py",
    "cellfinder_cell_density.py",
    "get_coordinates.py",
    # 3D
    "heatmap_3d.py",
    "region_annotation_custom.py",
    "plan.py",
]


@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive:UserWarning"
)
@pytest.mark.parametrize(
    "example",
    EXAMPLE,
)
def test_examples(example):
    """confirms that examples don't crash."""
    plt.close("all")
    try:
        script = EXAMPLES_DIR / example
        assert script.exists(), f"Example not found: {script}"

        runpy.run_path(str(script))
    finally:
        plt.close("all")
