"""
Run all default-atlas examples and save their visual output.

2D examples produce matplotlib figures compared by pytest-mpl
when --mpl is passed. Without --mpl they run as smoke tests.
3D examples produce brainrender screenshots (smoke tests only).

Output directory for 3D: test_example_outputs/ (gitignored).
"""

import os
import runpy
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest
from brainrender import settings

matplotlib.use("Agg")
settings.INTERACTIVE = False
settings.OFFSCREEN = True

EXAMPLES_DIR = Path(__file__).parents[2] / "examples"
OUTPUT_DIR = Path(__file__).parents[2] / "test_example_outputs"
BASELINE_DIR = str(Path(__file__).parent / "baseline")

EXAMPLES_2D = [
    "heatmap_2d.py",
    "heatmap_2d_subplots.py",
    "slicer_2D.py",
    "region_annotation.py",
    "region_annotation_specified.py",
    "cellfinder_cell_density.py",
]

EXAMPLES_3D = [
    "heatmap_3d.py",
    "region_annotation_custom.py",
]

SCENE_VAR = {
    "heatmap_3d.py": "scene",
    "region_annotation_custom.py": "f",
}


@pytest.fixture(scope="session", autouse=True)
def output_dir():
    """Create a clean output dir for 3D screenshots."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    yield OUTPUT_DIR


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR,
    tolerance=2,
    savefig_kwarg={"dpi": 150, "bbox_inches": "tight"},
    style="default",
)
@pytest.mark.parametrize(
    "example",
    EXAMPLES_2D,
    ids=EXAMPLES_2D,
)
def test_example_2d(example):
    """Run a 2D example; return its figure for comparison.

    With --mpl the returned figure is compared pixel-wise
    against the baseline. Without --mpl it is a smoke test.
    """
    script = EXAMPLES_DIR / example
    assert script.exists(), f"Example not found: {script}"

    plt.close("all")
    try:
        with patch.object(plt, "show"):
            runpy.run_path(str(script))

        fig = plt.gcf()
        assert fig.get_axes(), f"{example} produced no axes"
        return fig
    finally:
        # Clean up any leftover figures to avoid leaking.
        if plt.get_fignums():
            plt.close("all")


@pytest.mark.skipif(
    sys.platform != "linux" or not os.environ.get("DISPLAY"),
    reason="3D tests need xvfb (Linux only)",
)
@pytest.mark.parametrize(
    "example",
    EXAMPLES_3D,
    ids=EXAMPLES_3D,
)
def test_example_3d(example, output_dir):
    script = EXAMPLES_DIR / example
    assert script.exists(), f"Example not found: {script}"

    plt.close("all")
    try:
        with patch.object(plt, "show"):
            namespace = runpy.run_path(str(script))

        var = SCENE_VAR[example]
        scene = namespace.get(var)
        assert scene is not None, f"{example}: expected variable " f"'{var}'"

        stem = Path(example).stem
        filepath = str(output_dir / f"{stem}.png")
        scene.screenshot(filepath)
        scene.close()
    finally:
        plt.close("all")
