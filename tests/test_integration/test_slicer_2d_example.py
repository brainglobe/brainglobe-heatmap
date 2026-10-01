import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXAMPLE_SCRIPT = Path(__file__).parents[2] / "examples" / "slicer_2D.py"


# expects ax.images and ax.patches to be drawn
# expects ax.patches are drawn on ax.images
def test_region_contours_overlap_brain_reference():
    plt.close("all")
    try:
        runpy.run_path(str(EXAMPLE_SCRIPT))

        ax = plt.gcf().axes[0]

        assert len(ax.images) > 0, "No reference drawn"
        assert len(ax.patches) > 0, "No contour polygons drawn"

        left, right, bottom, top = ax.images[0].get_extent()
        x_min, x_max = sorted([left, right])
        y_min, y_max = sorted([bottom, top])

        for patch in ax.patches:
            verts = patch.get_path().vertices
            assert (
                verts[:, 0].min() >= x_min
            ), (
                "contour extends outside to left of image"
            )  # type: ignore[index]
            assert (
                verts[:, 0].max() <= x_max
            ), (
                "contour extends outside to right of image"
            )  # type: ignore[index]
            assert (
                verts[:, 1].min() >= y_min
            ), "contour extends outside to below image"  # type: ignore[index]
            assert (
                verts[:, 1].max() <= y_max
            ), "contour extends outside to above image"  # type: ignore[index]
    finally:
        plt.close("all")
