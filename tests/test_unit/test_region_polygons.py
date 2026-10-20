import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch

from brainglobe_heatmap.heatmaps import polygon_to_path, region_polygons


def square(x0, y0, size):
    return np.array(
        [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]],
        dtype=float,
    )


def test_inner_contour_becomes_hole():
    (polygon,), labels = region_polygons([square(0, 0, 10), square(4, 4, 2)])
    assert len(polygon.interiors) == 1
    assert polygon.area == 100 - 4

    # render it: the hole must stay unfilled
    fig = plt.figure(figsize=(1, 1), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")
    ax.add_patch(PathPatch(polygon_to_path(polygon), fc="k", lw=0))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    assert img[50, 50, 0] == 255  # hole centre is white
    assert img[90, 10, 0] == 0  # ring is black


def test_separate_contours_stay_separate():
    polygons, labels = region_polygons([square(0, 0, 2), square(10, 0, 2)])
    assert len(polygons) == 2
    assert all(not p.interiors for p in polygons)


def test_degenerate_contours_skipped():
    assert region_polygons([np.zeros((2, 2))]) == ([], [])


def test_partially_overlapping_contours_stay_filled():
    # e.g. left/right hemisphere contours overlapping at the midline
    (polygon,), labels = region_polygons([square(0, 0, 10), square(9, 0, 10)])
    assert not polygon.interiors
    assert polygon.area == 19 * 10
    assert len(labels) == 2  # one per hemisphere


def test_island_inside_hole_is_filled():
    polygons, labels = region_polygons(
        [square(0, 0, 10), square(2, 2, 6), square(4, 4, 2)]
    )
    assert sorted(p.area for p in polygons) == [4, 100 - 36]
    assert len(labels) == 2  # ring and island


def test_sliver_touching_container_is_not_a_hole():
    # mesh cut at the midline leaves a sliver sharing the cut edge
    (polygon,), labels = region_polygons(
        [square(0, 0, 10), np.array([[9, 2], [10, 2], [10, 8], [9, 8]], float)]
    )
    assert not polygon.interiors
    assert len(labels) == 1


def test_self_intersecting_contour_keeps_all_area():
    bowtie = np.array([[0, 0], [2, 2], [2, 0], [0, 2]], dtype=float)
    polygons, labels = region_polygons([bowtie])
    assert sum(p.area for p in polygons) == 2


def test_spiked_self_intersecting_contour_keeps_all_area():
    # make_valid returns a collection holding a MultiPolygon and a line
    spiked = np.array(
        [[0, 0], [2, 2], [2, 0], [0, 2], [0, 0], [5, 5]], dtype=float
    )
    polygons, labels = region_polygons([spiked])
    assert sum(p.area for p in polygons) == 2
