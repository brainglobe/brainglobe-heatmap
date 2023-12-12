from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("brainglobe-heatmap")
except PackageNotFoundError:
    # package is not installed
    pass

from brainglobe_heatmap.heatmaps import heatmap
from brainglobe_heatmap.planner import plan
from brainglobe_heatmap.slicer import get_structures_slice_coords
