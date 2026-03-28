import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from bg_atlasapi import BrainGlobeAtlas
from brainrender import Scene, cameras, settings
from brainrender.atlas import Atlas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myterial import grey_darker
from scipy.ndimage import (
    center_of_mass,
    uniform_filter1d,
)
from scipy.ndimage import (
    label as ndlabel,
)
from shapely import Polygon
from shapely.algorithms.polylabel import polylabel
from shapely.geometry.multipolygon import MultiPolygon
from skimage.measure import find_contours
from vedo.colors import color_map as map_color

from brainglobe_heatmap.slicer import Slicer

# Set settings for heatmap visualization
settings.SHOW_AXES = False
settings.SHADER_STYLE = "cartoon"
settings.ROOT_ALPHA = 0.3
settings.ROOT_COLOR = grey_darker

# Distinct pastel colors for atlas-style region coloring,
# inspired by the Allen Brain Atlas color scheme.
ATLAS_REGION_COLORS = [
    "#AEC6CF",
    "#FFD1DC",
    "#FFDAC1",
    "#B5EAD7",
    "#C7CEEA",
    "#FFFACD",
    "#FFB7B2",
    "#E2F0CB",
    "#DDA0DD",
    "#87CEEB",
    "#F0E68C",
    "#98FB98",
    "#FFC0CB",
    "#D8BFD8",
    "#AFEEEE",
    "#FFDEAD",
    "#E6E6FA",
    "#F5DEB3",
    "#B0E0E6",
    "#FFE4E1",
]


# Utility functions


def check_values(values: dict, atlas: Atlas) -> Tuple[float, float]:
    """
    Check that the passed heatmap values meet two criteria:
        - keys should be acronyms of brain regions
        - values should be numbers

    Parameters
    ----------
    values : dict
        Dictionary mapping region acronyms to numeric values.
    atlas : Atlas
        BrainGlobe atlas object used for region name validation.

    Returns
    -------
    Tuple[float, float]
        A (vmax, vmin) tuple of the non-NaN values.
    """
    for k, v in values.items():
        if not isinstance(v, (float, int)):
            raise ValueError(
                f"Heatmap values should be floats, "
                f'not: {type(v)} for entry "{k}"'
            )
        if k not in atlas.lookup_df.acronym.values:
            raise ValueError(f'Region name "{k}" not recognized')

    not_nan = [v for v in values.values() if not np.isnan(v)]
    if len(not_nan) == 0:
        return np.nan, np.nan
    vmax, vmin = max(not_nan), min(not_nan)
    return vmax, vmin


def find_annotation_position_inside_polygon(
    polygon_vertices: np.ndarray,
) -> Union[Tuple[float, float], None]:
    """
    Find a suitable point for annotation within a polygon.

    Returns
    -------
    Tuple[float, float] or None
        A tuple (x, y) representing the annotation point,
        or None if there are not enough vertices to form a valid polygon.

    Notes
    -----
    2D polygons only.
    Edge cases:

    - Requires at least 4 vertices (fewer returns None).
    - For invalid polygons, reconstructs the polygon using buffer(0),
      which resolves e.g. self-intersections.
    - For some types of invalid geometries, buffer(0) may create a
      shapely MultiPolygon object by splitting self-intersecting areas
      into separate valid polygons. When this happens, the largest
      polygon by area is used.
    - Uses Shapely's polylabel algorithm with a tolerance of 0.1.
    """
    if polygon_vertices.shape[0] < 4:
        return None
    polygon = Polygon(polygon_vertices.tolist())
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.geom_type == "MultiPolygon" and isinstance(
        polygon, MultiPolygon
    ):
        polygon = max(polygon.geoms, key=lambda p: p.area)
    label_position = polylabel(polygon, tolerance=0.1)
    return label_position.x, label_position.y


# Internal helpers for annotation-based 2D plotting


def _get_orientation_axis(orientation: str) -> int:
    """
    Return the volume axis index corresponding to the given orientation.

    Parameters
    ----------
    orientation : str
        One of "frontal", "horizontal", or "sagittal".

    Returns
    -------
    int
        Axis index (0, 1, or 2).
    """
    return {"frontal": 0, "horizontal": 1, "sagittal": 2}[orientation]


def _get_slice_from_volume(
    atlas: BrainGlobeAtlas,
    position: Union[float, int, np.ndarray, list, tuple],
    orientation: Union[str, tuple],
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Extract a 2D annotation slice directly from the atlas volume.

    This replaces the 3D-to-2D mesh projection used in the original
    implementation (via Slicer / plane.py), which incorrectly inverted
    the medio-lateral axis (bug #103).

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The loaded BrainGlobe atlas.
    position : float, int, np.ndarray, list, or tuple
        Position of the slice in micrometres.
        Can be a scalar (converted to a slice index along the orientation
        axis) or a 3-element array [AP, DV, ML] (the relevant axis is
        extracted automatically).
    orientation : str or tuple
        Slice orientation. One of "frontal", "horizontal", "sagittal".
        A non-string value (e.g. a custom normal vector) is not supported
        and causes the function to return (None, None), triggering a
        fallback to the legacy mesh-projection path.

    Returns
    -------
    slice_data : np.ndarray or None
        2D annotation array of shape (H, W), or None for custom
        orientations.
    section_idx : int or None
        Index of the extracted slice along the orientation axis,
        or None for custom orientations.

    Notes
    -----
    The returned slice is clamped to the valid index range of the volume.
    """
    if not isinstance(orientation, str):
        return None, None

    axis = _get_orientation_axis(orientation)
    res = atlas.resolution[axis]

    if isinstance(position, (float, int, np.number)):
        section_idx = int(round(float(position) / res))
    elif hasattr(position, "__len__"):
        pos_arr = np.array(position)
        section_idx = int(round(float(pos_arr[axis]) / res))
    else:
        section_idx = int(round(float(position) / res))

    max_idx = atlas.annotation.shape[axis] - 1
    section_idx = max(0, min(section_idx, max_idx))

    if orientation == "frontal":
        return atlas.annotation[section_idx, :, :], section_idx
    elif orientation == "horizontal":
        return atlas.annotation[:, section_idx, :], section_idx
    else:  # sagittal
        return atlas.annotation[:, :, section_idx], section_idx


def _build_id_to_acronym(atlas: BrainGlobeAtlas) -> Dict[int, str]:
    """
    Build a mapping from numeric structure IDs to region acronyms.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The loaded BrainGlobe atlas.

    Returns
    -------
    Dict[int, str]
        Dictionary mapping each structure's numeric ID to its acronym.
    """
    mapping = {}
    for acronym, info in atlas.structures.items():
        try:
            mapping[int(info["id"])] = str(acronym)
        except (KeyError, ValueError, TypeError):
            continue
    return mapping


def _build_region_masks_bottomup(
    atlas: BrainGlobeAtlas,
    slice_data: np.ndarray,
    id_to_acronym: Dict[int, str],
    target_regions: List[str],
) -> Dict[str, np.ndarray]:
    """
    Build binary pixel masks for each target region using a bottom-up
    ancestor traversal.

    In the Allen Mouse Brain Atlas annotation volume, pixels are labelled
    with the IDs of fine-grained leaf structures (e.g. sub-lobules of the
    cerebellum). A parent region such as "CB" therefore has **no** pixels
    labelled with its own ID; all of its pixels carry the IDs of its
    descendants.

    The top-down approach (``get_structure_descendants``) is unreliable
    because it may silently omit levels of the hierarchy or raise
    exceptions, leaving masks empty and regions uncoloured.

    This function takes the opposite approach: for every unique ID present
    in the slice it calls ``get_structure_ancestors`` to retrieve the full
    lineage, then assigns the corresponding pixels to every target region
    that appears in that lineage.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The loaded BrainGlobe atlas.
    slice_data : np.ndarray
        2D annotation array as returned by :func:`_get_slice_from_volume`.
    id_to_acronym : Dict[int, str]
        Mapping from numeric IDs to acronyms, as returned by
        :func:`_build_id_to_acronym`.
    target_regions : List[str]
        Acronyms of the regions for which masks should be built.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping each target region acronym to a boolean mask
        of the same shape as ``slice_data``.
    """
    target_set = set(target_regions)
    masks = {r: np.zeros(slice_data.shape, dtype=bool) for r in target_regions}

    for uid in np.unique(slice_data[slice_data > 0]):
        uid_int = int(uid)
        if uid_int not in id_to_acronym:
            continue

        acr = id_to_acronym[uid_int]
        pixel_mask = slice_data == uid

        try:
            ancestors = atlas.get_structure_ancestors(acr)
            ancestor_set = set(ancestors) | {acr}
        except Exception:
            ancestor_set = {acr}

        for rname in target_set & ancestor_set:
            masks[rname] |= pixel_mask

    return masks


def _smooth_contour_path(coords: np.ndarray, sigma: float) -> np.ndarray:
    """
    Smooth the (Y, X) coordinates of a contour using a cyclic uniform filter.

    The path is smoothed in coordinate space rather than on the mask itself,
    so the contour stays within the true region boundary without bleeding
    into adjacent regions.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) with columns [row, col].
    sigma : float
        Smoothing strength. Larger values produce smoother contours.

    Returns
    -------
    np.ndarray
        Smoothed coordinate array of the same shape as ``coords``.
        Returned unchanged if the contour has fewer than 6 points.
    """
    if len(coords) < 6:
        return coords
    size = max(3, int(sigma * 4) | 1)
    n = len(coords)
    ys = np.tile(coords[:, 0], 3)
    xs = np.tile(coords[:, 1], 3)
    ys_s = uniform_filter1d(ys, size=size)[n : 2 * n]
    xs_s = uniform_filter1d(xs, size=size)[n : 2 * n]
    return np.column_stack([ys_s, xs_s])


def _draw_smooth_contours(
    ax: plt.Axes,
    slice_data: np.ndarray,
    unique_ids: np.ndarray,
    id_to_acronym: Dict[int, str],
    x_scale: float,
    y_scale: float,
    x0: float,
    y0: float,
    color: str,
    linewidth: float,
    alpha: float,
    sigma: float,
) -> None:
    """
    Draw smooth vectorial contours for all visible regions in the slice.

    Contours are computed using :func:`skimage.measure.find_contours` rather
    than a Laplacian (``np.gradient`` / ``np.roll``). The Laplacian operates
    pixel-by-pixel and produces jagged, staircase-like outlines.
    ``find_contours`` returns sub-pixel iso-contour paths that can then be
    smoothed in coordinate space via :func:`_smooth_contour_path`, yielding
    clean vectorial boundaries.


    Parameters
    ----------
    ax : plt.Axes
        Axes on which to draw the contours.
    slice_data : np.ndarray
        2D annotation array.
    unique_ids : np.ndarray
        Array of unique non-zero IDs present in the slice.
    id_to_acronym : Dict[int, str]
        Mapping from numeric IDs to acronyms.
    x_scale, y_scale : float
        Pixel-to-micrometre scale factors for the X and Y axes.
    x0, y0 : float
        Spatial origin offsets in micrometres.
    color : str
        Line colour for the contours.
    linewidth : float
        Width of the contour lines.
    alpha : float
        Opacity of the contour lines.
    sigma : float
        Smoothing strength passed to :func:`_smooth_contour_path`.
    """
    for uid in unique_ids:
        if int(uid) not in id_to_acronym:
            continue
        mask = (slice_data == uid).astype(np.uint8)
        for contour in find_contours(mask, 0.5):
            if len(contour) < 4:
                continue
            smooth = _smooth_contour_path(contour, sigma=sigma)
            x_um = np.append(
                smooth[:, 1] * x_scale + x0, smooth[0, 1] * x_scale + x0
            )
            y_um = np.append(
                smooth[:, 0] * y_scale + y0, smooth[0, 0] * y_scale + y0
            )
            ax.plot(
                x_um,
                y_um,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                antialiased=True,
                solid_capstyle="round",
                solid_joinstyle="round",
            )


def _draw_region_labels(
    ax: plt.Axes,
    slice_data: np.ndarray,
    region_masks: Dict[str, np.ndarray],
    regions_to_label: List[str],
    x_scale: float,
    y_scale: float,
    x0: float,
    y0: float,
    fontsize: float,
    color: str,
    draw_bbox: bool,
    min_area: int,
) -> None:
    """
    Draw region acronyms at the centre of mass of each connected component,
    in the style of the Allen Brain Atlas viewer.

    Parameters
    ----------
    ax : plt.Axes
        Axes on which to draw the labels.
    slice_data : np.ndarray
        2D annotation array (used for connected-component labelling).
    region_masks : Dict[str, np.ndarray]
        Pre-computed boolean masks, as returned by
        :func:`_build_region_masks_bottomup`.
    regions_to_label : List[str]
        Acronyms of the regions to label.
    x_scale, y_scale : float
        Pixel-to-micrometre scale factors.
    x0, y0 : float
        Spatial origin offsets in micrometres.
    fontsize : float
        Font size for the labels.
    color : str
        Text colour.
    draw_bbox : bool
        If True, draws a semi-transparent white box behind each label.
    min_area : int
        Minimum connected-component area in pixels below which no label
        is drawn.
    """
    for rname in regions_to_label:
        mask = region_masks.get(rname)
        if mask is None or not mask.any():
            continue

        labeled_mask, n_comp = ndlabel(mask)
        for comp_idx in range(1, n_comp + 1):
            comp = labeled_mask == comp_idx
            if comp.sum() < min_area:
                continue
            cy, cx = center_of_mass(comp)
            bbox_props = (
                dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.70,
                )
                if draw_bbox
                else None
            )
            ax.text(
                cx * x_scale + x0,
                cy * y_scale + y0,
                rname,
                fontsize=fontsize,
                color=color,
                ha="center",
                va="center",
                fontweight="bold",
                bbox=bbox_props,
                clip_on=True,
                zorder=10,
            )


# Heatmap class


class Heatmap:
    def __init__(
        self,
        values: Dict,
        position: Union[list, tuple, np.ndarray, float],
        orientation: Union[str, tuple] = "frontal",
        hemisphere: str = "both",
        title: Optional[str] = None,
        cmap: str = "Reds",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        format: str = "3D",
        thickness: float = 10,
        interactive: bool = True,
        zoom: Optional[float] = None,
        atlas_name: Optional[str] = None,
        label_regions: Optional[bool] = False,
        annotate_regions: Optional[Union[bool, List[str], Dict]] = False,
        annotate_text_options_2d: Optional[Dict] = None,
        check_latest: bool = True,
        # New parameters (annotation-based 2D plotting)
        color_mode: str = "heatmap",
        show_labels: bool = False,
        label_region_list: Optional[List[str]] = None,
        label_fontsize: float = 6.0,
        label_color: Optional[str] = None,
        label_bbox: bool = True,
        label_min_area: int = 300,
        edge_smooth_sigma: float = 1.5,
        edge_linewidth: float = 1.0,
        background_color: str = "white",
        **kwargs,
    ):
        """
        Create a heatmap visualization of the provided values in 3D or 2D
        using brainrender or matplotlib in the specified atlas.

        Parameters
        ----------
        values : dict
            Dictionary with brain region acronyms as keys and numeric
            magnitudes as values.
        position : list, tuple, np.ndarray, or float
            Position of the slicing plane in the atlas (in micrometres).
        orientation : str or tuple, optional
            Orientation of the plane. One of "frontal", "sagittal",
            "horizontal", or a tuple representing the normal vector.
            Default is "frontal".
        hemisphere : str, optional
            Hemisphere to display. Default is "both".
        title : str, optional
            Title of the figure. Default is None.
        cmap : str, optional
            Matplotlib colormap name. Default is "Reds".
        vmin : float, optional
            Minimum value for the colormap. Default is None (auto).
        vmax : float, optional
            Maximum value for the colormap. Default is None (auto).
        format : str, optional
            "3D" for a brainrender scene, "2D" for a matplotlib figure.
            Default is "3D".
        thickness : float, optional
            Thickness of the slicing plane for 3D rendering. Default is 10.
        interactive : bool, optional
            If True, the brainrender scene is interactive. Default is True.
        zoom : float, optional
            Zoom level for the brainrender scene. Default is None.
        atlas_name : str, optional
            BrainGlobe atlas identifier. Defaults to "allen_mouse_25um".
        label_regions : bool, optional
            If True, label regions on the colorbar (2D only).
            Default is False.
        annotate_regions : bool, List[str], or Dict, optional
            Controls region annotation in 2D and 3D.
            Default is False.
        annotate_text_options_2d : dict, optional
            matplotlib.text keyword arguments for 2D annotations.
            Default is None.
        check_latest : bool, optional
            Check for the latest atlas version. Default is True.
        color_mode : str, optional
            Colouring strategy for 2D plots.
            "heatmap"  – continuous colour gradient mapped to values
                         (default).
            "atlas"    – distinct pastel colours per region, Allen-style.
            "discrete" – one colour per region sampled from ``cmap``.
        show_labels : bool, optional
            If True, draw region acronyms on the 2D plot. Default is False.
        label_region_list : List[str], optional
            Regions to label. If None, all regions in ``values`` are
            labelled. Default is None.
        label_fontsize : float, optional
            Font size for region labels. Default is 6.0.
        label_color : str, optional
            Text colour for labels. Defaults to black on light backgrounds
            and white on dark backgrounds.
        label_bbox : bool, optional
            Draw a semi-transparent white box behind each label.
            Default is True.
        label_min_area : int, optional
            Minimum connected-component area (pixels) required to draw a
            label. Default is 300.
        edge_smooth_sigma : float, optional
            Smoothing strength for contour paths (0 = sharp, 2 = smooth).
            Default is 1.5.
        edge_linewidth : float, optional
            Width of the region boundary lines. Default is 1.0.
        background_color : str, optional
            Background colour of the figure. Default is "white".
        """
        # Store parameters
        self.values = values
        self.format = format
        self.orientation = orientation
        self.interactive = interactive
        self.zoom = zoom
        self.title = title
        self.cmap = cmap
        self.label_regions = label_regions
        self.annotate_regions = annotate_regions
        self.annotate_text_options_2d = annotate_text_options_2d
        self.color_mode = color_mode
        self.show_labels = show_labels
        self.label_region_list = label_region_list
        self.label_fontsize = label_fontsize
        self.label_color = label_color
        self.label_bbox = label_bbox
        self.label_min_area = label_min_area
        self.edge_smooth_sigma = edge_smooth_sigma
        self.edge_linewidth = edge_linewidth
        self.background_color = background_color
        self._position = position

        # Create brainrender scene
        self.scene = Scene(
            atlas_name=atlas_name,
            title=title,
            title_color=grey_darker,
            check_latest=check_latest,
            **kwargs,
        )

        self.prepare_colors(values, cmap, vmin, vmax)

        self.scene.add_brain_region(*self.values.keys(), hemisphere=hemisphere)
        self.regions_meshes = [
            r
            for r in self.scene.get_actors(br_class="brain region")
            if r.name != "root"
        ]

        self.slicer = Slicer(position, orientation, thickness, self.scene.root)

        # Load atlas via bg_atlasapi for direct volume access in 2D mode
        self._bg_atlas = BrainGlobeAtlas(
            atlas_name or "allen_mouse_25um",
            check_latest=check_latest,
        )

    def prepare_colors(
        self,
        values: dict,
        cmap: str,
        vmin: Optional[float],
        vmax: Optional[float],
    ) -> None:
        """Compute per-region colours from the provided colormap and value range."""
        _vmax, _vmin = check_values(values, self.scene.atlas)
        if _vmax == _vmin:
            _vmin = _vmax * 0.5
        vmin = vmin if vmin == 0 or vmin else _vmin
        vmax = vmax if vmax == 0 or vmax else _vmax
        self.vmin, self.vmax = vmin, vmax
        self.colors = {
            r: list(map_color(v, name=cmap, vmin=vmin, vmax=vmax))
            for r, v in values.items()
        }
        self.colors["root"] = settings.ROOT_COLOR

    def get_region_annotation_text(self, region_name: str) -> Union[None, str]:
        """
        Return the annotation text for a region, or None if it should not
        be annotated.

        Notes
        -----
        The behaviour depends on the type of ``self.annotate_regions``:

        - ``bool``: all regions except "root" are annotated when True.
        - ``list``: only regions in the list are annotated.
        - ``dict``: only regions in the dict keys are annotated, using
          dict values as display text.
        """
        if region_name == "root":
            return None
        should_annotate = (
            (isinstance(self.annotate_regions, bool) and self.annotate_regions)
            or (
                isinstance(self.annotate_regions, list)
                and region_name in self.annotate_regions
            )
            or (
                isinstance(self.annotate_regions, dict)
                and region_name in self.annotate_regions.keys()
            )
        )
        if not should_annotate:
            return None
        if isinstance(self.annotate_regions, dict):
            return str(self.annotate_regions[region_name])
        return region_name

    def show(self, **kwargs) -> Union[Scene, plt.Figure]:
        """Create a 2D plot or 3D rendering of the heatmap."""
        if self.format == "3D":
            self.slicer.slice_scene(self.scene, self.regions_meshes)
            return self.render(**kwargs)
        else:
            return self.plot(**kwargs)

    def render(self, camera=None) -> Scene:
        """
        Render the heatmap as a 3D brainrender scene.

        Parameters
        ----------
        camera : str or dict, optional
            brainrender camera specification. If None, derived from
            ``self.orientation``.

        Returns
        -------
        Scene
            The rendered brainrender scene.
        """
        for region, color in self.colors.items():
            if region == "root":
                continue
            region_actor = self.scene.get_actors(
                br_class="brain region", name=region
            )[0]
            region_actor.color(color)
            display_text = self.get_region_annotation_text(region_actor.name)
            if (
                len(region_actor._mesh.vertices) > 0
                and display_text is not None
            ):
                self.scene.add_label(actor=region_actor, label=display_text)

        if camera is None:
            if isinstance(self.orientation, str):
                if self.orientation == "sagittal":
                    camera = cameras.sagittal_camera2
                elif self.orientation == "horizontal":
                    camera = "top"
                else:
                    camera = self.orientation
            else:
                self.orientation = np.array(self.orientation)
                com = self.slicer.plane0.center_of_mass()
                camera = {
                    "pos": com - self.orientation * 2 * np.linalg.norm(com),
                    "viewup": (0, -1, 0),
                    "clipping_range": (19531, 40903),
                }
        self.scene.render(
            camera=camera, interactive=self.interactive, zoom=self.zoom
        )
        return self.scene

    def plot(
        self,
        show_legend: bool = False,
        xlabel: str = "µm",
        ylabel: str = "µm",
        hide_axes: bool = False,
        filename: Optional[str] = None,
        cbar_label: Optional[str] = None,
        show_cbar: bool = True,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot the heatmap in 2D using matplotlib.

        Parameters
        ----------
        show_legend : bool, optional
            Display a legend for the plotted regions. Default is False.
        xlabel : str, optional
            Label for the x-axis. Default is "µm".
        ylabel : str, optional
            Label for the y-axis. Default is "µm".
        hide_axes : bool, optional
            Hide axes for a cleaner look. Default is False.
        filename : str, optional
            Path to save the figure. Default is None (not saved).
        cbar_label : str, optional
            Label for the colorbar. Default is None.
        show_cbar : bool, optional
            Display a colorbar. Default is True.

        Returns
        -------
        plt.Figure
            The matplotlib figure.
        """
        f, ax = plt.subplots(figsize=(9, 9), facecolor=self.background_color)
        f, ax = self.plot_subplot(
            fig=f,
            ax=ax,
            show_legend=show_legend,
            xlabel=xlabel,
            ylabel=ylabel,
            hide_axes=hide_axes,
            cbar_label=cbar_label,
            show_cbar=show_cbar,
            **kwargs,
        )
        if filename is not None:
            plt.savefig(
                filename,
                dpi=300,
                bbox_inches="tight",
                facecolor=self.background_color,
            )
        plt.show()
        return f

    # Refactored 2D plotting (annotation-based)

    def plot_subplot(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        show_legend: bool = False,
        xlabel: str = "µm",
        ylabel: str = "µm",
        hide_axes: bool = False,
        cbar_label: Optional[str] = None,
        show_cbar: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a heatmap in a subplot within a given figure and axes.

        This method reads the atlas annotation volume directly instead of
        projecting 3D meshes onto a 2D plane. This fixes the left-right
        inversion caused by the sign flip on the medio-lateral axis in
        the original Slicer / plane.py pipeline (issue #103).

        Region masks are built using a bottom-up ancestor traversal
        (:func:`_build_region_masks_bottomup`), which correctly handles
        parent regions whose pixels in the volume are all carried by their
        leaf descendants.

        For custom (non-axis-aligned) orientations the method falls back
        to the original mesh-projection behaviour via
        :meth:`_plot_subplot_legacy`.

        Parameters
        ----------
        fig : plt.Figure
            Figure in which the subplot is drawn.
        ax : plt.Axes
            Axes on which the heatmap is drawn.
        show_legend : bool, optional
            Display a legend. Default is False.
        xlabel : str, optional
            X-axis label. Default is "µm".
        ylabel : str, optional
            Y-axis label. Default is "µm".
        hide_axes : bool, optional
            Hide axes ticks and spines. Default is False.
        cbar_label : str, optional
            Colorbar label. Default is None.
        show_cbar : bool, optional
            Display a colorbar (heatmap mode only). Default is True.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes objects after plotting.
        """
        # Read the 2D slice directly from the annotation volume
        slice_data, section_idx = _get_slice_from_volume(
            self._bg_atlas, self._position, self.orientation
        )

        if slice_data is None:
            warnings.warn(
                "Custom orientation detected: falling back to legacy "
                "3D mesh projection (To prevent bug #103).",
                UserWarning,
                stacklevel=2,
            )
            return self._plot_subplot_legacy(
                fig,
                ax,
                show_legend,
                xlabel,
                ylabel,
                hide_axes,
                cbar_label,
                show_cbar,
                **kwargs,
            )

        # Build ID
        id_to_acronym = _build_id_to_acronym(self._bg_atlas)
        unique_ids = np.unique(slice_data[slice_data > 0])
        visible_acronyms = [
            id_to_acronym[int(uid)]
            for uid in unique_ids
            if int(uid) in id_to_acronym
        ]

        # Compute spatial extent in micrometres
        res = self._bg_atlas.resolution
        h_px, w_px = slice_data.shape
        if self.orientation == "frontal":
            extent = [0, w_px * res[2], h_px * res[1], 0]
        elif self.orientation == "horizontal":
            extent = [0, w_px * res[2], h_px * res[0], 0]
        else:
            extent = [0, w_px * res[0], h_px * res[1], 0]

        x_scale = (extent[1] - extent[0]) / w_px
        y_scale = (extent[2] - extent[3]) / h_px
        imshow_kw = dict(extent=extent, aspect="equal", origin="upper")

        # Derive text/edge colours from background luminance
        bg_lum = np.mean(mcolors.to_rgb(self.background_color))
        text_color = "black" if bg_lum > 0.5 else "white"
        edge_color = "#333333" if bg_lum > 0.5 else "#aaaaaa"
        label_color = self.label_color or text_color
        ax.set_facecolor(self.background_color)

        # Build region masks
        region_masks = _build_region_masks_bottomup(
            self._bg_atlas, slice_data, id_to_acronym, list(self.values.keys())
        )

        for rname in self.values:
            if not region_masks[rname].any():
                warnings.warn(
                    f"Region '{rname}' has no pixels in slice {section_idx} "
                    f"(position {self._position} µm, "
                    f"orientation '{self.orientation}'). "
                    "Try a different position or orientation.",
                    UserWarning,
                    stacklevel=2,
                )

        # Background: all visible regions in a pale shade
        _cmap_obj = mpl.colormaps.get_cmap(self.cmap)
        bg_canvas = np.zeros((*slice_data.shape, 4))

        if self.color_mode in ("atlas", "discrete"):
            for i, acr in enumerate(visible_acronyms):
                if acr not in self._bg_atlas.structures:
                    continue
                mask = slice_data == self._bg_atlas.structures[acr]["id"]
                if not mask.any():
                    continue
                c = (
                    mcolors.to_rgba(
                        ATLAS_REGION_COLORS[i % len(ATLAS_REGION_COLORS)]
                    )
                    if self.color_mode == "atlas"
                    else _cmap_obj(i / max(len(visible_acronyms) - 1, 1))
                )
                bg_canvas[mask] = (*c[:3], 0.35)
        else:
            bg_canvas[slice_data > 0] = (0.88, 0.88, 0.90, 0.30)

        ax.imshow(bg_canvas, **imshow_kw)

        # Regions of interest in vivid colours
        roi_canvas = np.zeros((*slice_data.shape, 4))
        heatmap_arr = np.full(slice_data.shape, np.nan)
        legend_handles = []

        for i, (rname, value) in enumerate(self.values.items()):
            mask = region_masks[rname]
            if not mask.any():
                continue

            if self.color_mode == "heatmap":
                heatmap_arr[mask] = value

            elif self.color_mode == "atlas":
                idx_vis = next(
                    (j for j, a in enumerate(visible_acronyms) if a == rname),
                    i,
                )
                c = mcolors.to_rgba(
                    ATLAS_REGION_COLORS[idx_vis % len(ATLAS_REGION_COLORS)]
                )
                roi_canvas[mask] = (*c[:3], 0.92)
                legend_handles.append(
                    mpatches.Patch(color=mcolors.to_hex(c[:3]), label=rname)
                )

            else:  # discrete
                c = _cmap_obj(i / max(len(self.values) - 1, 1))
                roi_canvas[mask] = (*c[:3], 0.92)
                legend_handles.append(
                    mpatches.Patch(color=mcolors.to_hex(c[:3]), label=rname)
                )

        if self.color_mode in ("atlas", "discrete"):
            ax.imshow(roi_canvas, **imshow_kw)
        else:
            ax.imshow(
                np.ma.masked_invalid(heatmap_arr),
                cmap=self.cmap,
                norm=mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax),
                interpolation="nearest",
                alpha=0.92,
                **imshow_kw,
            )

        # Smooth vectorial contours
        _draw_smooth_contours(
            ax=ax,
            slice_data=slice_data,
            unique_ids=unique_ids,
            id_to_acronym=id_to_acronym,
            x_scale=x_scale,
            y_scale=y_scale,
            x0=extent[0],
            y0=extent[3],
            color=edge_color,
            linewidth=self.edge_linewidth,
            alpha=0.85,
            sigma=self.edge_smooth_sigma,
        )

        # Region labels
        if self.show_labels:
            _draw_region_labels(
                ax=ax,
                slice_data=slice_data,
                region_masks=region_masks,
                regions_to_label=(
                    self.label_region_list
                    if self.label_region_list is not None
                    else list(self.values.keys())
                ),
                x_scale=x_scale,
                y_scale=y_scale,
                x0=extent[0],
                y0=extent[3],
                fontsize=self.label_fontsize,
                color=label_color,
                draw_bbox=self.label_bbox,
                min_area=self.label_min_area,
            )

        # Region annotations (original behaviour)
        if self.annotate_regions:
            for rname in self.values:
                display_text = self.get_region_annotation_text(rname)
                if display_text is None:
                    continue
                mask = region_masks.get(rname)
                if mask is None or not mask.any():
                    continue
                cy, cx = center_of_mass(mask)
                ax.annotate(
                    display_text,
                    xy=(cx * x_scale + extent[0], cy * y_scale + extent[3]),
                    ha="center",
                    va="center",
                    **(self.annotate_text_options_2d or {}),
                )

        # Colorbar for heatmap mode only
        if show_cbar and self.color_mode == "heatmap":
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            if self.label_regions is True:
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(
                        norm=None,
                        cmap=mpl.colormaps.get_cmap(self.cmap, len(self.values)),
                    ),
                    cax=cax,
                )
                cbar.ax.set_yticklabels(
                    [r.strip() for r in self.values.keys()]
                )
            else:
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax
                )
            if cbar_label is not None:
                cbar.set_label(cbar_label)

        # Legend (atlas / discrete modes)
        if legend_handles and self.color_mode in ("atlas", "discrete"):
            ax.legend(
                handles=legend_handles,
                loc="lower right",
                fontsize=6.5,
                ncol=max(1, len(legend_handles) // 20),
                framealpha=0.7,
                title="Regions",
                title_fontsize=8,
            )

        # Axis styling
        ax.set(title=self.title)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if isinstance(self.orientation, str):
            ax.set(xlabel=xlabel, ylabel=ylabel)

        if hide_axes:
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set(xlabel="", ylabel="")

        if show_legend:
            ax.legend()

        return fig, ax

    def _plot_subplot_legacy(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        show_legend: bool,
        xlabel: str,
        ylabel: str,
        hide_axes: bool,
        cbar_label: Optional[str],
        show_cbar: bool,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Original 2D plotting via 3D mesh projection (Slicer).

        Used only as a fallback when ``orientation`` is a custom normal
        vector rather than one of the standard axis-aligned strings.
        """
        projected, _ = self.slicer.get_structures_slice_coords(
            self.regions_meshes, self.scene.root
        )
        segments = []
        for r, coords in projected.items():
            name, segment_nr = r.split("_segment_")
            x, y = coords[:, 0], coords[:, 1]
            area = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )
            segments.append(
                dict(
                    name=name,
                    segment_nr=int(segment_nr),
                    coords=coords,
                    area=area,
                )
            )
        segments.sort(key=lambda s: s["area"], reverse=True)

        for segment in segments:
            name = segment["name"]
            segment_nr = segment["segment_nr"]
            coords = segment["coords"]
            ax.fill(
                coords[:, 0],
                coords[:, 1],
                color=self.colors[name],
                label=name if segment_nr == "0" and name != "root" else None,
                lw=1,
                ec="k",
                zorder=-1 if name == "root" else None,
                alpha=0.3 if name == "root" else None,
            )
            display_text = self.get_region_annotation_text(str(name))
            if display_text is not None:
                pos = find_annotation_position_inside_polygon(coords)
                if pos is not None:
                    ax.annotate(
                        display_text,
                        xy=pos,
                        ha="center",
                        va="center",
                        **(self.annotate_text_options_2d or {}),
                    )

        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            if self.label_regions is True:
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(
                        norm=None,
                        cmap=mpl.colormaps.get_cmap(self.cmap, len(self.values)),
                    ), cax=cax)
                cbar.ax.set_yticklabels(
                    [r.strip() for r in self.values.keys()]
                )
            else:
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax
                )
            if cbar_label is not None:
                cbar.set_label(cbar_label)

        ax.invert_yaxis()
        ax.axis("equal")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set(title=self.title)
        if isinstance(self.orientation, str) or np.sum(self.orientation) == 1:
            ax.set(xlabel=xlabel, ylabel=ylabel)
        if hide_axes:
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set(xlabel="", ylabel="")
        if show_legend:
            ax.legend()
        return fig, ax
