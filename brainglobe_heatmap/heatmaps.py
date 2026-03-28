import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from bg_atlasapi import BrainGlobeAtlas
from brainrender import Scene, cameras, settings
from brainrender.atlas import Atlas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myterial import grey_darker
from scipy.ndimage import center_of_mass, uniform_filter1d
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

# Set settings for transparent background
# vedo for transparent bg
# settings.vsettings.screenshot_transparent_background = True

# This needs to be false for transparent bg
# settings.vsettings.use_fxaa = False


def check_values(values: dict, atlas: Atlas) -> Tuple[float, float]:
    """
    Checks that the passed heatmap values meet two criteria:
        - keys should be acronyms of brainregions
        - values should be numbers
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
    Finds a suitable point for annotation within a polygon.

    Returns
    -------
    Tuple[float, float] or None
        A tuple (x, y) representing the point
        None if not enough vertices to form a valid polygon.

    Notes
    -----
    2D polygons only
    Edge cases:
    - Requires at least 4 vertices (< 4 returns None)
    - For invalid polygons, reconstructs the polygon using buffer(0),
      this resolves e.g., self-intersections
    - For some types of invalid geometries,
      buffer(0) may create a shapely MultiPolygon object by
      splitting self-intersecting areas into separate valid polygons.
      When this happens, the function gets the largest polygon by area.
    - Uses Shapely's polylabel algorithm with a tolerance of 0.1
      that accepts a polygon after edge cases are resolved.
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
    Returns the volume axis index corresponding to the given orientation.

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
    Extracts a 2D annotation slice directly from the atlas volume.

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
    Builds a mapping from numeric structure IDs to region acronyms.

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
    Builds binary pixel masks for each target region using a bottom-up
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
    Smooths the (Y, X) coordinates of a contour using a cyclic uniform filter.

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
    linewidth: float = 1.0,
    sigma: float = 1.5,
) -> None:
    """
    Draws smooth vectorial contours for all visible regions in the slice.

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
    linewidth : float, optional
        Width of the contour lines. Default is 1.0.
    sigma : float, optional
        Smoothing strength passed to :func:`_smooth_contour_path`.
        Default is 1.5.
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
                color="#333333",
                linewidth=linewidth,
                alpha=0.85,
                antialiased=True,
                solid_capstyle="round",
                solid_joinstyle="round",
            )


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
        format: str = "3D",  # 3D -> brainrender, 2D -> matplotlib
        # brainrender, 3D HM specific
        thickness: float = 10,
        interactive: bool = True,
        zoom: Optional[float] = None,
        atlas_name: Optional[str] = None,
        label_regions: Optional[Union[bool, List[str], Dict]] = False,
        annotate_regions: Optional[Union[bool, List[str], Dict]] = False,
        annotate_text_options_2d: Optional[Dict] = None,
        check_latest: bool = True,
        edge_smooth_sigma: float = 1.5,
        edge_linewidth: float = 1.0,
        **kwargs,
    ):
        """
        Creates a heatmap visualization of the provided values in 3D or 2D
        using brainrender or matplotlib in the specified atlas.

        Parameters
        ----------
        values : dict
            Dictionary with brain regions acronyms as keys and
            magnitudes as the values.
        position : list, tuple, np.ndarray, float
            Position of the plane in the atlas.
        orientation : str or tuple, optional
            Orientation of the plane in the atlas. Either, "frontal",
            "sagittal", "horizontal" or a tuple with the normal vector.
            Default is "frontal".
        hemisphere : str, optional
            Hemisphere to display the heatmap. Default is "both".
        title : str, optional
            Title of the heatmap. Default is None.
        cmap : str, optional
            Colormap to use for the heatmap. Default is "Reds".
        vmin : float, optional
            Minimum value for the colormap. Default is None.
        vmax : float, optional
            Maximum value for the colormap. Default is None.
        format : str, optional
            Format of the heatmap visualization.
            "3D" for brainrender or "2D" for matplotlib. Default is "3D".
        thickness : float, optional
            Thickness of the slicing plane in the brainrender scene.
            Default is 10.
        interactive : bool, optional
            If True, the brainrender scene is interactive. Default is True.
        zoom : float, optional
            Zoom level for the brainrender scene. Default is None.
        atlas_name : str, optional
            Name of the atlas to use for the heatmap.
            If None allen_mouse_25um is used. Default is None.
        label_regions :
            bool, List[str], Dict[str, str], optional
            Controls region labelling on the colorbar (2D only).
            If True, labels all visible regions.
            If a list, labels only the specified regions.
            If a dict, labels specified regions with custom text.
            Default is False.

        annotate_regions :
            bool, List[str], Dict[str, Union[str, float, int]], optional
            Controls region annotation in 2D and 3D format.
            If True, annotates all regions with their names.
            If a list, annotates only the specified regions.
            If a dict, uses custom text/values for annotations.
            Default is False.
        annotate_text_options_2d : dict, optional
            Options for customizing region annotations text in 2D format.
            matplotlib.text parameters
            Default is None
        check_latest : bool, optional
            Check for the latest version of the atlas. Default is True.
        """
        # store arguments
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
        self.edge_smooth_sigma = edge_smooth_sigma
        self.edge_linewidth = edge_linewidth
        self._position = position

        # create a scene
        self.scene = Scene(
            atlas_name=atlas_name,
            title=title,
            title_color=grey_darker,
            check_latest=check_latest,
            **kwargs,
        )

        # prep colors range
        self.prepare_colors(values, cmap, vmin, vmax)

        # add regions to the brainrender scene
        self.scene.add_brain_region(*self.values.keys(), hemisphere=hemisphere)
        self.regions_meshes = [
            r
            for r in self.scene.get_actors(br_class="brain region")
            if r.name != "root"
        ]

        # prepare slicer object
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
        """
        Prepares per-region colours from the provided colormap and value range.
        """
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
        Gets the annotation text for a region if it should be annotated.

        Returns
        -------
        None or str
            None if the region should not be annotated.

        Notes
        -----
        The behavior depends on the type of self.annotate_regions:
        - If bool: All regions except "root" are annotated when True
        - If list: Only regions in the list are annotated except "root"
        - If dict: Only regions in the dict keys are annotated,
          using dict values as display text
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
        """
        Creates a 2D plot or 3D rendering of the heatmap.
        """
        if self.format == "3D":
            self.slicer.slice_scene(self.scene, self.regions_meshes)
            return self.render(**kwargs)
        else:
            return self.plot(**kwargs)

    def render(self, camera=None) -> Scene:
        """
        Renders the heatmap as a 3D brainrender scene.

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
        Plots the heatmap in 2D using matplotlib.

        This method generates a 2D visualization of the heatmap data in
        a standalone matplotlib figure.

        Parameters
        ----------
        show_legend : bool, optional
            If True, displays a legend for the plotted regions.
            Default is False.
        xlabel : str, optional
            Label for the x-axis. Default is "µm".
        ylabel : str, optional
            Label for the y-axis. Default is "µm".
        hide_axes : bool, optional
            If True, hides the axes for a cleaner look. Default is False.
        filename : Optional[str], optional
            Path to save the figure to. If None, the figure is not saved.
            Default is None.
        cbar_label : Optional[str], optional
            Label for the colorbar. If None, no label is displayed.
            Default is None.
        show_cbar : bool, optional
            If True, displays a colorbar alongside the subplot.
            Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        plt.Figure
            The matplotlib figure object for the plot.

        Notes
        -----
        This method is used to generate a standalone plot of
        the heatmap data.
        """
        
        f, ax = plt.subplots(figsize=(9, 9))
        
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
            plt.savefig(filename, dpi=300)
            
        plt.show()
        return f

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
        Plots a heatmap in a subplot within a given figure and axes.

        This method is responsible for plotting a single subplot within a
        larger figure, allowing for the creation of complex multi-plot
        visualizations.

        Parameters
        ----------
        fig : plt.Figure, optional
            The figure object in which the subplot is plotted.
        ax : plt.Axes, optional
            The axes object in which the subplot is plotted.
        show_legend : bool, optional
            If True, displays a legend for the plotted regions.
            Default is False.
        xlabel : str, optional
            Label for the x-axis. Default is "µm".
        ylabel : str, optional
            Label for the y-axis. Default is "µm".
        hide_axes : bool, optional
            If True, hides the axes for a cleaner look. Default is False.
        cbar_label : Optional[str], optional
            Label for the colorbar. If None, no label is displayed.
            Default is None.
        show_cbar : bool, optional
            Display a colorbar alongside the subplot. Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        plt.Figure, plt.Axes
            A tuple containing the figure and axes objects used for the plot.

        Notes
        -----
        This method modifies the provided figure and axes objects in-place.
        """
        slice_data, section_idx = _get_slice_from_volume(
            self._bg_atlas, self._position, self.orientation
        )

        if slice_data is None:
            warnings.warn(
                "Custom orientation detected: falling back to legacy "
                "3D mesh projection (to prevent bug #103).",
                UserWarning,
                stacklevel=2,
            )
            return self._plot_subplot_legacy(
                fig, ax, show_legend, xlabel, ylabel,
                hide_axes, cbar_label, show_cbar, **kwargs,
            )

        id_to_acronym = _build_id_to_acronym(self._bg_atlas)
        unique_ids = np.unique(slice_data[slice_data > 0])

        # Calculate spatial extent in micrometres
        res = self._bg_atlas.resolution
        h_px, w_px = slice_data.shape
        if self.orientation == "frontal":
            extent = [0, w_px * res[2], h_px * res[1], 0]
        elif self.orientation == "horizontal":
            extent = [0, w_px * res[2], h_px * res[0], 0]
        else:  # sagittal
            extent = [0, w_px * res[0], h_px * res[1], 0]

        x_scale = (extent[1] - extent[0]) / w_px
        y_scale = (extent[2] - extent[3]) / h_px
        imshow_kw = dict(extent=extent, aspect="equal", origin="upper")

        # Build region masks using bottom-up ancestor traversal
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

        # Background: pale grey for all annotated voxels
        bg_canvas = np.zeros((*slice_data.shape, 4))
        bg_canvas[slice_data > 0] = (0.88, 0.88, 0.90, 0.30)
        ax.imshow(bg_canvas, **imshow_kw)

        # Regions of interest: continuous heatmap colour
        heatmap_arr = np.full(slice_data.shape, np.nan)
        for rname, value in self.values.items():
            mask = region_masks[rname]
            if mask.any():
                heatmap_arr[mask] = value

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
            linewidth=self.edge_linewidth,
            sigma=self.edge_smooth_sigma,
        )

        # Region annotations (original behaviour preserved)
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

        # Colorbar
        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
<<<<<<< HEAD
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax
            )
=======
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
>>>>>>> 3ba525f (test: boost coverage to ~75%+ and fix get_cmap() compatibility)
            if cbar_label is not None:
                cbar.set_label(cbar_label)

            if self.label_regions:
                unique_visible_regions = set()
                for uid in unique_ids:
                    acr = id_to_acronym.get(int(uid))
                    if acr and acr != "root":
                        unique_visible_regions.add(acr)

                if isinstance(self.label_regions, dict):
                    regions_to_label = (
                        set(self.label_regions.keys()) & unique_visible_regions
                    )
                elif isinstance(self.label_regions, list):
                    regions_to_label = (
                        set(self.label_regions) & unique_visible_regions
                    )
                else:
                    regions_to_label = unique_visible_regions

                tick_labels: list[str] = []
                tick_values: list[float] = []
                for region in regions_to_label:
                    value = self.values.get(region)
                    if value is None:
                        continue
                    if value > self.vmax or value < self.vmin:
                        continue
                    if isinstance(self.label_regions, dict):
                        tick_labels.append(str(self.label_regions[region]))
                    else:
                        tick_labels.append(region)
                    tick_values.append(value)

                cbar.set_ticks(ticks=tick_values, labels=tick_labels)

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
        This path retains the original behaviour including the known
        axis inversion for custom orientations (issue #103 is not
        applicable for non-axis-aligned slices).
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
                label=name if segment_nr == 0 and name != "root" else None,
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
<<<<<<< HEAD
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax
            )
=======
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
>>>>>>> 3ba525f (test: boost coverage to ~75%+ and fix get_cmap() compatibility)
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