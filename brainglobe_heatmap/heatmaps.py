from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from brainrender import Scene, cameras, settings
from brainrender.atlas import Atlas
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myterial import grey_darker
from shapely import Polygon, get_parts, make_valid, union_all
from shapely.algorithms.polylabel import polylabel
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
        - values should be numbers, or dicts with "left" and/or "right"
          keys mapping to numbers
    """
    scalars = []
    for k, v in values.items():
        if isinstance(v, dict):
            if not v or not v.keys() <= {"left", "right"}:
                raise ValueError(
                    f'Per-hemisphere values for "{k}" must have "left" '
                    f'and/or "right" keys, got: {list(v.keys())}'
                )
            sides = list(v.items())
        else:
            sides = [(None, v)]

        for side, sv in sides:
            if not isinstance(sv, (float, int)):
                entry = k if side is None else f"{k}[{side}]"
                raise ValueError(
                    f"Heatmap values should be floats, "
                    f'not: {type(sv)} for entry "{entry}"'
                )
            scalars.append(sv)

        if k not in atlas.lookup_df.acronym.values:
            raise ValueError(f'Region name "{k}" not recognized')

    not_nan = [v for v in scalars if not np.isnan(v)]
    if len(not_nan) == 0:
        return np.nan, np.nan
    vmax, vmin = max(not_nan), min(not_nan)
    return vmax, vmin


def region_polygons(
    contours: List[np.ndarray],
) -> Tuple[List[Polygon], List[Polygon]]:
    """
    Combines a region's slice contours into polygons with holes.

    A mesh sliced by a plane gives one closed contour per boundary, so a
    region that surrounds another (e.g. ACB around aco) yields an outer
    contour plus an inner one. A contour nested inside an odd number of
    other contours is a hole. Contours at the same depth are unioned, so
    pieces that merely overlap stay filled.

    Returns
    -------
    polygons : list of Polygon
        Shapes to draw, with holes as interiors.
    label_areas : list of Polygon
        One area per filled contour (e.g. per hemisphere) to annotate.
    """
    shapes = []
    for coords in contours:
        if coords.shape[0] < 3:
            continue
        # repair self-intersections; make_valid may return a collection
        # holding a MultiPolygon, so flatten twice and drop stray lines
        valid = make_valid(Polygon(coords.tolist()))
        shapes += [
            p for p in get_parts(get_parts(valid)) if isinstance(p, Polygon)
        ]

    def nested(inner: Polygon, outer: Polygon) -> bool:
        # within 1 µm of the outline counts as overlapping, not nested,
        # e.g. slivers left where a mesh is cut at the midline
        return outer.contains(inner) and outer.exterior.distance(inner) > 1

    depths = defaultdict(list)
    for shape in shapes:
        depth = sum(
            other is not shape and nested(shape, other) for other in shapes
        )
        depths[depth].append(shape)

    merged = Polygon()
    for depth in sorted(depths):
        layer = union_all(depths[depth])
        merged = merged.difference(layer) if depth % 2 else merged.union(layer)
    polygons = [p for p in get_parts(merged) if not p.is_empty]

    label_areas = []
    for depth, layer in depths.items():
        if depth % 2:
            continue
        for shape in layer:
            # skip slivers mostly inside a larger contour at the same depth;
            # hemispheres overlapping at the midline share only a little
            if any(
                other.area > shape.area
                and shape.intersection(other).area > shape.area / 2
                for other in layer
            ):
                continue
            pieces = get_parts(merged.intersection(shape))
            pieces = [p for p in pieces if isinstance(p, Polygon)]
            if pieces:
                label_areas.append(max(pieces, key=lambda p: p.area))
    return polygons, label_areas


def polygon_to_path(polygon: Polygon) -> Path:
    """Matplotlib path for a polygon, with its interiors as holes."""
    return Path.make_compound_path(
        Path(np.asarray(polygon.exterior.coords), closed=True),
        *[Path(np.asarray(r.coords), closed=True) for r in polygon.interiors],
    )


def flatten_values(values: dict) -> Dict[str, float]:
    """
    Flattens per-hemisphere values so every actor name maps to one number:
    {"TH": 1, "VISp": {"left": 0.8}} -> {"TH": 1, "VISp__left": 0.8}
    """
    flat = {}
    for region, v in values.items():
        if isinstance(v, dict):
            for side, sv in v.items():
                flat[f"{region}__{side}"] = sv
        else:
            flat[region] = v
    return flat


def display_name(actor_name: str) -> str:
    """ "VISp__left" -> "VISp (left)", "TH" -> "TH" """
    acronym, _, side = actor_name.partition("__")
    return f"{acronym} ({side})" if side else acronym


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
        alpha: Optional[Union[float, Dict[str, float]]] = None,
        check_latest: bool = True,
        **kwargs,
    ):
        """
        Creates a heatmap visualization of the provided values in 3D or 2D
        using brainrender or matplotlib in the specified atlas.

        Parameters
        ----------
        values : dict
            Dictionary with brain regions acronyms as keys and
            magnitudes as the values. A value can also be a dict with
            "left" and/or "right" keys to show different magnitudes in
            each hemisphere, e.g. {"TH": 1.0, "VISp": {"left": 0.8,
            "right": 0.2}, "MOp": {"left": 0.5}}.
        position : list, tuple, np.ndarray, float
            Position of the plane in the atlas.
        orientation : str or tuple, optional
            Orientation of the plane in the atlas. Either, "frontal",
            "sagittal", "horizontal" or a tuple with the normal vector.
            Default is "frontal".
        hemisphere : str, optional
            Hemisphere to display the heatmap. Only applies to regions
            with a single (non per-hemisphere) value. Default is "both".
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
        alpha : float or dict, optional
            Transparency of brain regions in 3D format.
            If a float, the same alpha is applied to all regions.
            If a dict, maps region acronyms to individual alpha values.
            Values must be between 0.0 (fully transparent) and
            1.0 (fully opaque). Region acronyms need to be a subset of
            `values` argument. Regions not present in a dict keep
            their default opacity. Has no effect in 2D format.
            Default is None.
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

        # validate and store alpha
        if alpha is not None:
            if isinstance(alpha, (int, float)):
                if not 0.0 <= float(alpha) <= 1.0:
                    raise ValueError(
                        f"`alpha` must be between 0.0 and 1.0, got {alpha}"
                    )
            elif isinstance(alpha, dict):
                for region_name, alpha_val in alpha.items():
                    if not 0.0 <= float(alpha_val) <= 1.0:
                        raise ValueError(
                            f"`alpha` for region '{region_name}' must be "
                            f"between 0.0 and 1.0, got {alpha_val}"
                        )
                    if region_name not in values.keys():
                        raise ValueError(
                            f"`alpha` specified for region '{region_name}',"
                            f"but region was not specified in values."
                        )
            else:
                raise TypeError(
                    f"`alpha` must be a float or a dict mapping region "
                    f"acronyms to floats, got {type(alpha)}"
                )
        self.alpha = alpha

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
        bilateral = [r for r, v in values.items() if not isinstance(v, dict)]
        if bilateral:
            self.scene.add_brain_region(*bilateral, hemisphere=hemisphere)

        # per-hemisphere regions get one actor per side, renamed to
        # "REGION__side" so colors and slice segments stay distinct
        for region, sides in values.items():
            if isinstance(sides, dict):
                for side in sides:
                    actor = self.scene.add_brain_region(
                        region, hemisphere=side
                    )
                    actor.name = f"{region}__{side}"

        self.regions_meshes = [
            r
            for r in self.scene.get_actors(br_class="brain region")
            if r.name != "root"
        ]

        # prepare slicer object
        self.slicer = Slicer(position, orientation, thickness, self.scene.root)

    def prepare_colors(
        self,
        values: dict,
        cmap: str,
        vmin: Optional[float],
        vmax: Optional[float],
    ):
        # get brain regions colors
        _vmax, _vmin = check_values(values, self.scene.atlas)
        if _vmax == _vmin:
            _vmin = _vmax * 0.5

        vmin = vmin if vmin == 0 or vmin else _vmin
        vmax = vmax if vmax == 0 or vmax else _vmax
        self.vmin, self.vmax = vmin, vmax

        self.colors = {
            r: list(map_color(v, name=cmap, vmin=vmin, vmax=vmax))
            for r, v in flatten_values(values).items()
        }
        self.colors["root"] = settings.ROOT_COLOR

    def get_region_annotation_text(self, region_name: str) -> Union[None, str]:
        """
        Gets the annotation text for a region if it should be annotated

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

        # Determine what text to use for annotation
        if isinstance(self.annotate_regions, dict):
            return str(self.annotate_regions[region_name])

        return region_name

    def show(self, **kwargs) -> Union[Scene, plt.Figure]:
        """
        Creates a 2D plot or 3D rendering of the heatmap
        """
        if self.format == "3D":
            self.slicer.slice_scene(self.scene, self.regions_meshes)
            view = self.render(**kwargs)
        else:
            view = self.plot(**kwargs)
        return view

    def render(self, camera=None) -> Scene:
        """
        Renders the heatmap visualization as a 3D scene in brainrender.

        Parameters:
        ----------
        camera : str or dict, optional
            The `brainrender` camera to render the scene.
            If not provided, `self.orientation` is used.
        Returns:
        -------
        scene : Scene
            The rendered 3D scene.
        """

        # set brain regions colors and annotations
        for region, color in self.colors.items():
            if region == "root":
                continue
            region_actor = self.scene.get_actors(
                br_class="brain region", name=region
            )[0]
            region_actor.color(color)
            acronym = region.split("__")[0]

            # apply transparency if requested
            if self.alpha is not None:
                if isinstance(self.alpha, dict):
                    if acronym in self.alpha:
                        region_actor.alpha(float(self.alpha[acronym]))
                else:
                    region_actor.alpha(float(self.alpha))

            display_text = self.get_region_annotation_text(acronym)

            if (
                len(region_actor._mesh.vertices) > 0
                and display_text is not None
            ):
                self.scene.add_label(
                    actor=region_actor,
                    label=display_text,
                )

        if camera is None:
            # set camera position and render
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
        projected, _ = self.slicer.get_structures_slice_coords(
            self.regions_meshes, self.scene.root
        )

        region_contours = defaultdict(list)
        for r, coords in projected.items():
            region_contours[r.split("_segment_")[0]].append(coords)

        shapes = []
        label_areas = []
        for name, contours in region_contours.items():
            polygons, areas = region_polygons(contours)
            shapes += [(name, nr, p) for nr, p in enumerate(polygons)]
            label_areas += [(name, area) for area in areas]
        # Draw largest first so smaller regions stay on top
        shapes.sort(key=lambda s: s[2].area, reverse=True)

        for name, nr, polygon in shapes:
            ax.add_patch(
                PathPatch(
                    polygon_to_path(polygon),
                    facecolor=self.colors[name],
                    label=(
                        display_name(name)
                        if nr == 0 and name != "root"
                        else None
                    ),
                    lw=1,
                    ec="k",
                    zorder=-1 if name == "root" else None,
                    alpha=0.3 if name == "root" else None,
                )
            )

        for name, area in label_areas:
            display_text = self.get_region_annotation_text(name.split("__")[0])
            if display_text is not None:
                annotation_pos = polylabel(area, tolerance=0.1)
                ax.annotate(
                    display_text,
                    xy=(annotation_pos.x, annotation_pos.y),
                    ha="center",
                    va="center",
                    **(
                        self.annotate_text_options_2d
                        if self.annotate_text_options_2d is not None
                        else {}
                    ),
                )

        if show_cbar:
            # make colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # cmap = mpl.cm.cool
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax
            )

            if cbar_label is not None:
                cbar.set_label(cbar_label)

            if self.label_regions:
                unique_visible_regions = set()
                for r in projected.keys():
                    name = r.split("_segment_")[0]
                    if name != "root":
                        unique_visible_regions.add(name)

                if isinstance(self.label_regions, (dict, list)):
                    regions_to_label = {
                        r
                        for r in unique_visible_regions
                        if r.split("__")[0] in self.label_regions
                    }
                else:
                    regions_to_label = unique_visible_regions

                flat_values = flatten_values(self.values)
                tick_labels: list[str] = []
                tick_values: list[float] = []
                for region in regions_to_label:
                    value = flat_values[region]
                    if value > self.vmax or value < self.vmin:
                        continue
                    if isinstance(self.label_regions, dict):
                        acronym, _, side = region.partition("__")
                        label = str(self.label_regions[acronym])
                        tick_labels.append(
                            f"{label} ({side})" if side else label
                        )
                    else:
                        tick_labels.append(display_name(region))
                    tick_values.append(value)

                cbar.set_ticks(ticks=tick_values, labels=tick_labels)

        # style axes
        ax.invert_yaxis()
        ax.axis("equal")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.set(title=self.title)

        if isinstance(self.orientation, str) or np.sum(self.orientation) == 1:
            # orthogonal projection
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
