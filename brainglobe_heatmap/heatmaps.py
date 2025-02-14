from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.path as pltpath
import matplotlib.pyplot as plt
import numpy as np
from brainrender import Scene, cameras, settings
from brainrender.atlas import Atlas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myterial import grey_darker
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
    polygon_vertices: np.ndarray, precision: float = 1.0
) -> Tuple[float, float]:
    """
    Find the optimal position for placing an annotation inside a polygon.
    Known as the pole of inaccessibility, the point inside the polygon that is
    farthest from any of its edges.

    The algorithm works as follows:
      1. Determine a bounding box around the polygon and generate an
         initial coarse grid of candidate points within that box.
      2. For each candidate point, compute the distance to the
         closest polygon edge if the point is inside the polygon.
      3. Use a priority queue (max-heap) that have the
         greatest potential to contain a point with a larger distance.
      4. Repeatedly extract the cell with the
         largest potential distance from the queue:
      5. Continue until no cells can offer an improvement within precision.

    Parameters
    ----------
    polygon_vertices : np.ndarray of shape (N, 2)
        Array of (x, y) coordinates defining the polygon vertices.
    precision : float, optional
        The precision tolerance for the result (default: 1.0).
        A smaller value yields higher accuracy.

    Returns
    -------
    tuple
        The (x, y) coordinates of the optimal annotation position.
    """

    def calculate_point_to_edges_distance(
        px: float, py: float, vertices: np.ndarray
    ) -> float:
        """
        Calculate the minimum distance from a point to the edges of a polygon.

        Returns
        -------
        float
            Minimum distance from (x, y) to any edge if inside the polygon,
            float('-inf') if outside the polygon.
        """
        point = np.array([px, py])

        if not polygon.contains_point((px, py)):
            return float("-inf")

        segments = np.vstack((vertices, vertices[0]))
        segment_starts = segments[:-1]
        segment_ends = segments[1:]

        edges = segment_ends - segment_starts
        lengths_sq = np.sum(edges**2, axis=1)
        valid_edges = lengths_sq > 0
        min_dist = float("inf")

        # find the minimum distance from a point to a line segment
        if np.any(valid_edges):
            segment_ratio = (
                np.sum(
                    (point - segment_starts[valid_edges]) * edges[valid_edges],
                    axis=1,
                )
                / lengths_sq[valid_edges]
            )
            segment_ratio = np.clip(segment_ratio, 0, 1)

            closest = (
                segment_starts[valid_edges]
                + (edges[valid_edges].T * segment_ratio).T
            )
            distances = np.sqrt(np.sum((point - closest) ** 2, axis=1))
            min_dist = min(min_dist, np.min(distances))

        if np.any(~valid_edges):
            closest = segment_starts[~valid_edges]
            vertex_distances = np.sqrt(np.sum((point - closest) ** 2, axis=1))
            if len(vertex_distances) > 0:
                min_dist = min(min_dist, np.min(vertex_distances))

        return min_dist

    if not isinstance(polygon_vertices, np.ndarray):
        polygon_vertices = np.asarray(polygon_vertices)

    polygon = pltpath.Path(polygon_vertices)

    # determine the bounding box of the polygon
    minx, miny = np.min(polygon_vertices, axis=0)
    maxx, maxy = np.max(polygon_vertices, axis=0)
    width = maxx - minx
    height = maxy - miny

    # bigger divisor makes initial grid denser
    # can help on very narrow polygons
    cell_size = min(width, height) / 4
    cell_radius = cell_size / 2

    # create an initial grid of sample points throughout the bounding box
    x_coords = np.arange(minx + cell_radius, maxx, cell_size)
    y_coords = np.arange(miny + cell_radius, maxy, cell_size)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # determine which points of the initial grid lie inside the polygon
    inside_mask = polygon.contains_points(points)
    inside_points = points[inside_mask]

    cell_queue: List[Tuple[float, float, float, float, float]] = []
    for center_x, center_y in inside_points:
        distance = calculate_point_to_edges_distance(
            center_x, center_y, polygon_vertices
        )
        max_potential = distance + cell_radius * np.sqrt(2)
        # simulate max-heap behavior storing negative values
        heappush(
            cell_queue,
            (-max_potential, distance, center_x, center_y, cell_radius),
        )

    # start with center of the bounding box
    bbox_x = minx + width / 2
    bbox_y = miny + height / 2
    bbox_distance = calculate_point_to_edges_distance(
        bbox_x, bbox_y, polygon_vertices
    )
    best_distance = bbox_distance
    best_x, best_y = bbox_x, bbox_y

    while cell_queue:
        max_potential, distance, x, y, cell_radius = heappop(cell_queue)
        max_potential = -max_potential

        if max_potential - best_distance <= precision:
            break

        if distance > best_distance:
            best_distance = distance
            best_x = x
            best_y = y

        # only subdivide further if the cell is large enough
        if cell_radius > precision / 2:
            new_cell_radius = cell_radius / 2
            # four new sub-cells
            for dx, dy in [
                (-new_cell_radius, -new_cell_radius),
                (new_cell_radius, -new_cell_radius),
                (-new_cell_radius, new_cell_radius),
                (new_cell_radius, new_cell_radius),
            ]:
                new_x = x + dx
                new_y = y + dy

                # compute distance for the center of the new sub-cell
                new_distance = calculate_point_to_edges_distance(
                    new_x, new_y, polygon_vertices
                )
                if new_distance != float("-inf"):
                    new_max_potential = (
                        new_distance + new_cell_radius * np.sqrt(2)
                    )
                    # push into the queue if potentially can beat best_distance
                    if new_max_potential > best_distance + precision:
                        heappush(
                            cell_queue,
                            (
                                -new_max_potential,
                                new_distance,
                                new_x,
                                new_y,
                                new_cell_radius,
                            ),
                        )

    return best_x, best_y


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
        label_regions: Optional[bool] = False,
        annotate_regions: Optional[Union[bool, List[str], Dict]] = False,
        annotate_text_options: Optional[Dict] = None,
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
        label_regions : bool, optional
            If True, labels the regions on the colorbar (only valid in 2D).
            Default is False.
        annotate_regions :
            bool, List[str], Dict[str, Union[str, float, int]], optional
            Controls region annotation in 2D format.
            If True, annotates all regions with their names.
            If a list, annotates only the specified regions.
            If a dict, uses custom text/values for annotations.
            Default is False.
        annotate_text_options : dict, optional
            Options for customizing region annotations text.
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
        self.annotate_text_options = annotate_text_options

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
            for r, v in values.items()
        }
        self.colors["root"] = settings.ROOT_COLOR

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

        # set brain regions colors
        for region, color in self.colors.items():
            if region == "root":
                continue

            self.scene.get_actors(br_class="brain region", name=region)[
                0
            ].color(color)

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

        segments: List[Dict[str, Union[str, np.ndarray, float]]] = []
        for r, coords in projected.items():
            name, segment_nr = r.split("_segment_")
            x = coords[:, 0]
            y = coords[:, 1]
            # calculate area of polygon with Shoelace formula
            area = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )

            segments.append(
                {
                    "name": name,
                    "segment_nr": segment_nr,
                    "coords": coords,
                    "area": area,
                }
            )

        # Sort region segments by area (largest first)
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

            should_annotate = (
                (
                    isinstance(self.annotate_regions, bool)
                    and self.annotate_regions
                )
                or (
                    isinstance(self.annotate_regions, list)
                    and name in self.annotate_regions
                )
                or (
                    isinstance(self.annotate_regions, dict)
                    and name in self.annotate_regions.keys()
                )
            )
            if should_annotate and self.format == "2D":
                if name != "root":
                    display_text = (
                        str(self.annotate_regions[name])
                        if isinstance(self.annotate_regions, dict)
                        else name
                    )
                    ax.annotate(
                        display_text,
                        xy=find_annotation_position_inside_polygon(
                            coords, precision=0.1
                        ),
                        ha="center",
                        va="center",
                        **(
                            self.annotate_text_options
                            if self.annotate_text_options is not None
                            else {}
                        ),
                    )

        if show_cbar:
            # make colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # cmap = mpl.cm.cool
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            if self.label_regions is True:
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(
                        norm=None,
                        cmap=mpl.cm.get_cmap(self.cmap, len(self.values)),
                    ),
                    cax=cax,
                )
            else:
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax
                )

            if cbar_label is not None:
                cbar.set_label(cbar_label)

            if self.label_regions is True:
                cbar.ax.set_yticklabels(
                    [r.strip() for r in self.values.keys()]
                )

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
