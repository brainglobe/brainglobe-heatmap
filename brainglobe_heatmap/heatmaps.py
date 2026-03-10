from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from brainrender import Scene, cameras, settings
from brainrender.atlas import Atlas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myterial import grey_darker
from scipy.ndimage import gaussian_filter
from shapely import Polygon
from shapely.algorithms.polylabel import polylabel
from shapely.geometry.multipolygon import MultiPolygon
from skimage import measure
from vedo.colors import color_map as map_color

from brainglobe_heatmap.slicer import Slicer

# Set settings for heatmap visualization
settings.SHOW_AXES = False
settings.SHADER_STYLE = "cartoon"
settings.ROOT_ALPHA = 0.3
settings.ROOT_COLOR = grey_darker


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


# ── Annotation-based 2D helpers (fix for issue #103) ─────────────────────────


def _get_annotation_slice(
    atlas,
    position: float,
    orientation: Union[str, tuple],
) -> Tuple[np.ndarray, float, float]:
    """
    Extract a 2D slice from the atlas annotation volume.

    Returns
    -------
    sl : np.ndarray  shape (rows, cols)  — integer region-ID array
    res_row : float  — µm per pixel along rows
    res_col : float  — µm per pixel along cols
    """
    ann = atlas.annotation  # shape (AP, DV, ML), dtype uint32/int
    res = atlas.resolution  # (res_AP, res_DV, res_ML) in µm

    if isinstance(orientation, str):
        orientation = orientation.lower()

    if orientation in ("frontal", "coronal"):
        # slice along AP axis (axis 0)
        idx = int(round(position / res[0]))
        idx = int(np.clip(idx, 0, ann.shape[0] - 1))
        sl = ann[idx, :, :]  # shape (DV, ML)
        res_row, res_col = res[1], res[2]
    elif orientation == "sagittal":
        # slice along ML axis (axis 2)
        idx = int(round(position / res[2]))
        idx = int(np.clip(idx, 0, ann.shape[2] - 1))
        sl = ann[:, :, idx]  # shape (AP, DV)
        res_row, res_col = res[0], res[1]
    elif orientation == "horizontal":
        # slice along DV axis (axis 1)
        idx = int(round(position / res[1]))
        idx = int(np.clip(idx, 0, ann.shape[1] - 1))
        sl = ann[:, idx, :]  # shape (AP, ML)
        res_row, res_col = res[0], res[2]
    else:
        # custom normal vector — fall back to frontal
        idx = int(round(float(np.ravel(position)[0]) / res[0]))
        idx = int(np.clip(idx, 0, ann.shape[0] - 1))
        sl = ann[idx, :, :]
        res_row, res_col = res[1], res[2]

    return sl, res_row, res_col


def _build_id_to_value(values: dict, atlas) -> Dict[int, float]:
    """
    Build a mapping from region-ID → scalar value.

    Parent entries propagate to all descendant regions that don't already
    have their own explicit value (same semantics as the old mesh path where
    the whole CB mesh was coloured when 'CB' was in values).
    """
    acronym_to_id: Dict[str, int] = {
        s["acronym"]: s["id"] for s in atlas.structures_list
    }
    id_to_value: Dict[int, float] = {}

    for acronym, val in values.items():
        if np.isnan(val):
            continue
        rid = acronym_to_id.get(acronym)
        if rid is None:
            continue
        # Propagate to children first (own entry will overwrite below)
        try:
            for desc in atlas.get_structure_descendants(acronym):
                did = acronym_to_id.get(desc)
                if did is not None and did not in id_to_value:
                    id_to_value[did] = val
        except Exception:
            pass
        id_to_value[rid] = val

    return id_to_value


def _smooth_contour(xy: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Apply gaussian smoothing to a closed contour to remove pixel staircase.
    xy : shape (N, 2) in (row, col) order.
    """
    if len(xy) < 6:
        return xy
    # Wrap around so smoothing is continuous at the join
    pad = min(int(sigma * 4) + 1, len(xy) // 2)
    padded = np.concatenate([xy[-pad:], xy, xy[:pad]], axis=0)
    smooth_r = gaussian_filter(padded[:, 0].astype(float), sigma=sigma)
    smooth_c = gaussian_filter(padded[:, 1].astype(float), sigma=sigma)
    return np.column_stack([smooth_r[pad:-pad], smooth_c[pad:-pad]])


def _annotation_to_rgba(
    sl: np.ndarray,
    id_to_value: Dict[int, float],
    cmap,
    vmin: float,
    vmax: float,
    upsample: int = 4,
    fill_sigma: float = 2.0,
) -> np.ndarray:
    """
    Convert an integer annotation slice to an RGBA image.

    Steps
    -----
    1. Map each region-ID to a colour (heatmap colour if in values, neutral
       grey for anatomical context, transparent for background).
    2. Upsample 4x with nearest-neighbour (sharper than bilinear).
    3. Gaussian blur on heatmap regions only (smooth colour gradients).

    Returns
    -------
    rgba  : np.ndarray shape (rows*up, cols*up, 4)  float32 [0,1]
    """
    norm_cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    span = vmax - vmin if vmax != vmin else 1.0

    rows, cols = sl.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.float32)

    unique_ids = np.unique(sl)
    for rid in unique_ids:
        if rid == 0:
            continue
        mask = sl == rid
        if rid in id_to_value:
            norm_val = float(np.clip((id_to_value[rid] - vmin) / span, 0, 1))
            r, g, b, a = norm_cmap(norm_val)
            rgba[mask] = [r, g, b, 1.0]
        else:
            # Neutral grey — keeps anatomical context visible
            rgba[mask] = [0.78, 0.78, 0.78, 0.55]

    # Upsample with nearest-neighbour (preserves sharp region edges before
    # contours are drawn on top)
    if upsample > 1:
        from PIL import Image

        # Image.Resampling.NEAREST is canonical in Pillow >= 9.1;
        # fall back to Image.NEAREST for older installs.
        _nearest = getattr(Image, "Resampling", Image).NEAREST
        img_u8 = (rgba * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_u8, mode="RGBA")
        pil = pil.resize(
            (cols * upsample, rows * upsample),
            resample=_nearest,
        )
        rgba = np.asarray(pil).astype(np.float32) / 255.0
        rows, cols = rgba.shape[:2]

    # Gaussian blur only on heatmap-coloured pixels (alpha == 1)
    if fill_sigma > 0:
        hm_mask = (rgba[:, :, 3] > 0.9).astype(np.float32)
        for c in range(3):
            ch = rgba[:, :, c]
            blurred = gaussian_filter(ch * hm_mask, sigma=fill_sigma)
            weight = gaussian_filter(hm_mask, sigma=fill_sigma)
            rgba[:, :, c] = np.where(
                hm_mask > 0.5,
                blurred / np.where(weight > 1e-6, weight, 1.0),
                ch,
            )

    return rgba


# ── Main Heatmap class ───────────────────────────────────────────────────────


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
            Position of the plane in the atlas (µm).
        orientation : str or tuple, optional
            Orientation of the plane. "frontal", "sagittal", "horizontal"
            or a tuple with the normal vector. Default is "frontal".
        hemisphere : str, optional
            Hemisphere to display. Default is "both".
        title : str, optional
            Title of the heatmap. Default is None.
        cmap : str, optional
            Colormap. Default is "Reds".
        vmin : float, optional
            Minimum value for the colormap. Default is None.
        vmax : float, optional
            Maximum value for the colormap. Default is None.
        format : str, optional
            "3D" for brainrender or "2D" for matplotlib. Default is "3D".
        thickness : float, optional
            Thickness of the slicing plane (3D only). Default is 10.
        interactive : bool, optional
            If True, brainrender scene is interactive. Default is True.
        zoom : float, optional
            Zoom for brainrender. Default is None.
        atlas_name : str, optional
            BrainGlobe atlas name. Default is allen_mouse_25um.
        label_regions : bool, optional
            Label regions on colorbar (2D only). Default is False.
        annotate_regions : bool, List[str] or Dict, optional
            Controls region annotation. Default is False.
        annotate_text_options_2d : dict, optional
            matplotlib.text options for 2D annotations. Default is None.
        check_latest : bool, optional
            Check for latest atlas version. Default is True.
        **kwargs
            Additional arguments passed to the rendering/plotting function.
            For 2D plots (`format="2D"`), this accepts styling parameters:
            `upsample`, `fill_sigma`, `contour_sigma`, `contour_lw`,
            `contour_color`, `brain_outline_color`, `brain_outline_lw`
            (see `Heatmap.plot_subplot` for details).
        """
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

        # Store position (scalar or array) — used by both 2D and 3D paths
        self._position = position

        # create a scene (needed for both 2D and 3D: atlas lookup + 3D render)
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

        # prepare slicer object (3D path only, but harmless to build always)
        self.slicer = Slicer(position, orientation, thickness, self.scene.root)

    def prepare_colors(
        self,
        values: dict,
        cmap: str,
        vmin: Optional[float],
        vmax: Optional[float],
    ):
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

        Parameters
        ----------
        **kwargs
            Additional arguments passed to the rendering/plotting function.
            For 2D plots, this accepts styling parameters like `upsample`,
            `fill_sigma`, `contour_sigma`, `contour_lw`, `contour_color`,
            `brain_outline_color`, `brain_outline_lw`
            (see `Heatmap.plot_subplot` for full details).
        """
        if self.format == "3D":
            self.slicer.slice_scene(self.scene, self.regions_meshes)
            view = self.render(**kwargs)
        else:
            view = self.plot(**kwargs)
        return view

    def render(self, camera=None) -> Scene:
        """Renders the heatmap as a 3D brainrender scene (unchanged)."""
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
                self.scene.add_label(
                    actor=region_actor,
                    label=display_text,
                )

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
        """Plots the heatmap in 2D using matplotlib."""
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
        upsample: int = 4,
        fill_sigma: float = 2.0,
        contour_sigma: float = 1.5,
        contour_lw: float = 0.7,
        contour_color: str = "white",
        brain_outline_color: str = "#2b2b2b",
        brain_outline_lw: float = 1.5,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots a pixel-accurate 2D heatmap slice in the given axes.

        Uses atlas.annotation directly instead of mesh slicing, which fixes
        the asymmetric / split contours reported in issue #103.

        The visual pipeline (Hybrid Approach):
          1. Extract 2D slice from atlas.annotation (integer region-ID array)
          2. Map IDs → RGBA colours (heatmap cmap for values, grey for context)
          3. Upsample 4× with nearest-neighbour (crisp, no blur artefacts)
          4. Gaussian blur on heatmap fill only (smooth colour gradients)
          5. Draw skimage contours per region (pixel-perfect boundaries)
          6. Smooth contour paths with gaussian filter (removes staircase)
          7. Bold outer brain outline

        All existing parameters (cmap, vmin, vmax, annotate_regions,
        label_regions, show_legend, hide_axes, etc.) are preserved.

        Extra Parameters
        ----------------
        upsample : int
            Nearest-neighbour upsampling factor. Default 4.
        fill_sigma : float
            Gaussian sigma for heatmap colour blur (voxels). Default 2.0.
        contour_sigma : float
            Gaussian sigma for contour-path smoothing. Default 1.5.
        contour_lw : float
            Linewidth of region boundary contours. Default 0.7.
        contour_color : str
            Colour of region boundary contours. Default "white".
        brain_outline_color : str
            Colour of the outer brain boundary. Default "#2b2b2b".
        brain_outline_lw : float
            Linewidth of the outer brain boundary. Default 1.5.
        """
        atlas = self.scene.atlas

        # ── 1. Extract annotation slice ──────────────────────────────────────
        position_scalar = (
            float(np.ravel(self._position)[0])
            if not isinstance(self._position, (int, float))
            else float(self._position)
        )
        sl, res_row, res_col = _get_annotation_slice(
            atlas, position_scalar, self.orientation
        )

        # ── 2. Build id → value mapping (with child propagation) ─────────────
        id_to_value = _build_id_to_value(self.values, atlas)

        # ── 3. Build RGBA image (upsample + blur) ────────────────────────────
        rgba = _annotation_to_rgba(
            sl,
            id_to_value,
            self.cmap,
            self.vmin,
            self.vmax,
            upsample=upsample,
            fill_sigma=fill_sigma,
        )

        # Physical extent in µm for axis labels
        extent = [0, sl.shape[1] * res_col, sl.shape[0] * res_row, 0]

        ax.imshow(
            rgba,
            extent=extent,
            aspect="equal",
            interpolation="bilinear",
            origin="upper",
            zorder=0,
        )

        # ── 4. Draw region contours (pixel-perfect + smoothed) ───────────────
        # Work on the original (non-upsampled) sl for contour extraction —
        # skimage contours are in (row, col) units; convert to µm via res.
        for rid in np.unique(sl):
            if rid == 0:
                continue
            binary = (sl == rid).astype(np.uint8)
            contours = measure.find_contours(binary, level=0.5)
            for contour in contours:
                contour_s = _smooth_contour(contour, sigma=contour_sigma)
                xs = contour_s[:, 1] * res_col
                ys = contour_s[:, 0] * res_row
                ax.plot(
                    xs,
                    ys,
                    color=contour_color,
                    linewidth=contour_lw,
                    alpha=0.75,
                    solid_capstyle="round",
                    zorder=2,
                )

        # ── 5. Bold outer brain outline ──────────────────────────────────────
        brain_binary = (sl != 0).astype(np.uint8)
        for contour in measure.find_contours(brain_binary, level=0.5):
            contour_s = _smooth_contour(contour, sigma=contour_sigma)
            xs = contour_s[:, 1] * res_col
            ys = contour_s[:, 0] * res_row
            ax.plot(
                xs,
                ys,
                color=brain_outline_color,
                linewidth=brain_outline_lw,
                alpha=0.92,
                zorder=3,
            )

        # ── 6. Region annotations (same logic as before) ─────────────────────
        if self.annotate_regions:
            acronym_to_id = {
                s["acronym"]: s["id"] for s in atlas.structures_list
            }
            for acronym in self.values:
                display_text = self.get_region_annotation_text(acronym)
                if display_text is None:
                    continue
                rid = acronym_to_id.get(acronym)
                if rid is None:
                    continue

                # Build combined mask: parent ID + all descendant IDs
                # (parent regions like TH don't appear in annotation
                # voxels — only their children do)
                all_ids = {rid}
                try:
                    for desc in atlas.get_structure_descendants(acronym):
                        did = acronym_to_id.get(desc)
                        if did is not None:
                            all_ids.add(did)
                except Exception:
                    pass

                binary = np.isin(sl, list(all_ids)).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                # Build polygon from largest contour for polylabel
                contours = measure.find_contours(binary, level=0.5)
                if not contours:
                    continue
                largest = max(contours, key=len)
                coords_um = np.column_stack(
                    [
                        largest[:, 1] * res_col,
                        largest[:, 0] * res_row,
                    ]
                )
                pos = find_annotation_position_inside_polygon(coords_um)
                if pos is not None:
                    ax.annotate(
                        display_text,
                        xy=pos,
                        ha="center",
                        va="center",
                        zorder=4,
                        **(
                            self.annotate_text_options_2d
                            if self.annotate_text_options_2d is not None
                            else {}
                        ),
                    )

        # ── 7. Colorbar ──────────────────────────────────────────────────────
        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)

            if self.label_regions is True:
                cbar = fig.colorbar(
                    mpl.cm.ScalarMappable(
                        norm=None,
                        cmap=mpl.colormaps.get_cmap(
                            self.cmap, len(self.values)
                        ),
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

        # ── 8. Axis styling (unchanged from original) ────────────────────────
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
