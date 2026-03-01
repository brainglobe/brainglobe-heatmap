import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from brainrender import Scene, cameras, settings
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myterial import grey_darker
from shapely import Polygon
from shapely.algorithms.polylabel import polylabel
from shapely.geometry.multipolygon import MultiPolygon
from vedo.colors import color_map as map_color

from brainglobe_heatmap.slicer import Slicer

settings.SHOW_AXES = False
settings.SHADER_STYLE = "cartoon"
settings.ROOT_ALPHA = 0.3
settings.ROOT_COLOR = grey_darker


def parse_values(values):
    """
    Splits values dict into bilateral (scalar) and per_hemisphere (dict) parts.

    Parameters
    ----------
    values : dict
        Keys are region acronyms. Values are either:
        - float/int: same value for both hemispheres
        - dict with "left" and/or "right" keys: hemisphere-specific values

    Returns
    -------
    bilateral : dict
    per_hemisphere : dict
    """
    bilateral = {}
    per_hemisphere = {}
    for region, val in values.items():
        if isinstance(val, dict):
            if not val.keys() <= {"left", "right"}:
                raise ValueError(
                    f'Per-hemisphere dict for "{region}" may only contain '
                    f'"left" and/or "right" keys, got: {list(val.keys())}'
                )
            if not val:
                raise ValueError(
                    f'Per-hemisphere dict for "{region}" is empty.'
                )
            per_hemisphere[region] = val
        else:
            bilateral[region] = val
    return bilateral, per_hemisphere


def check_values(values, atlas):
    """
    Validates region names and value types.
    Returns global (vmax, vmin) across all values.
    """
    all_scalars = []
    for k, v in values.items():
        if k not in atlas.lookup_df.acronym.values:
            raise ValueError(f'Region name "{k}" not recognized')
        if isinstance(v, dict):
            for side, sv in v.items():
                if not isinstance(sv, (float, int)):
                    raise ValueError(
                        f"Heatmap values should be floats, "
                        f'not: {type(sv)} for entry "{k}[{side}]"'
                    )
                all_scalars.append(sv)
        else:
            if not isinstance(v, (float, int)):
                raise ValueError(
                    f"Heatmap values should be floats, "
                    f'not: {type(v)} for entry "{k}"'
                )
            all_scalars.append(v)
    not_nan = [v for v in all_scalars if not np.isnan(v)]
    if len(not_nan) == 0:
        return np.nan, np.nan
    return max(not_nan), min(not_nan)


def find_annotation_position_inside_polygon(polygon_vertices):
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


class Heatmap:
    def __init__(
        self,
        values,
        position,
        orientation="frontal",
        hemisphere="both",
        title=None,
        cmap="Reds",
        vmin=None,
        vmax=None,
        format="3D",
        thickness=10,
        interactive=True,
        zoom=None,
        atlas_name=None,
        label_regions=False,
        annotate_regions=False,
        annotate_text_options_2d=None,
        check_latest=True,
        **kwargs,
    ):
        """
        Creates a heatmap visualization of the provided values in 3D or 2D.

        Parameters
        ----------
        values : dict
            Keys are region acronyms. Values can be:
            - float/int: same color for both hemispheres (backwards compatible)
            - dict with "left"/"right" keys for hemisphere-specific colors

            Example::

                {
                    "TH": 1.0,
                    "VISp": {"left": 0.8, "right": 0.2},
                    "MOp": {"left": 0.5},
                }

        position : list, tuple, np.ndarray, or float
        orientation : str or tuple, optional
        hemisphere : str, optional
            Applies only to bilateral (scalar) regions. Default "both".
        title : str, optional
        cmap : str, optional
        vmin, vmax : float, optional
        format : str, optional. "3D" or "2D"
        thickness : float, optional
        interactive : bool, optional
        zoom : float, optional
        atlas_name : str, optional
        label_regions : bool, optional
        annotate_regions : bool, list, or dict, optional
        annotate_text_options_2d : dict, optional
        check_latest : bool, optional
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

        bilateral_values, per_hemisphere_values = parse_values(values)

        self.scene = Scene(
            atlas_name=atlas_name,
            title=title,
            title_color=grey_darker,
            check_latest=check_latest,
            **kwargs,
        )

        self.prepare_colors(values, cmap, vmin, vmax)

        # Add bilateral regions (original behaviour, backwards compatible)
        if bilateral_values:
            self.scene.add_brain_region(
                *bilateral_values.keys(), hemisphere=hemisphere
            )

        # Add per-hemisphere regions: one actor per requested side.
        # We add them bilaterally and cut manually to avoid brainrender's
        # get_plane() which has a numpy>=2.0 compat bug.
        for region, side_vals in per_hemisphere_values.items():
            for _ in side_vals:
                self.scene.add_brain_region(region, force=True)

        self.regions_meshes = [
            r
            for r in self.scene.get_actors(br_class="brain region")
            if r.name != "root"
        ]

        # Cut and rename per-hemisphere actors
        self._split_hemisphere_actors(per_hemisphere_values)

        # Map each actor -> color
        self._build_actor_color_map()

        self.slicer = Slicer(position, orientation, thickness, self.scene.root)

    def _get_midplane_center(self):
        """Returns root mesh CoM as the atlas midpoint."""
        return self.scene.root._mesh.center_of_mass()

    def _split_hemisphere_actors(self, per_hemisphere_values):
        """
        Cuts per-hemisphere region meshes to the correct side and renames
        each actor to "REGION__left" or "REGION__right" so the slicer can
        distinguish two actors that share the same region name.

        One actor is added per requested side in __init__, assigned in
        dict insertion order (left before right if both specified).

        Normals confirmed empirically with Allen Mouse atlas:
          normal=(0, 0,  1) -> keeps z > mid_z -> LEFT hemisphere
          normal=(0, 0, -1) -> keeps z < mid_z -> RIGHT hemisphere
        """
        if not per_hemisphere_values:
            return
        mesh_center = self._get_midplane_center()
        seen = {}
        for actor in self.regions_meshes:
            name = actor.name
            if name not in per_hemisphere_values:
                continue
            requested_sides = list(per_hemisphere_values[name].keys())
            seen[name] = seen.get(name, 0)
            side = requested_sides[seen[name]]
            seen[name] += 1
            actor.name = f"{name}__{side}"
            if side == "left":
                actor._mesh.cut_with_plane(
                    origin=mesh_center, normal=(0, 0, 1)
                )
                actor._mesh.cap()
            elif side == "right":
                actor._mesh.cut_with_plane(
                    origin=mesh_center, normal=(0, 0, -1)
                )
                actor._mesh.cap()

    def _build_actor_color_map(self):
        """
        Builds self.actor_colors: {actor -> color}.

        Per-hemisphere actors are named "REGION__side" after
        _split_hemisphere_actors,
        so we parse the side directly from the name — no CoM detection needed.
        Bilateral actors keep their plain region name.
        """
        self.actor_colors = {}
        for actor in self.regions_meshes:
            name = actor.name
            if name == "root":
                continue
            if "__" in name:
                region, side = name.rsplit("__", 1)
                self.actor_colors[actor] = self.colors.get(
                    f"{side}:{region}", settings.ROOT_COLOR
                )
            else:
                self.actor_colors[actor] = self.colors.get(
                    name, settings.ROOT_COLOR
                )

    def prepare_colors(self, values, cmap, vmin, vmax):
        """
        Builds self.colors flat dict:
        - "REGION" -> color  (bilateral)
        - "left:REGION" / "right:REGION" -> color  (per-hemisphere)
        - "root" -> ROOT_COLOR
        """
        _vmax, _vmin = check_values(values, self.scene.atlas)
        if _vmax == _vmin:
            _vmin = _vmax * 0.5
        vmin = vmin if vmin == 0 or vmin else _vmin
        vmax = vmax if vmax == 0 or vmax else _vmax
        self.vmin, self.vmax = vmin, vmax

        self.colors = {}
        for region, val in values.items():
            if isinstance(val, dict):
                for side, sv in val.items():
                    self.colors[f"{side}:{region}"] = list(
                        map_color(sv, name=cmap, vmin=vmin, vmax=vmax)
                    )
            else:
                self.colors[region] = list(
                    map_color(val, name=cmap, vmin=vmin, vmax=vmax)
                )
        self.colors["root"] = settings.ROOT_COLOR

    def get_region_annotation_text(self, region_name):
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

    def show(self, **kwargs):
        if self.format == "3D":
            self.slicer.slice_scene(self.scene, self.regions_meshes)
            view = self.render(**kwargs)
        else:
            view = self.plot(**kwargs)
        return view

    def render(self, camera=None):
        for actor, color in self.actor_colors.items():
            actor.color(color)
            # Strip __side suffix for annotation lookup
            display_name = (
                actor.name.split("__")[0] if "__" in actor.name else actor.name
            )
            display_text = self.get_region_annotation_text(display_name)
            if len(actor._mesh.vertices) > 0 and display_text is not None:
                self.scene.add_label(actor=actor, label=display_text)

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
        show_legend=False,
        xlabel="µm",
        ylabel="µm",
        hide_axes=False,
        filename=None,
        cbar_label=None,
        show_cbar=True,
        **kwargs,
    ):
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
        fig,
        ax,
        show_legend=False,
        xlabel="µm",
        ylabel="µm",
        hide_axes=False,
        cbar_label=None,
        show_cbar=True,
        **kwargs,
    ):
        projected, _ = self.slicer.get_structures_slice_coords(
            self.regions_meshes, self.scene.root
        )

        # actor_name_to_color maps full actor name (incl. __side suffix)
        # to color
        actor_name_to_color = {
            actor.name: color for actor, color in self.actor_colors.items()
        }

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
            color = actor_name_to_color.get(name, self.colors.get(name))
            # Strip __side suffix for display purposes
            display_name = name.split("__")[0] if "__" in name else name
            ax.fill(
                coords[:, 0],
                coords[:, 1],
                color=color,
                label=display_name
                if segment_nr == 0 and display_name != "root"
                else None,
                lw=1,
                ec="k",
                zorder=-1 if name == "root" else None,
                alpha=0.3 if name == "root" else None,
            )
            display_text = self.get_region_annotation_text(display_name)
            if display_text is not None:
                annotation_pos = find_annotation_position_inside_polygon(
                    coords
                )
                if annotation_pos is not None:
                    ax.annotate(
                        display_text,
                        xy=annotation_pos,
                        ha="center",
                        va="center",
                        **(
                            self.annotate_text_options_2d
                            if self.annotate_text_options_2d is not None
                            else {}
                        ),
                    )

        if show_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
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
