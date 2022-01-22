from vedo.colors import colorMap as map_color
from typing import Optional, Union, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

from myterial import grey_darker

from brainrender import Scene
from brainrender import settings, cameras
from brainrender.atlas import Atlas

from bgheatmaps.slicer import Slicer

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
                f'Heatmap values should be floats, not: {type(v)} for entry "{k}"'
            )

        if k not in atlas.lookup_df.acronym.values:
            raise ValueError(f'Region name "{k}" not recognized')

    vmax, vmin = max(values.values()), min(values.values())
    return vmax, vmin


class heatmap:
    def __init__(
        self,
        values: Dict[str, float],
        position: Union[list, tuple, np.ndarray],
        orientation: Union[str, tuple] = "frontal",
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
        **kwargs,
    ):
        # store arguments
        self.values = values
        self.format = format
        self.orientation = orientation
        self.interactive = interactive
        self.zoom = zoom
        self.title = title
        self.cmap = cmap

        # create a scene
        self.scene = Scene(
            atlas_name=atlas_name,
            title=title,
            title_color=grey_darker,
            **kwargs,
        )

        # prep colors range
        self.prepare_colors(values, cmap, vmin, vmax)

        # add regions to the brainrender scene
        self.scene.add_brain_region(*self.values.keys())

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

        vmin = vmin or _vmin
        vmax = vmax or _vmax
        self.vmin, self.vmax = vmin, vmax

        self.colors = {
            r: list(map_color(v, name=cmap, vmin=vmin, vmax=vmax))
            for r, v in values.items()
        }
        self.colors["root"] = grey_darker

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

    def render(self, **kwargs) -> Scene:
        """
            Renders the hetamap visualization as a 3D scene in brainrender.
        """

        # set brain regions colors
        for region, color in self.colors.items():
            if region == "root":
                continue

            self.scene.get_actors(br_class="brain region", name=region)[
                0
            ].color(color)

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
            com = self.slicer.plane0.centerOfMass()
            camera = {
                "pos": com - self.orientation * 2 * np.linalg.norm(com),
                "viewup": (0, -1, 0),
                "clippingRange": (19531, 40903),
            }

        self.scene.render(
            camera=camera, interactive=self.interactive, zoom=self.zoom
        )
        return self.scene

    def plot(
        self,
        show_legend: bool = False,
        xlabel: str = "μm",
        ylabel: str = "μm",
        **kwargs,
    ) -> plt.Figure:
        """
            Plots the heatmap in 2D using matplotlib
        """
        self.scene.close()
        projected, _ = self.slicer.get_structures_slice_coords(
            self.regions_meshes, self.scene.root
        )

        f, ax = plt.subplots(figsize=(9, 9))
        for r, coords in projected.items():
            name, segment = r.split("_segment_")
            ax.fill(
                coords[:, 0],
                coords[:, 1],
                color=self.colors[name],
                label=name if segment == "0" and name != "root" else None,
                lw=1,
                ec="k",
                zorder=-1 if name == "root" else None,
                alpha=0.3 if name == "root" else None,
            )

        # make colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # cmap = mpl.cm.cool
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap), cax=cax)

        # style axes
        ax.invert_yaxis()
        ax.axis("equal")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set(title=self.title)

        if isinstance(self.orientation, str) or np.sum(self.orientation) == 1:
            # orthogonal projection
            ax.set(xlabel=xlabel, ylabel=ylabel)

        if show_legend:
            ax.legend()
        plt.show()

        return f


if __name__ == "__main__":
    values = dict(  # scalar values for each region
        TH=1,
        RSP=0.2,
        AI=0.4,
        SS=-3,
        MO=2.6,
        PVZ=-4,
        LZ=-3,
        VIS=2,
        AUD=0.3,
        RHP=-0.2,
        STR=0.5,
        CB=0.5,
        FRP=-1.7,
        HIP=3,
        PA=-4,
    )

    heatmap(
        values,
        position=None,
        orientation=(
            1,
            1,
            0,
        ),  # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
        thickness=250,  # thickness of the slices used for rendering (in microns)
        title="frontal",
        vmin=-5,
        vmax=3,
        format="3D",
    ).show()
