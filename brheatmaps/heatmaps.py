from vedo.colors import colorMap as map_color
from typing import Optional, Union, Dict
import numpy as np
import matplotlib.pyplot as plt

from brainrender import Scene
from brainrender import settings, cameras

from brheatmaps.utils import check_values
from brheatmaps.planes import get_planes, get_plane_regions_intersections

# Set settings for heatmap visualization
settings.SHOW_AXES = False
settings.SHADER_STYLE = "cartoon"
settings.BACKGROUND_COLOR = "#242424"
settings.ROOT_ALPHA = 0.7
settings.ROOT_COLOR = "w"


class heatmap:
    def __init__(
        self,
        values: Dict[str, float],
        position: float = 0,
        orientation: Union[str, tuple] = "frontal",
        title: Optional[str] = None,
        cmap: str = "bwr",
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

        self.projected = dict()  # type: ignore

        # create a scene
        self.scene = Scene(
            atlas_name=atlas_name, title=title, title_color="w", **kwargs
        )

        # get brain regions colors
        _vmax, _vmin = check_values(values, self.scene.atlas)
        vmin = vmin or _vmin
        vmax = vmax or _vmax
        self.colors = {
            r: list(map_color(v, name=cmap, vmin=vmin, vmax=vmax))
            for r, v in values.items()
        }

        # get the position of planes to 'slice' thes cene
        self.plane0, self.plane1 = get_planes(
            self.scene,
            orientation=orientation,
            position=position,
            thickness=thickness,
        )

        # slice regions and get intersections
        self.slice()

    def show(self):
        # create visualization
        if self.format == "3D":
            return self.render()
        else:
            return self.plot()

    def slice(self):
        """
            It populates a brainrender scene with all the brain regions in the keys of 
            of the value dictionary and slices them with two planes.
            Optionally it can compute the 2D projection of the plane/region intersection points
            in the plane's coordinates system for 2D plotting.
        """
        # add brain regions to scene
        for region, value in self.values.items():
            self.scene.add_brain_region(region)

        regions = [
            r
            for r in self.scene.get_actors(br_class="brain region")
            if r.name != "root"
        ]

        if self.format == "2D":
            # get plane/regions intersections in plane's coordinates system
            self.projected = get_plane_regions_intersections(
                self.plane0, regions
            )
        else:
            # slice the scene
            for n, plane in enumerate((self.plane0, self.plane1)):
                self.scene.slice(plane, actors=regions, close_actors=True)

            self.scene.slice(
                self.plane0, actors=self.scene.root, close_actors=False
            )

    def render(self) -> Scene:
        """
            Renders the hetamap visualization as a 3D scene in brainrender.
        """

        # set brain regions colors
        for region, color in self.colors.items():
            self.scene.get_actors(br_class="brain region", name=region)[
                0
            ].color(color)

        # set camera position and render
        if isinstance(self.orientation, str):
            if self.orientation == "sagittal":
                camera = cameras.sagittal_camera2
            else:
                camera = self.orientation
        else:
            self.orientation = np.array(self.orientation)
            com = self.plane0.centerOfMass()
            camera = {
                "pos": com - self.orientation * 2 * np.linalg.norm(com),
                "viewup": (0, -1, 0),
                "clippingRange": (19531, 40903),
            }

        self.scene.render(
            camera=camera, interactive=self.interactive, zoom=self.zoom
        )
        return self.scene

    def plot(self) -> plt.Figure:
        """
            Plots the heatmap in 2D using matplotlib
        """
        f, ax = plt.subplots(figsize=(9, 9))
        for r, coords in self.projected.items():
            ax.fill(coords[:, 0], coords[:, 1], color=self.colors[r], label=r)

        ax.legend()
        plt.show()

        return f


if __name__ == "__main__":
    import matplotlib.pyplot as plt

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
        position=5200,  # displacement along the AP axis relative to midpoint
        orientation="frontal",  # or 'sagittal', or 'top' or a tuple (x,y,z)
        thickness=10,  # thickness of the slices used for rendering (in microns)
        title="frontal",
        vmin=-5,
        vmax=3,
    ).show()
