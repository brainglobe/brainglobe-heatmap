from typing import Union
import numpy as np
from vedo import Arrow

from myterial import pink_dark, blue_dark, amber_lighter, grey

from bgheatmaps.heatmaps import heatmap
from bgheatmaps.planes import print_plane

from brainrender import settings

settings.BACKGROUND_COLOR = amber_lighter
settings.ROOT_COLOR = grey


class plan(heatmap):
    def __init__(
        self,
        regions: Union[dict, list],
        position: float = 0,
        orientation: Union[str, tuple] = "frontal",
        thickness: float = 10,
        arrow_scale: float = 10,
        **kwagrs,
    ):
        self.arrow_scale = arrow_scale

        if isinstance(regions, list):
            regions = {r: 1 for r in regions}
        self.regions = regions

        super().__init__(
            regions,
            position=position,
            orientation=orientation,
            thickness=thickness,
            format="3D",
            **kwagrs,
        )

        # print planes information
        print_plane("Plane 0", self.plane0, blue_dark)
        print_plane("Plane 1", self.plane1, pink_dark)

    def slice(self):
        # add brain regions
        for region, value in self.regions.items():
            self.scene.add_brain_region(region)

    def show(self):
        """
            Renders the hetamap visualization as a 3D scene in brainrender.
        """
        self.scene.root._mesh.alpha(0.3)

        # show sliced brain regions
        for region in self.colors.keys():
            actor = self.scene.get_actors(
                br_class="brain region", name=region
            )[0]
            intersection = self.plane0.intersectWith(actor._mesh)

            if len(intersection.points()):
                intersection.lw(3)
                self.scene.add(intersection, transform=False)

            if region != "root":
                self.scene.remove(actor)

        # add slicing planes and their norms
        for plane, color, alpha in zip(
            (self.plane0, self.plane1), (blue_dark, pink_dark), (0.8, 0.3)
        ):
            plane.alpha(alpha).color(color)

            self.scene.add(plane, transform=False)
            self.scene.add(
                Arrow(
                    plane.center,
                    np.array(plane.center)
                    + np.array(plane.mesh.normal) * self.arrow_scale,
                    c=color,
                ),
                transform=False,
            )

        self.scene.render(interactive=self.interactive, zoom=self.zoom)
        return self.scene


if __name__ == "__main__":
    regions = dict(  # scalar values for each region
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

    plan(
        regions,
        position=5200,  # displacement along the AP axis relative to midpoint
        orientation="frontal",  # or 'sagittal', or 'top' or a tuple (x,y,z)
        thickness=2000,  # thickness of the slices used for rendering (in microns)
        arrow_scale=750,
    ).show()
