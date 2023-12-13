from typing import Union

import numpy as np
from brainrender import settings
from myterial import (
    amber_lighter,
    blue_dark,
    green_dark,
    grey,
    orange,
    pink_dark,
    red_dark,
)
from rich import print
from rich.panel import Panel
from rich.table import Table
from vedo import Arrow, Sphere

from brainglobe_heatmap.heatmaps import heatmap
from brainglobe_heatmap.plane import Plane

settings.BACKGROUND_COLOR = amber_lighter
settings.ROOT_COLOR = grey


def print_plane(name: str, plane: Plane, color: str):
    """
    Prints nicely formatted information about a plane
    """

    def fmt_array(x: np.ndarray) -> str:
        return str(tuple([round(v, 2) for v in x]))

    tb = Table(box=None)
    tb.add_column(style=f"bold {orange}", justify="right")
    tb.add_column(style="white")

    tb.add_row("center point: ", fmt_array(plane.center))
    tb.add_row("norm: ", str(tuple(plane.normal)))
    tb.add_row("u: ", str(tuple(plane.u)))
    tb.add_row("v: ", str(tuple(plane.v)))

    print(
        Panel.fit(
            tb, title=f"[white bold]{name}", style=color, title_align="left"
        )
    )


class plan(heatmap):
    def __init__(
        self,
        regions: Union[dict, list],
        position: Union[list, tuple, np.ndarray],
        orientation: Union[str, tuple] = "frontal",
        thickness: float = 10,
        arrow_scale: float = 10,
        **kwargs,
    ):
        self.arrow_scale = arrow_scale

        if isinstance(regions, list):
            regions = {r: 1 for r in regions}

        super().__init__(
            regions,
            position=position,
            orientation=orientation,
            thickness=thickness,
            format="3D",
            **kwargs,
        )

        # print planes information
        print_plane("Plane 0", self.slicer.plane0, blue_dark)
        print_plane("Plane 1", self.slicer.plane1, pink_dark)

    def show(self):
        """
        Renders the hetamap visualization as a 3D scene in brainrender.
        """
        self.scene.root._mesh.alpha(0.3)

        # show sliced brain regions
        self.slicer.show_plane_intersection(
            self.scene, self.regions_meshes, self.scene.root
        )

        # add slicing planes and their norms
        for plane, color, alpha in zip(
            (self.slicer.plane0, self.slicer.plane1),
            (blue_dark, pink_dark),
            (0.8, 0.3),
        ):
            plane_mesh = plane.to_mesh(self.scene.root)
            plane_mesh.alpha(alpha).color(color)

            self.scene.add(plane_mesh, transform=False)
            for vector, v_color in zip(
                (plane.normal, plane.u, plane.v), (color, red_dark, green_dark)
            ):
                self.scene.add(
                    Arrow(
                        plane.center,
                        np.array(plane.center)
                        + np.array(vector) * self.arrow_scale,
                        c=v_color,
                    ),
                    transform=False,
                )

            self.scene.add(
                Sphere(plane_mesh.center, r=plane_mesh.width / 125, c="k"),
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
        # position of the center point of the plane
        position=(
            8000,
            5000,
            5000,
        ),
        # or 'sagittal', or 'horizontal' or a tuple (x,y,z)
        orientation="frontal",
        # thickness of the slices used for rendering (in microns)
        thickness=2000,
        arrow_scale=750,
    ).show()
