from typing import Union, Tuple
import numpy as np
from rich.panel import Panel
from rich.table import Table
from rich import print

from myterial import orange, amber

from vedo import Plane
from brainrender import Scene


def get_planes(
    scene: Scene,
    position: float = 0,
    orientation: Union[str, tuple] = "frontal",
    thickness: float = 100,
) -> Tuple[Plane, Plane]:
    """
    Returns the two planes used to slices the brainreder scene.
    The planes have different norms based on the desired orientation and
    they're thickness micrometers apart.
    """
    if isinstance(orientation, str):
        # get the index of the axis
        if orientation == "frontal":
            axidx = 0
        elif orientation == "sagittal":
            axidx = 2
        elif orientation == "top":
            axidx = 1
        else:
            raise ValueError(f'Orientation "{orientation}" not recognized')

        # get the two points the plances are cenered at
        shift = np.zeros(3)
        shift[axidx] -= thickness

        p0 = scene.root._mesh.centerOfMass()
        p1 = scene.root._mesh.centerOfMass()
        p1 -= shift

        # get the two planes
        norm0, norm1 = np.zeros(3), np.zeros(3)
        norm0[axidx] = 1
        norm1[axidx] = -1
    else:
        orientation = np.array(orientation)

        p0 = scene.root._mesh.centerOfMass()  # type: ignore
        p1 = p0 + orientation * thickness  # type: ignore

        norm0 = orientation  # type: ignore
        norm1 = -orientation  # type: ignore

    # get the length of the largest dimension of the atlas
    bounds = scene.root.bounds()
    length = max(
        bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4],
    )

    plane0 = scene.atlas.get_plane(
        pos=p0, norm=tuple(norm0), sx=length, sy=length
    )
    plane1 = scene.atlas.get_plane(
        pos=p1, norm=tuple(norm1), sx=length, sy=length
    )

    return plane0, plane1


def print_plane(name: str, plane: Plane, color: str):
    """
        Prints nicely formatted information about a plane
    """

    def fmt_array(x: np.ndarray) -> str:
        return str(tuple([round(v, 2) for v in x]))

    # create a table to display the vertices posittion
    vert_tb = Table(box=None)
    vert_tb.add_column(style=f"{amber}", justify="right")
    vert_tb.add_column(style="white")

    for i in range(4):
        vert_tb.add_row(f"({i})", fmt_array(plane.mesh.points()[i]))

    tb = Table(box=None)
    tb.add_column(style=f"bold {orange}", justify="right")
    tb.add_column(style="white")

    tb.add_row("center point: ", fmt_array(plane.mesh.center))
    tb.add_row("norm: ", str(tuple(plane.mesh.normal)))
    tb.add_row("Vertices: ", vert_tb)

    print(
        Panel.fit(
            tb, title=f"[white bold]{name}", style=color, title_align="left"
        )
    )


def get_plane_regions_intersections(
    plane: Plane, regions_actors: list, **kwargs
) -> dict:
    """
    It computes the intersection between the first slice plane and all brain regions,
    returning the coordinates of each region as a set of XY (i.e. in the plane's
    coordinates system) coordinates
    """
    pts = plane.points() - plane.points()[0]
    v = pts[1] / np.linalg.norm(pts[1])
    w = pts[2] / np.linalg.norm(pts[2])

    M = np.vstack([v, w]).T  # 3 x 2

    projected = {}
    for n, actor in enumerate(regions_actors):
        # get region/plane intersection
        intersection = plane.intersectWith(
            actor._mesh.triangulate()
        )  # points: (N x 3)

        if not intersection.points().shape[0]:
            continue  # no intersection

        pieces = intersection.splitByConnectivity()
        for piece_n, piece in enumerate(pieces):

            # sort coordinates
            points = piece.join(reset=True).points()

            projected[actor.name + f"_segment_{piece_n}"] = points @ M

    return projected
