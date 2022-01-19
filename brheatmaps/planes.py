from typing import Union, Tuple
import numpy as np

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

    plane0 = scene.atlas.get_plane(pos=p0, norm=tuple(norm0))
    plane1 = scene.atlas.get_plane(pos=p1, norm=tuple(norm1))

    return plane0, plane1


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
