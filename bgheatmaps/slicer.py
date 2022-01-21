from typing import List, Dict, Union
from vedo import Plane
import numpy as np

from brainrender.actor import Actor
from brainrender.scene import Scene


class Slicer:
    def __init__(
        self,
        position: Union[list, tuple, np.ndarray],
        orientation: Union[str, tuple],
        thickness: float,
        root: Actor,
    ):
        """
            Computes the position of two planes given a point (position) and an orientation (named orientation or 
            3D vector) + thickness (spacing between the two planes)
        """
        position = np.array(position)
        position[2] = -position[2]

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
            p1 = position - shift

            # get the two planes
            norm0, norm1 = np.zeros(3), np.zeros(3)
            norm0[axidx] = 1
            norm1[axidx] = -1
        else:
            orientation = np.array(orientation)

            p1 = position + orientation * thickness  # type: ignore

            norm0 = orientation  # type: ignore
            norm1 = -orientation  # type: ignore

        # get the length of the largest dimension of the atlas
        bounds = root.bounds()
        length = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        )

        self.plane0 = Actor(
            Plane(pos=position, normal=norm0, sx=length, sy=length),
            name=f"Plane at {position} norm: {norm0}",
            br_class="plane",
        )
        self.plane0.width = length

        self.plane1 = Actor(
            Plane(pos=p1, normal=norm1, sx=length, sy=length),
            name=f"Plane at {p1} norm: {norm1}",
            br_class="plane",
        )
        self.plane1.width = length

    def get_plane_coordinates(self, regions: List[Actor], root: Actor):
        """
        It computes the intersection between the first slice plane and all 
        user given brain regions,
        returning the coordinates of each region as a set of XY (i.e. in the plane's
        coordinates system) coordinates
        """
        pts = self.plane0.points() - self.plane0.points()[0]
        v = pts[1] / np.linalg.norm(pts[1])
        w = pts[2] / np.linalg.norm(pts[2])

        M = np.vstack([v, w]).T  # 3 x 2

        projected: Dict[str, np.ndarray] = {}
        for n, actor in enumerate(regions + [root]):
            # get region/plane intersection
            intersection = self.plane0.intersectWith(
                actor._mesh.triangulate()
            )  # points: (N x 3)

            if not intersection.points().shape[0]:
                continue  # no intersection

            pieces = intersection.splitByConnectivity()
            for piece_n, piece in enumerate(pieces):

                # sort coordinates
                points = piece.join(reset=True).points()

                projected[actor.name + f"_segment_{piece_n}"] = points @ M

        # get output coordinates
        coordinates: Dict[str, List[np.ndarray]] = dict()
        for region in projected.keys():
            coordinates[region] = [
                v for k, v in projected.items() if region in k
            ]

        return projected, coordinates

    def show_plane_intersection(
        self, scene: Scene, regions: List[Actor], root: Actor
    ):
        """
            Slices regions' meshjes with plane0 and adds the resulting intersection
            to the brainrender scene.
        """
        for region in regions + [root]:
            intersection = self.plane0.intersectWith(region._mesh)

            if len(intersection.points()):
                scene.add(intersection, transform=False)

            if region.name != "root":
                scene.remove(region)

    def slice_scene(self, scene: Scene, regions: List[Actor]):
        """
            Slices the meshes in a 3D brainrender scene using the gien planes
        """
        # slice the scene
        for n, plane in enumerate((self.plane0, self.plane1)):
            scene.slice(plane, actors=regions, close_actors=True)

        scene.slice(self.plane0, actors=scene.root, close_actors=False)
