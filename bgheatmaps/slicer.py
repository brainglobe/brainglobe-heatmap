from typing import List, Dict, Union, Optional
from vedo import Plane
import numpy as np

from brainrender.actor import Actor
from brainrender.scene import Scene


def get_ax_idx(orientation: str) -> int:
    """
        Given a named orientation get the idx
        of the axis orthogonal to the plane,
    """
    if orientation == "frontal":
        return 0
    elif orientation == "sagittal":
        return 2
    elif orientation == "horizontal":
        return 1
    else:
        raise ValueError(f'Orientation "{orientation}" not recognized')


class Slicer:
    def __init__(
        self,
        position: Optional[Union[list, tuple, np.ndarray, float]],
        orientation: Union[str, tuple],
        thickness: float,
        root: Actor,
    ):
        """
            Computes the position of two planes given a point (position) and an orientation (named orientation or 
            3D vector) + thickness (spacing between the two planes)
        """
        if position is None:
            position = root.centerOfMass()

        if isinstance(position, (float, int)):
            if isinstance(orientation, str):
                pval = position
                position = root.centerOfMass()
                position[get_ax_idx(orientation)] = pval
            else:
                raise ValueError(
                    "When a single float value is passed for position, the orientation should be one of the named orientations values"
                )

        position = np.array(position)
        position[2] = -position[2]

        if isinstance(orientation, str):
            axidx = get_ax_idx(orientation)

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
        length += length / 3

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

    def get_structures_slice_coords(self, regions: List[Actor], root: Actor):
        """
        It computes the intersection between the first slice plane and all 
        user given brain regions,
        returning the coordinates of each region as a set of XY (i.e. in the plane's
        coordinates system) coordinates
        """
        regions = regions + [root]

        pts = self.plane0.points() - self.plane0.points()[0]
        v = pts[1] / np.linalg.norm(pts[1])
        w = pts[2] / np.linalg.norm(pts[2])

        M = np.vstack([v, w]).T  # 3 x 2

        projected: Dict[str, np.ndarray] = {}
        for n, actor in enumerate(regions):
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

        for region in [r.name for r in regions]:
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


def get_structures_slice_coords(
    regions: List[str],
    position: Union[list, tuple, np.ndarray],
    orientation: Union[str, tuple] = "frontal",
    atlas_name: Optional[str] = None,
) -> Dict[str, List[np.ndarray]]:
    """
        Given a list of regions name and a set of plane parameters, it returns 
        the coordinates of the plane/regions' intersection in the plane's coordinates
    """

    scene = Scene(atlas_name=atlas_name)
    if len(regions) == 1:
        regions_actors = [scene.add_brain_region(*regions)]
    else:
        regions_actors = scene.add_brain_region(*regions)

    slicer = Slicer(position, orientation, 100, scene.root)

    structures_coords = slicer.get_structures_slice_coords(
        regions_actors, scene.root
    )[1]
    scene.close()
    return structures_coords
