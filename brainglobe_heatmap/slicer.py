from typing import Dict, List, Optional, Union

import numpy as np
from brainrender.actor import Actor
from brainrender.scene import Scene

from brainglobe_heatmap.plane import Plane

# Atlas-space (AP, DV, LR) -> 2D (x, y) projection for each plane,
# keyed by the index of the axis orthogonal to it (see get_ax_idx)
PROJ_M = {
    0: np.array([[0, 0], [0, 1], [1, 0]]),  # frontal: x=LR, y=DV
    1: np.array([[0, 1], [0, 0], [1, 0]]),  # horizontal: x=LR, y=AP
    2: np.array([[1, 0], [0, 1], [0, 0]]),  # sagittal: x=AP, y=DV
}


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
        Computes the position of two planes given a point (position) and an
        orientation (named orientation or
        3D vector) + thickness (spacing between the two planes)
        """
        if position is None:
            _position = root.center_of_mass()

        if isinstance(position, (float, int, np.number)):
            if isinstance(orientation, str):
                _position = root.center_of_mass()
                _position[get_ax_idx(orientation)] = position
            else:
                raise ValueError(
                    "When a single float value is passed for "
                    "position, the orientation "
                    "should be one of the named orientations values"
                )
        else:
            _position = np.array(position)

        _position[2] = -_position[2]

        if isinstance(orientation, str):
            axidx = get_ax_idx(orientation)

            # get the two points the planes are centered at
            shift = np.zeros(3)
            shift[axidx] -= thickness
            p1 = _position - shift

            # get the two planes
            # assures that u0×v0 is all-positive -> it's for plane0
            if orientation == "frontal":
                u0, v0 = np.array([[0, 0, -1], [0, 1, 0]])
            elif orientation == "sagittal":
                u0, v0 = np.array([[1, 0, 0], [0, 1, 0]])
            else:  # orientation == "horizontal"
                u0, v0 = np.array([[0, 0, 1], [1, 0, 0]])
            plane0 = Plane(_position, u0, v0)
            u1, v1 = u0.copy(), -v0.copy()  # set u1:=u0 and v1:=-v0
            plane1 = Plane(p1, u1, v1)

            self._proj_M = PROJ_M[axidx]
        else:
            orientation = np.array(orientation)

            p1 = _position + orientation * thickness  # type: ignore

            norm0 = orientation  # type: ignore
            plane0 = Plane.from_norm(_position, norm0)
            norm1 = -orientation  # type: ignore
            plane1 = Plane.from_norm(p1, norm1)

            dominant = int(np.argmax(np.abs(orientation)))
            self._proj_M = PROJ_M[dominant]

        self.plane0 = Actor(
            plane0,
            name=f"Plane at {plane0.center} norm: {plane0.normal}",
            br_class="plane",
        )
        self.plane1 = Actor(
            plane1,
            name=f"Plane at {plane1.center} norm: {plane1.normal}",
            br_class="plane",
        )

    def _project_to_2d(self, points_br: np.ndarray) -> np.ndarray:
        """Project 3D brainrender points to 2D atlas-space coordinates."""
        pts = points_br.copy()
        pts[:, 2] = -pts[:, 2]  # undo brainrender Z-flip
        return pts @ self._proj_M

    def get_structures_slice_coords(self, regions: List[Actor], root: Actor):
        """
        It computes the intersection between the first slice plane and all
        user given brain regions,
        returning the coordinates of each region as a
        set of XY (i.e., in the plane's
        coordinates system) coordinates
        """
        regions = regions + [root]

        projected = {
            key: self._project_to_2d(points)
            for key, points in self.plane0.get_intersections(regions).items()
        }

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
        Slices regions' meshes with plane0 and adds the resulting intersection
        to the brainrender scene.
        """
        for region in regions + [root]:
            intersection = self.plane0.intersect_with(region._mesh)

            if len(intersection.vertices):
                scene.add(intersection, transform=False)

            if region.name != "root":
                scene.remove(region)

    def slice_scene(self, scene: Scene, regions: List[Actor]):
        """
        Slices the meshes in a 3D brainrender scene using the given planes
        """
        # slice the scene
        for _, plane in enumerate((self.plane0, self.plane1)):
            scene.slice(plane, actors=regions, close_actors=True)

        scene.slice(self.plane0, actors=scene.root, close_actors=False)


def get_structures_slice_coords(
    regions: List[str],
    position: Union[list, tuple, np.ndarray],
    orientation: Union[str, tuple] = "frontal",
    atlas_name: Optional[str] = None,
    check_latest: bool = True,
) -> Dict[str, List[np.ndarray]]:
    """
    Given a list of region name and a set of plane parameters,
    it returns the coordinates of the plane/regions'
    intersection in atlas space (µm):

    - frontal-like planes: (LR, DV)
    - horizontal-like planes: (LR, AP)
    - sagittal-like planes: (AP, DV)

    Tuple orientations use the axes of the named plane closest to them
    (largest normal component). Each point keeps its own atlas coordinates,
    i.e. oblique slices are viewed along that axis, so distances along the
    tilted direction are shortened by cos(tilt).
    """

    scene = Scene(atlas_name=atlas_name, check_latest=check_latest)
    if len(regions) == 1:
        regions_actors = [scene.add_brain_region(*regions)]
    else:
        regions_actors = scene.add_brain_region(*regions)

    slicer = Slicer(position, orientation, 100, scene.root)

    structures_coords = slicer.get_structures_slice_coords(
        regions_actors, scene.root
    )[1]
    return structures_coords
