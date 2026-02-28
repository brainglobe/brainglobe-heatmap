from typing import Dict, List

import numpy as np
import vedo as vd
import vtkmodules.all as vtk
from brainrender.actor import Actor

vtk.vtkLogger.SetStderrVerbosity(
    vtk.vtkLogger.VERBOSITY_OFF
)  # remove logger's prints during intersect_with_plane()


class Plane:
    def __init__(
        self, origin: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> None:
        self.center = origin
        self.u = u / np.linalg.norm(u)
        self.v = v / np.linalg.norm(v)
        assert np.isclose(np.dot(self.u, self.v), 0), (
            f"The plane vectors must be orthonormal to each "
            f"other (u ⋅ v = {np.dot(self.u, self.v)})"
        )
        self.normal = np.cross(self.u, self.v)
        self.M = np.vstack([u, v]).T

    @staticmethod
    def from_norm(origin: np.ndarray, norm: np.ndarray):
        u = np.zeros(3)
        m = np.where(norm != 0)[0][0]  # orientation can't be all-zeros
        n = (m + 1) % 3
        u[n] = norm[m]
        u[m] = -norm[n]
        norm = norm / np.linalg.norm(norm)
        u = u / np.linalg.norm(u)
        v = np.cross(norm, u)
        return Plane(origin, u, v)

    def to_mesh(self, actor: Actor):
        bounds = actor.bounds()
        length = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        )
        length += length / 3
        plane_mesh = Actor(
            vd.Plane(pos=self.center, normal=self.normal, s=(length, length)),
            name=f"PlaneMesh at {self.center} norm: {self.normal}",
            br_class="plane_mesh",
        )
        plane_mesh.width = length
        return plane_mesh

    def center_of_mass(self):
        return self.center

    def p3_to_p2(self, ps):
        # ps is a list of 3D points
        # returns a list of 2D points mapped on
        # the plane (u -> x axis, v -> y axis)
        return (ps - self.center) @ self.M

    def intersect_with(self, mesh: vd.Mesh):
        return mesh.intersect_with_plane(
            origin=self.center, normal=self.normal
        )

    def get_projections(self, actors: List[Actor]) -> Dict[str, np.ndarray]:
        """
        Slice each mesh actor with this plane and return 2D projections.

        NOTE: This method is used by the 3D rendering path only.
        The 2D path now uses atlas.annotation directly (see heatmaps.py),
        which completely avoids the mesh-slicing artefacts reported in #103.

        The mesh.cap() call (credit: @zenWai) closes any open meshes before
        intersection, improving robustness for the 3D path.
        """
        projected = {}
        for actor in actors:
            mesh: vd.Mesh = actor._mesh

            # Cap open meshes before intersection to reduce split artefacts
            # in 3D mode (fix suggested by @zenWai in issue #103)
            if not mesh.is_closed():
                mesh = mesh.cap()

            intersection = self.intersect_with(mesh)
            if not intersection.vertices.shape[0]:
                continue

            pieces = intersection.split()
            for piece_n, piece in enumerate(pieces):
                points = piece.join(reset=True).vertices
                projected[actor.name + f"_segment_{piece_n}"] = self.p3_to_p2(
                    points
                )
        return projected
