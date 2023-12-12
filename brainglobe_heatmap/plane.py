from typing import Dict, List

import numpy as np
import vedo as vd
import vtkmodules.all as vtk
from brainrender.actor import Actor
from vtkmodules.vtkFiltersCore import vtkPolyDataPlaneCutter

np.float = float  # for compatibility with old vedo
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)


# from vedo 2023.4.6
def intersect_with_plane(mesh: vd.Mesh, origin=(0, 0, 0), normal=(1, 0, 0)):
    """
    Intersect this Mesh with a plane to return a set of lines.

    Example:
        ```python
        from vedo import *
        sph = Sphere()
        mi = sph.clone().intersect_with_plane().join()
        print(mi.lines())
        show(sph, mi, axes=1).close()
        ```
        ![](https://vedo.embl.es/images/feats/intersect_plane.png)
    """
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    cutter = vtkPolyDataPlaneCutter()
    cutter.SetInputData(mesh.dataset)
    cutter.SetPlane(plane)
    cutter.InterpolateAttributesOn()
    cutter.ComputeNormalsOff()
    cutter.Update()

    msh = vd.Mesh(cutter.GetOutput(), "k", 1).lighting("off")
    msh.properties.SetLineWidth(3)
    msh.name = "PlaneIntersection"
    return msh


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
    def from_norm(origin: np.ndarray, norm: np.ndarray) -> vd.Plane:
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

    def centerOfMass(self):
        return self.center

    def P3toP2(self, ps):
        # ps is a list of 3D points
        # returns a list of 2D point mapped on
        # the plane (u -> x axis, v -> y axis)
        return (ps - self.center) @ self.M

    def intersectWith(self, mesh: vd.Mesh):
        # mesh.intersect_with_plane(
        # origin=self.center, normal=self.normal) in newer vedo
        return intersect_with_plane(
            mesh, origin=self.center, normal=self.normal
        )

    # for Slicer.get_structures_slice_coords()
    def get_projections(self, actors: List[Actor]) -> Dict[str, np.ndarray]:
        projected = {}
        for actor in actors:
            mesh: vd.Mesh = actor._mesh
            intersection = self.intersectWith(mesh)
            if not intersection.vertices.shape[0]:
                continue
            pieces = intersection.split()  # intersection.split() in newer vedo
            for piece_n, piece in enumerate(pieces):
                # sort coordinates
                points = piece.join(reset=True).vertices
                projected[actor.name + f"_segment_{piece_n}"] = self.P3toP2(
                    points
                )
        return projected
