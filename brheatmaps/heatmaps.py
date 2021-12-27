from vedo import Plane
from vedo.colors import colorMap as map_color
from typing import Optional, Tuple
import numpy as np

from brainrender import Scene
from brainrender.atlas import Atlas
from brainrender import settings

settings.SHOW_AXES = False
settings.SHADER_STYLE = "cartoon"


def check_values(values: dict, atlas: Atlas) -> Tuple[float, float]:
    """
        Checks that the passed heatmap values meet two criteria:
            - keys should be acronyms of brainregions
            - values should be numbers
    """
    for k, v in values.items():
        if not isinstance(v, (float, int)):
            raise ValueError(
                f'Heatmap values should be floats, not: {type(v)} for entry "{k}"'
            )

        if k not in atlas.lookup_df.acronym.values:
            raise ValueError(f'Region name "{k}" not recognized')

    vmax, vmin = max(values.values()), min(values.values())
    return vmax, vmin


def get_planes(
    scene: Scene,
    position: float = 0,
    orientation: str = "frontal",
    thickness: float = 100,
) -> Tuple[Plane, Plane]:
    """
        Returns the two planes used to slices the brainreder scene
    """
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

    plane0 = scene.atlas.get_plane(pos=p0, norm=tuple(norm0))
    plane1 = scene.atlas.get_plane(pos=p1, norm=tuple(norm1))

    return plane0, plane1


def prepare_heatmap(
    scene: Scene,
    plane0: Plane,
    plane1: Plane,
    values: dict,
    title: Optional[str] = None,
    cmap: str = "bwr",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Scene:
    # inspect values
    _vmax, _vmin = check_values(values, scene.atlas)
    vmin = vmin or _vmin
    vmax = vmax or _vmax

    # add brain regions to scene
    for region, value in values.items():
        color = list(map_color(value, name=cmap, vmin=vmin, vmax=vmax))
        scene.add_brain_region(region, color=color)

    # slice the scene
    regions = scene.get_actors(br_class="brain region")
    for plane in (plane0, plane1):
        scene.slice(plane, actors=regions, close_actors=True)
        scene.slice(plane, actors=scene.root, close_actors=False)
    return scene


def heatmap(
    values: dict,
    position: float = 0,
    orientation: str = "frontal",
    thickness: float = 100,
    title: Optional[str] = None,
    cmap: str = "bwr",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    interactive: bool = True,
    zoom: Optional[float] = None,
    **kwargs,
) -> Scene:
    """
        Create a heatmap showing the brain in coronal/frontal slices at a given position
    """
    # create a scene
    scene = Scene(title=title, root=False, **kwargs)

    # get the plane position
    plane0, plane1 = get_planes(
        scene, orientation=orientation, position=position, thickness=thickness
    )

    scene = prepare_heatmap(
        scene,
        plane0,
        plane1,
        values,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # render and return
    scene.render(camera=orientation, interactive=interactive, zoom=zoom)
    return scene
