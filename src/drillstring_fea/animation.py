"""Animations of the deformed structure with PyVista (MP4 video or interactive window).

Install the optional dependencies with ``pip install -e ".[animation]"``.

Drawing
-------
Each element is drawn as a cylinder with ``n_panels`` quadrilateral panels.
The two end rings of the cylinder belong to the two element nodes and move
with them: a ring point P of node n goes to

    P' = X_n + u_n + R(theta_n) (P - X_n)

where X_n is the initial node position, u_n its displacement and R(theta_n)
the Rodrigues matrix (Eq. 5) of the node rotations stored in U. By default
(``rings="tangent"``) the rings are instead kept perpendicular to the drawn
axis, so the cylinders stay round when the displacements are magnified.
The well is a fixed cylinder, the floor an annulus at the top node and the
drill bit a surface mesh attached to the bottom node.

Drill strings are thousands of meters long and move a few centimeters
sideways, so displacements and diameters can be magnified (see ``Scene``).

Drill-bit mesh (``data/simple_drill_bit.txt``, from the visualization scripts
of the dissertation): 365 points and 781 triangles, coordinates relative to
the bit diameter (multiplied by it when drawn). Format: a header line
``NNodes NEle``, one line ``X Y Z`` per point, then one line per triangle with
its three 0-based point numbers.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np

from .beam import rodrigues
from .model import NDOF
from .results import Result

DRILL_BIT_MESH = files("drillstring_fea") / "data" / "simple_drill_bit.txt"
COLOR_OPTIONS = ("N", "Vy", "Vz", "T", "My", "Mz", "lateral")


def _pyvista():
    try:
        import pyvista
    except ImportError:  # pragma: no cover - depends on the installation
        raise ImportError('animations need PyVista: pip install -e ".[animation]"') from None
    return pyvista


# --------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------
def ring_axes(x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to each element, used to draw its rings."""
    axis = (x2 - x1) / np.linalg.norm(x2 - x1, axis=1)[:, None]
    a = np.cross([0.0, 0.0, 1.0], axis)  # horizontal vector normal to the element
    vertical = np.linalg.norm(a, axis=1) < 1e-10
    a[vertical] = [0.0, 1.0, 0.0]
    a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(axis, a)
    return a, b


def cylinders(coords, connect, radii, n_panels: int):
    """Cylinder surface of every element.

    Returns ``points`` (n_elements * 2 * n_panels, 3), ``faces`` (PyVista format)
    and ``owner`` (node that moves each point).
    """
    n1, n2 = connect[:, 0], connect[:, 1]
    a, b = ring_axes(coords[n1], coords[n2])
    angles = 2 * np.pi * np.arange(n_panels) / n_panels
    # ring offsets: r (cos(t) a + sin(t) b), shape (n_elements, n_panels, 3)
    offset = radii[:, None, None] * (np.cos(angles)[None, :, None] * a[:, None, :]
                                     + np.sin(angles)[None, :, None] * b[:, None, :])
    ring1 = coords[n1][:, None, :] + offset
    ring2 = coords[n2][:, None, :] + offset
    points = np.stack([ring1, ring2], axis=2).reshape(-1, 3)  # ring1/ring2 alternate
    owner = np.stack([np.repeat(n1[:, None], n_panels, 1), np.repeat(n2[:, None], n_panels, 1)],
                     axis=2).ravel()

    # quad k of element e joins points (k, node1) (k+1, node1) (k+1, node2) (k, node2)
    base = 2 * n_panels * np.arange(len(connect))[:, None]
    k = np.arange(n_panels)[None, :]
    k1 = (k + 1) % n_panels
    quads = np.stack([base + 2 * k, base + 2 * k1, base + 2 * k1 + 1, base + 2 * k + 1], axis=2)
    faces = np.hstack([np.full((quads.size // 4, 1), 4), quads.reshape(-1, 4)]).ravel()
    return points, faces, owner


def read_drill_bit(path=DRILL_BIT_MESH):
    """Triangulated drill bit: points relative to the bit (scaled by its diameter) and faces."""
    numbers = []
    text = Path(path).read_text() if isinstance(path, str) else path.read_text()
    for line in text.splitlines():
        try:
            values = [float(v) for v in line.split()]
        except ValueError:  # header lines
            continue
        if values:  # skip blank lines
            numbers.append(values)
    n_points, n_triangles = (int(v) for v in numbers[0])
    points = np.array(numbers[1:1 + n_points])
    triangles = np.array(numbers[1 + n_points:1 + n_points + n_triangles], dtype=int)
    faces = np.hstack([np.full((n_triangles, 1), 3), triangles]).ravel()
    return points, faces


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------
class Scene:
    """Surfaces of one result and their positions at each saved step.

    Magnification factors (the original ParaView animations of the drill
    strings used 250 / 20 / 125):

    * ``displacement_scale`` multiplies the lateral (x, y) displacements;
    * ``axial_scale`` multiplies the axial (z) displacements (default 1: real scale);
    * ``pipe_scale`` multiplies the pipe (and drill bit) diameters;
    * ``well_scale`` multiplies the well diameter.

    Ring orientation (``rings``):

    * ``"tangent"`` (default): rings kept perpendicular to the drawn (magnified)
      axis of the pipe, so the cylinders stay round when displacements are magnified;
    * ``"rotations"``: rings rotated by the node rotations stored in U,
      P' = C + R(theta) P (exact without magnification).
    """

    def __init__(self, result: Result, displacement_scale: float = 1.0,
                 axial_scale: float = 1.0, pipe_scale: float = 1.0,
                 well_scale: float = 1.0, rings: str = "tangent", n_panels: int = 16,
                 z_range: tuple[float, float] | None = None, color_by: str | None = None):
        if color_by is not None and color_by not in COLOR_OPTIONS:
            raise ValueError(f"color_by must be one of {COLOR_OPTIONS}")
        if rings not in ("rotations", "tangent"):
            raise ValueError("rings must be 'rotations' or 'tangent'")
        self.result = result
        self.color_by = color_by
        self.rings = rings
        self.n_panels = n_panels
        self.magnify = np.array([displacement_scale, displacement_scale, axial_scale])
        x0 = result.coords

        # elements to draw (all, or those inside z_range)
        connect = result.connect
        if z_range is not None:
            z = x0[:, 2]
            inside = (z >= min(z_range)) & (z <= max(z_range))
            connect = connect[inside[connect[:, 0]] & inside[connect[:, 1]]]
        self.connect = connect

        diam = np.asarray(result.pipe_diam if result.pipe_diam is not None else 0.0, float)
        diam = np.broadcast_to(diam, (len(result.connect),))
        size = np.ptp(x0, axis=0).max()
        diam = np.where(diam > 0, diam, 0.01 * size)  # cases without a pipe diameter
        elem_index = {tuple(c): i for i, c in enumerate(result.connect)}
        self.radii = pipe_scale * np.array([diam[elem_index[tuple(c)]] for c in connect]) / 2

        # cylinders in the initial configuration (drawn radius)
        self.pipe_points0, self.pipe_faces, self.pipe_owner = cylinders(x0, connect, self.radii,
                                                                        n_panels)
        self.base_axis = ring_axes(x0[connect[:, 0]], x0[connect[:, 1]])[0]

        self.bottom = int(connect.max())
        has_well = np.isfinite(result.well_diam)
        self.bit_points0 = self.bit_faces = None
        if has_well and self.bottom == len(x0) - 1 and DRILL_BIT_MESH.is_file():
            local, self.bit_faces = read_drill_bit()
            self.bit_points0 = x0[self.bottom] + local * diam[-1] * pipe_scale

        self.well = self.floor = None
        if has_well:
            pv = _pyvista()
            top = int(connect.min())
            z_top, z_bottom = x0[top, 2], x0[self.bottom, 2]
            r_well = result.well_diam / 2 * well_scale
            self.well = pv.Cylinder(center=(0, 0, (z_top + z_bottom) / 2), direction=(0, 0, 1),
                                    radius=r_well, height=z_top - z_bottom, resolution=48,
                                    capping=False)
            if top == 0:
                self.floor = pv.Disc(center=(0, 0, z_top), inner=r_well, outer=4 * r_well,
                                     c_res=48)

    def positions(self, i: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Drawn positions of the pipe and bit points at saved step index ``i``."""
        r = self.result
        u = r.U[i].reshape(-1, NDOF)
        center = r.coords + u[:, :3] * self.magnify  # drawn node positions
        rotation = rodrigues(u[:, 3:])  # R(theta_n) of every node, Eq. 5

        if self.rings == "rotations":
            # P' = C_n + R(theta_n) (P - X_n): each ring turns with its node
            owner = self.pipe_owner
            offset = self.pipe_points0 - r.coords[owner]
            pipe = center[owner] + np.einsum("pij,pj->pi", rotation[owner], offset)
        else:
            pipe = self._tangent_rings(center)

        bit = None
        if self.bit_points0 is not None:
            offset = self.bit_points0 - r.coords[self.bottom]
            bit = center[self.bottom] + offset @ rotation[self.bottom].T
        return pipe, bit

    def _tangent_rings(self, center: np.ndarray) -> np.ndarray:
        """Rings perpendicular to the drawn axis (mean direction of the adjacent elements)."""
        n1, n2 = self.connect[:, 0], self.connect[:, 1]
        direction = center[n2] - center[n1]
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        tangent = np.zeros_like(center)
        np.add.at(tangent, n1, direction)
        np.add.at(tangent, n2, direction)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-30)[:, None]

        angles = 2 * np.pi * np.arange(self.n_panels) / self.n_panels
        rings = []
        for node in (n1, n2):
            t = tangent[node]
            # initial ring axis projected on the plane normal to the tangent
            a = self.base_axis - np.sum(self.base_axis * t, axis=1)[:, None] * t
            a /= np.linalg.norm(a, axis=1)[:, None]
            b = np.cross(t, a)
            offset = self.radii[:, None, None] * (np.cos(angles)[None, :, None] * a[:, None, :]
                                                  + np.sin(angles)[None, :, None] * b[:, None, :])
            rings.append(center[node][:, None, :] + offset)
        return np.stack(rings, axis=2).reshape(-1, 3)

    def scalars(self, i: int) -> np.ndarray | None:
        """Value used to color each pipe point (section force or lateral displacement)."""
        if self.color_by is None:
            return None
        if self.color_by == "lateral":
            u = self.result.U[i].reshape(-1, NDOF)
            node_values = np.hypot(u[:, 0], u[:, 1])
        else:
            column = COLOR_OPTIONS.index(self.color_by)
            node_values = self.result.section_forces(int(self.result.steps[i]))[:, column]
        return node_values[self.pipe_owner]


# --------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------
def _frames(result: Result, start: int | None, end: int | None, every: int) -> np.ndarray:
    steps = result.steps
    keep = np.ones(len(steps), bool)
    if start is not None:
        keep &= steps >= start
    if end is not None:
        keep &= steps <= end
    return np.flatnonzero(keep)[::every]


def _plotter(scene: Scene, i: int, off_screen: bool, window_size, clim):
    pv = _pyvista()
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    pipe_points, bit_points = scene.positions(i)
    pipe = pv.PolyData(pipe_points, scene.pipe_faces)
    scalars = scene.scalars(i)
    if scalars is not None:
        pipe.point_data[scene.color_by] = scalars
        plotter.add_mesh(pipe, scalars=scene.color_by, clim=clim, cmap="coolwarm",
                         smooth_shading=True)
    else:
        plotter.add_mesh(pipe, color="lightsteelblue", smooth_shading=True)
    bit = None
    if bit_points is not None:
        bit = pv.PolyData(bit_points, scene.bit_faces)
        plotter.add_mesh(bit, color="goldenrod", smooth_shading=True)
    if scene.well is not None:
        plotter.add_mesh(scene.well, color="sienna", opacity=0.25)
    if scene.floor is not None:
        plotter.add_mesh(scene.floor, color="gray", opacity=0.6)
    plotter.add_axes()
    return plotter, pipe, bit


def _label(result: Result, i: int) -> str:
    return f"step {result.steps[i]}   t = {result.time[i]:.2f} s"


def animate(result: Result, path: str | Path, *, start: int | None = None,
            end: int | None = None, every: int = 1, fps: int = 25,
            window_size=(1280, 720), **scene_options) -> Path:
    """Write an MP4 video of the saved steps ``start``..``end`` (every ``every``-th).

    ``scene_options`` are the arguments of :class:`Scene` (``displacement_scale``,
    ``axial_scale``, ``pipe_scale``, ``well_scale``, ``rings``, ``color_by``,
    ``z_range``, ``n_panels``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = Scene(result, **scene_options)
    color_by = scene.color_by
    frames = _frames(result, start, end, every)
    if frames.size == 0:
        raise ValueError("no saved step in the requested range")
    clim = None
    if color_by is not None:  # same color scale in every frame
        values = np.concatenate([scene.scalars(i) for i in frames])
        clim = (values.min(), values.max())

    plotter, pipe, bit = _plotter(scene, frames[0], True, window_size, clim)
    plotter.open_movie(str(path), framerate=fps)
    for i in frames:
        pipe_points, bit_points = scene.positions(i)
        pipe.points = pipe_points
        if color_by is not None:
            pipe.point_data[color_by] = scene.scalars(i)
        if bit is not None:
            bit.points = bit_points
        plotter.add_text(_label(result, i), name="label", font_size=10)
        plotter.write_frame()
    plotter.close()
    return path


def show(result: Result, step: int | None = None, *, screenshot: str | Path | None = None,
         **scene_options):
    """Interactive window (or a PNG with ``screenshot``) of one saved step (default: last)."""
    scene = Scene(result, **scene_options)
    i = len(result.steps) - 1 if step is None else result.index(step)
    plotter, _, _ = _plotter(scene, i, screenshot is not None, (1280, 720), None)
    plotter.add_text(_label(result, i), font_size=10)
    if screenshot is not None:
        plotter.screenshot(str(screenshot))
        plotter.close()
        return screenshot
    plotter.show()
    return None
