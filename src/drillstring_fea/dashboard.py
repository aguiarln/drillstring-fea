"""Dashboard animation of a drill string: 3D view, top view, profiles and time histories.

Layout (1920 x 1080):

    +--------------------------------------------------------------------------+
    | title                                    KPI cards (time, WOB, rpm, ...) |
    +----------------+----------------------+-------------+--------------------+
    |                | top view (looking    | lateral     | axial force        |
    |    3D view     | down the well,       | displacement| along the depth    |
    |  (PyVista)     | PyVista)             | vs depth    |                    |
    |                +----------------------+-------------+--------------------+
    |                | time histories: weight on bit and rotation speeds       |
    +----------------+---------------------------------------------------------+

The 3D view and the top view are rendered off screen by PyVista from the same
pipe panels (``animation.Scene``, same magnification and colors) and inserted
as images; the other panels are Matplotlib axes. Requires the
optional dependencies ``pip install -e ".[animation,plot]"``.

Sign convention of the plots: axial force positive in tension; weight on bit
(WOB) = compression at the bottom node.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .animation import Scene, _frames, _pyvista
from .model import DX, DY, NDOF, RZ
from .results import Result

THEME = {
    "background": "#0b1220",
    "panel": "#131c2e",
    "border": "#24324a",
    "text": "#e6edf7",
    "muted": "#8b9bb4",
    "grid": "#22304a",
    "accent": "#38bdf8",  # cyan
    "accent2": "#f59e0b",  # amber
    "accent3": "#a78bfa",  # violet
    "alert": "#f43f5e",  # red
    "pipe": "#cbd5e1",
    "well": "#64748b",
    "bit": "#eab308",
}

QUANTITIES = {  # color_by -> (label, unit, scale, colormap, symmetric range)
    "lateral": ("Lateral displacement", "mm", 1e3, "plasma", False),
    "N": ("Axial force (tension +)", "kN", -1e-3, "coolwarm", True),
    "T": ("Torque", "kN·m", 1e-3, "coolwarm", True),
    "My": ("Bending moment My", "kN·m", 1e-3, "coolwarm", True),
    "Mz": ("Bending moment Mz", "kN·m", 1e-3, "coolwarm", True),
}
FORCE_COLUMN = {"N": 0, "T": 3, "My": 4, "Mz": 5}
TRAIL_LENGTH = 60  # saved steps shown in the path of the node in the top view


def _style(ax, title: str) -> None:
    ax.set_facecolor(THEME["panel"])
    for side in ax.spines.values():
        side.set_color(THEME["border"])
    ax.tick_params(colors=THEME["muted"], labelsize=9)
    ax.xaxis.label.set_color(THEME["muted"])
    ax.yaxis.label.set_color(THEME["muted"])
    ax.grid(color=THEME["grid"], linewidth=0.6)
    ax.set_title(title, color=THEME["text"], fontsize=12, loc="left", pad=8, fontweight="bold")


class Dashboard:
    """Figure with all panels; ``draw(i)`` updates it to saved step index ``i``.

    ``frames``: saved-step indices of the video (fix the color scale and the
    axis limits). ``color_by``: quantity of the colors (see ``QUANTITIES``).
    ``node``: node whose path is drawn in the top view (default: largest
    lateral motion). ``z_range``: part of the string to draw. The scales
    magnify the lateral displacements and the pipe and well diameters, as in
    ``animation.Scene``. ``size`` and ``dpi`` give the picture size in pixels.
    """

    def __init__(self, result: Result, frames: np.ndarray, *, color_by: str = "lateral",
                 node: int | None = None, z_range: tuple[float, float] | None = None,
                 displacement_scale: float = 250.0, pipe_scale: float = 20.0,
                 well_scale: float = 125.0, title: str | None = None,
                 size=(1920, 1080), dpi: int = 100):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors

        if not np.isfinite(result.well_diam):
            raise ValueError("the dashboard needs a model with a well (drill strings)")
        if color_by not in QUANTITIES:
            raise ValueError(f"color_by must be one of {list(QUANTITIES)}")
        self.result, self.frames, self.color_by = result, frames, color_by
        r = result
        z = r.coords[:, 2]
        self.depth = -z

        # ---- nodes drawn and data of every saved step (precomputed once) ----
        self.scene = Scene(r, displacement_scale=displacement_scale, pipe_scale=pipe_scale,
                           well_scale=well_scale, z_range=z_range)
        self.nodes = np.unique(self.scene.connect)
        u = r.U.reshape(len(r.steps), -1, NDOF)
        self.lateral = np.hypot(u[:, :, DX], u[:, :, DY])  # (steps, nodes), m
        self.ux, self.uy = u[:, :, DX], u[:, :, DY]
        gap_elem = (r.well_diam - r.pipe_diam) / 2
        self.gap = np.append(gap_elem, gap_elem[-1])  # radial clearance per node (m)
        forces = np.stack([r.section_forces(int(s)) for s in r.steps])  # (steps, nodes, 6)
        self.tension = -forces[:, :, 0]  # tension positive
        self.wob = forces[:, -1, 0]  # compression at the bit
        t = r.time
        dynamic = t > 0
        rpm = np.zeros((len(t), 2))
        if dynamic.sum() > 2:
            for k, n in enumerate((0, r.n_nodes - 1)):  # top and bit
                theta = u[:, n, RZ]
                rpm[dynamic, k] = -np.gradient(theta[dynamic], t[dynamic]) * 60 / (2 * np.pi)
        self.rpm = rpm
        if node is None:  # node with the largest lateral displacement in the video
            node = int(self.nodes[np.argmax(self.lateral[np.ix_(frames, self.nodes)].max(0))])
        self.node = node

        # color scale of the 3D view, fixed for the whole video
        label, unit, factor, cmap, symmetric = QUANTITIES[color_by]
        values = np.concatenate([self._node_values(i)[self.nodes] for i in frames])
        lo, hi = values.min(), values.max()
        if symmetric:
            lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
        self.norm = colors.Normalize(lo, hi if hi > lo else lo + 1)
        self.cmap = matplotlib.colormaps[cmap]

        # ---- figure and panels ----
        w, h = size
        self.fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=THEME["background"])
        fig = self.fig
        ax3d = fig.add_axes([0.012, 0.03, 0.30, 0.83])
        ax_top = fig.add_axes([0.355, 0.40, 0.25, 0.46])
        ax_lat = fig.add_axes([0.655, 0.40, 0.155, 0.46])
        ax_axial = fig.add_axes([0.835, 0.40, 0.155, 0.46], sharey=ax_lat)
        ax_time = fig.add_axes([0.355, 0.07, 0.60, 0.23])
        self._header(title or r.model_name.replace("_", " ").title())

        # 3D view: PyVista image + color bar
        ax3d.set_facecolor(THEME["panel"])
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        for side in ax3d.spines.values():
            side.set_color(THEME["border"])
        drawn = self.depth[self.nodes]
        ax3d.set_title(f"3D view — {drawn.min():.0f} to {drawn.max():.0f} m", color=THEME["text"],
                       fontsize=12, loc="left", pad=8, fontweight="bold")
        self.plotter = self._init_3d(*self._pixels(ax3d))
        self.image = self._image_axes(ax3d)
        ax3d.text(0.02, 0.015, f"magnification: displacements x{displacement_scale:g}, "
                  f"pipe x{pipe_scale:g}, well x{well_scale:g}", transform=ax3d.transAxes,
                  color=THEME["muted"], fontsize=8)
        cax = fig.add_axes([0.285, 0.10, 0.008, 0.30])
        bar = fig.colorbar(cm.ScalarMappable(self.norm, self.cmap), cax=cax)
        bar.set_label(f"{label} ({unit})", color=THEME["muted"], fontsize=9)
        bar.ax.tick_params(colors=THEME["muted"], labelsize=8)
        bar.outline.set_edgecolor(THEME["border"])

        # Top view: the same panels seen from above (looking down the well)
        ax_top.set_facecolor(THEME["panel"])
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for side in ax_top.spines.values():
            side.set_color(THEME["border"])
        ax_top.set_title("Top view — looking down the well", color=THEME["text"], fontsize=12,
                         loc="left", pad=8, fontweight="bold")
        self.top_plotter = self._init_top(*self._pixels(ax_top))
        self.top_image = self._image_axes(ax_top)
        ax_top.text(0.02, 0.015, f"path of node {node} ({self.depth[node]:.0f} m)",
                    transform=ax_top.transAxes, color=THEME["accent2"], fontsize=8)
        ax_top.text(0.98, 0.015, "same magnification as the 3D view", ha="right",
                    transform=ax_top.transAxes, color=THEME["muted"], fontsize=8)
        for tip in ((0.97, 0.92), (0.89, 1.0)):  # x and y axes of the view
            ax_top.annotate("", xy=tip, xytext=(0.89, 0.92), xycoords="axes fraction",
                            arrowprops={"arrowstyle": "->", "color": THEME["muted"]})
        ax_top.text(0.975, 0.92, "x", transform=ax_top.transAxes, color=THEME["muted"],
                    fontsize=9, ha="left", va="center")
        ax_top.text(0.865, 0.985, "y", transform=ax_top.transAxes, color=THEME["muted"],
                    fontsize=9, ha="right", va="top")

        # Profiles along the depth
        d = self.depth[self.nodes]
        _style(ax_lat, "Lateral displacement")
        ax_lat.plot(self.gap[self.nodes] * 1e3, d, "--", color=THEME["muted"], lw=1,
                    label="radial clearance")
        (self.lat_line,) = ax_lat.plot([], [], color=THEME["accent"], lw=1.6, label="|u| lateral")
        self.contact_dots = ax_lat.scatter([], [], s=10, color=THEME["alert"], zorder=5,
                                           label="contact")
        ax_lat.set_xlim(0, 1.1 * max(self.gap[self.nodes].max(),
                                     self.lateral[np.ix_(frames, self.nodes)].max()) * 1e3)
        ax_lat.set_ylim(d.max(), d.min())
        ax_lat.set_xlabel("mm")
        ax_lat.set_ylabel("depth (m)")
        ax_lat.legend(loc="upper right", fontsize=7, facecolor=THEME["panel"],
                      edgecolor=THEME["border"], labelcolor=THEME["muted"])

        _style(ax_axial, "Axial force")
        ax_axial.axvline(0, color=THEME["muted"], lw=0.8)
        (self.axial_line,) = ax_axial.plot([], [], color=THEME["accent2"], lw=1.6)
        self.np_marker = ax_axial.axhline(np.nan, color=THEME["alert"], lw=1, ls=":")
        self.np_text = ax_axial.text(0.04, 0.96, "", transform=ax_axial.transAxes,
                                     color=THEME["alert"], fontsize=8, va="top")
        tension = self.tension[np.ix_(frames, self.nodes)] / 1e3
        ax_axial.set_xlim(1.05 * min(tension.min(), 0), 1.05 * max(tension.max(), 0) + 1e-9)
        ax_axial.set_xlabel("kN (tension +)")
        ax_axial.tick_params(labelleft=False)

        # Time histories
        _style(ax_time, "Weight on bit and rotation speed")
        tt = r.time
        ax_time.plot(tt, self.wob / 1e3, color=THEME["accent2"], lw=1.6, label="WOB (kN)")
        ax_time.set_xlabel("time (s)")
        ax_time.set_ylabel("WOB (kN)", color=THEME["accent2"])
        ax_rpm = ax_time.twinx()
        ax_rpm.plot(tt, rpm[:, 0], color=THEME["muted"], lw=1, ls="--", label="top (rpm)")
        ax_rpm.plot(tt, rpm[:, 1], color=THEME["accent"], lw=1.4, label="bit (rpm)")
        ax_rpm.tick_params(colors=THEME["muted"], labelsize=9)
        ax_rpm.set_ylabel("rotation speed (rpm)", color=THEME["accent"])
        for side in ax_rpm.spines.values():
            side.set_color(THEME["border"])
        ax_time.set_xlim(tt[frames[0]], tt[frames[-1]] + 1e-9)
        self.cursor = ax_time.axvline(tt[frames[0]], color=THEME["text"], lw=1)
        lines = ax_time.get_lines()[:1] + ax_rpm.get_lines()
        ax_time.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8,
                       facecolor=THEME["panel"], edgecolor=THEME["border"],
                       labelcolor=THEME["muted"])

    # ------------------------------------------------------------------
    def _header(self, title: str) -> None:
        fig = self.fig
        heading = fig.text(0.012, 0.955, title, color=THEME["text"], fontsize=20,
                           fontweight="bold", va="center")
        # shrink long titles so they do not run into the indicator cards (x = 0.40)
        renderer = fig.canvas.get_renderer()
        while (heading.get_window_extent(renderer).x1 > 0.39 * fig.bbox.width
               and heading.get_fontsize() > 9):
            heading.set_fontsize(heading.get_fontsize() - 1)
        fig.text(0.012, 0.918, "Nonlinear dynamic analysis of a drill string — drillstring-fea",
                 color=THEME["muted"], fontsize=10, va="center")
        names = ["Time", "Step", "WOB", "Bit speed", "Max lateral", "Nodes in contact"]
        self.kpi = {}
        x0, width = 0.40, 0.098
        for k, name in enumerate(names):
            x = x0 + k * (width + 0.003)
            box = fig.add_axes([x, 0.895, width, 0.085])
            box.set_xticks([])
            box.set_yticks([])
            box.set_facecolor(THEME["panel"])
            for side in box.spines.values():
                side.set_color(THEME["border"])
            box.text(0.08, 0.72, name.upper(), color=THEME["muted"], fontsize=8,
                     transform=box.transAxes)
            self.kpi[name] = box.text(0.08, 0.25, "", color=THEME["text"], fontsize=16,
                                      fontweight="bold", transform=box.transAxes)

    @staticmethod
    def _pixels(ax) -> tuple[int, int]:
        bbox = ax.get_window_extent()
        return int(bbox.width), int(bbox.height)

    @staticmethod
    def _image_axes(ax):
        width, height = Dashboard._pixels(ax)
        return ax.imshow(np.zeros((height, width, 3), np.uint8), aspect="auto")

    def _plotter(self, width: int, height: int, edges: bool):
        """Off-screen plotter with the pipe panels, the bit and the well outline."""
        p = _pyvista().Plotter(off_screen=True, window_size=(width, height))
        p.set_background(THEME["panel"])
        p.add_mesh(self.pipe, scalars="value", cmap=self.cmap, clim=(self.norm.vmin,
                   self.norm.vmax), smooth_shading=True, specular=0.4, show_scalar_bar=False,
                   show_edges=edges, edge_color=THEME["background"], line_width=0.5)
        if self.bit is not None:
            p.add_mesh(self.bit, color=THEME["bit"], smooth_shading=True, specular=0.6)
        if self.scene.well is not None:
            p.add_mesh(self.scene.well.extract_feature_edges(), color=THEME["well"], opacity=0.5)
        p.enable_anti_aliasing("ssaa")
        p.enable_parallel_projection()
        return p

    def _init_3d(self, width: int, height: int):
        """3D view: oblique camera, translucent well, floor and depth labels."""
        pv = _pyvista()
        s = self.scene
        # meshes shared by the 3D view and the top view (updated in draw)
        pipe_points, bit_points = s.positions(self.frames[0])
        self.pipe = pv.PolyData(pipe_points, s.pipe_faces)
        self.pipe.point_data["value"] = self._point_values(self.frames[0])
        self.bit = None if bit_points is None else pv.PolyData(bit_points, s.bit_faces)

        p = self._plotter(width, height, edges=False)
        if s.well is not None:
            p.add_mesh(s.well, color=THEME["well"], opacity=0.12, smooth_shading=True)
        if s.floor is not None:
            p.add_mesh(s.floor, color=THEME["well"], opacity=0.35)
        # depth labels at the top and bottom of the drawn part of the string
        x0 = self.result.coords
        top, bottom = int(s.connect.min()), int(s.connect.max())
        r_label = 1.6 * (s.well.bounds[1] if s.well is not None else 1.0)
        labels = np.array([[r_label, 0, x0[top, 2]], [r_label, 0, x0[bottom, 2]]])
        p.add_point_labels(labels, [f"{-x0[top, 2]:.0f} m", f"{-x0[bottom, 2]:.0f} m"],
                           text_color=THEME["muted"], font_size=14, shape=None,
                           show_points=False, always_visible=True)
        p.view_xz()
        p.camera.azimuth = 35
        p.camera.elevation = 12
        p.reset_camera()
        p.camera.zoom(1.3)
        return p

    def _init_top(self, width: int, height: int):
        """Top view: the same panels, with their edges, seen along -z, and the path of
        the selected node drawn above the string."""
        pv = _pyvista()
        s = self.scene
        p = self._plotter(width, height, edges=True)
        self.trail_z = self.result.coords[int(s.connect.min()), 2] + 1.0  # above the string
        points = np.tile(self._node_center(self.frames[0]), (TRAIL_LENGTH, 1))
        self.trail = pv.PolyData(points, lines=np.r_[TRAIL_LENGTH, np.arange(TRAIL_LENGTH)])
        p.add_mesh(self.trail, color=THEME["accent2"], line_width=3)
        self.marker = pv.PolyData(points[:1].copy())
        p.add_mesh(self.marker, color=THEME["accent2"], point_size=12,
                   render_points_as_spheres=True)
        p.view_xy()  # camera on +z looking down: x to the right, y up
        p.reset_camera()
        bounds = s.well.bounds if s.well is not None else self.pipe.bounds
        p.camera.parallel_scale = 1.08 * max(abs(v) for v in bounds[:4])
        return p

    def _node_center(self, i: int) -> np.ndarray:
        """Drawn (magnified) position of the selected node, lifted above the string."""
        n = self.node
        mx, my = self.scene.magnify[:2]
        x0 = self.result.coords[n]
        return np.array([x0[0] + mx * self.ux[i, n], x0[1] + my * self.uy[i, n], self.trail_z])

    def _node_values(self, i: int) -> np.ndarray:
        """Quantity used for the colors, at every node, in display units."""
        label, unit, factor, cmap, symmetric = QUANTITIES[self.color_by]
        if self.color_by == "lateral":
            return self.lateral[i] * factor
        forces = self.result.section_forces(int(self.result.steps[i]))
        return forces[:, FORCE_COLUMN[self.color_by]] * factor

    def _point_values(self, i: int) -> np.ndarray:
        return self._node_values(i)[self.scene.pipe_owner]

    # ------------------------------------------------------------------
    def draw(self, i: int) -> np.ndarray:
        """Update every panel to saved step ``i`` and return the RGB image."""
        r, nodes = self.result, self.nodes
        # 3D view
        pipe_points, bit_points = self.scene.positions(i)
        self.pipe.points = pipe_points
        self.pipe.point_data["value"] = self._point_values(i)
        if self.bit is not None:
            self.bit.points = bit_points
        self.plotter.render()
        self.image.set_data(self.plotter.screenshot(return_img=True))

        # top view (same meshes): path of the node over the last TRAIL_LENGTH frames
        k = int(np.searchsorted(self.frames, i))
        trail = np.array([self._node_center(j) for j in self.frames[max(0, k - TRAIL_LENGTH + 1):
                                                                     k + 1]])
        self.trail.points = np.vstack([np.repeat(trail[:1], TRAIL_LENGTH - len(trail), 0),
                                       trail])
        self.marker.points = trail[-1:]
        self.top_plotter.render()
        self.top_image.set_data(self.top_plotter.screenshot(return_img=True))

        # profiles
        d = self.depth[nodes]
        lateral = self.lateral[i, nodes] * 1e3
        self.lat_line.set_data(lateral, d)
        contact = lateral >= self.gap[nodes] * 1e3 * 0.999
        self.contact_dots.set_offsets(np.column_stack([lateral[contact], d[contact]])
                                      if contact.any() else np.empty((0, 2)))
        tension = self.tension[i, nodes] / 1e3
        self.axial_line.set_data(tension, d)
        sign_change = np.flatnonzero(np.diff(np.sign(tension)) != 0)
        if sign_change.size:
            k = sign_change[-1]  # linear interpolation of the zero crossing
            depth_np = d[k] + (d[k + 1] - d[k]) * tension[k] / (tension[k] - tension[k + 1])
            self.np_marker.set_ydata([depth_np, depth_np])
            self.np_text.set_text(f"neutral point\n{self.depth[-1] - depth_np:.1f} m above bit")
        else:
            self.np_marker.set_ydata([np.nan, np.nan])
            self.np_text.set_text("")

        # time cursor and KPI cards
        self.cursor.set_xdata([r.time[i], r.time[i]])
        self.kpi["Time"].set_text(f"{r.time[i]:.2f} s")
        self.kpi["Step"].set_text(f"{r.steps[i]}")
        self.kpi["WOB"].set_text(f"{self.wob[i] / 1e3:.1f} kN")
        self.kpi["Bit speed"].set_text(f"{self.rpm[i, 1]:.0f} rpm")
        self.kpi["Max lateral"].set_text(f"{lateral.max():.1f} mm")
        self.kpi["Nodes in contact"].set_text(f"{int(contact.sum())}")

        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self) -> None:
        import matplotlib.pyplot as plt

        self.plotter.close()
        self.top_plotter.close()
        plt.close(self.fig)


def dashboard(result: Result, path: str | Path, *, start: int | None = None,
              end: int | None = None, every: int = 1, fps: int = 25, **options) -> Path:
    """Write an MP4 dashboard video (``options``: see :class:`Dashboard`)."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _frames(result, start, end, every)
    if frames.size == 0:
        raise ValueError("no saved step in the requested range")
    board = Dashboard(result, frames, **options)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1) as video:
        for i in frames:
            video.append_data(board.draw(i))
    board.close()
    return path


def dashboard_frame(result: Result, path: str | Path, step: int | None = None,
                    **options) -> Path:
    """Write a PNG of the dashboard at one saved step (default: last)."""
    import imageio.v2 as imageio

    i = len(result.steps) - 1 if step is None else result.index(step)
    board = Dashboard(result, np.arange(len(result.steps)), **options)
    imageio.imwrite(path, board.draw(i))
    board.close()
    return Path(path)
