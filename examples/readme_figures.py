"""Pictures of the README (docs/images/).

Usage:  python examples/readme_figures.py    (needs the [animation] extras; ~3 minutes)

* ``pure_bending.png``: Section 6.1, the cantilever rolled into a circle by an
  end moment, finite elements against the exact solution (Eqs. 64-66).
* ``helical_buckling.png``: dashboard of the Section 6.5 string with a weight
  on bit of 150 kN, helically buckled after the rotation starts. The analysis
  (~2 minutes) is saved in results/helical_drillstring.npz and reused.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

from drillstring_fea import DX, DZ, Result, cases, solve  # noqa: E402
from drillstring_fea.animation import cylinders  # noqa: E402
from drillstring_fea.dashboard import THEME, _style, dashboard_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"


def pure_bending(path: Path) -> None:
    model, settings = cases.pure_bending(n_elements=10)
    r = solve(model, settings)
    L, tip = model.info["length"], model.n_nodes - 1
    steps = (4, 8, 12, 16, 20)  # rotation of the tip: 2 pi k / 20
    cmap = matplotlib.colormaps["plasma"]

    # 3D view: finite elements (tubes) and exact circular arcs (white lines)
    p = pv.Plotter(off_screen=True, window_size=(1100, 900))
    p.set_background(THEME["panel"])
    p.view_xz()  # x to the right, z up; the camera looks along +y
    front = -1.0  # the exact arcs are drawn in front of the tubes (camera side)
    straight, faces, _ = cylinders(model.coords, model.connect, np.full(model.n_elements, 0.07), 24)
    p.add_mesh(pv.PolyData(straight, faces), color=THEME["muted"], opacity=0.35)
    for n, step in enumerate(steps):
        x = model.coords + r.displacements(step)[:, :3]
        points, faces, _ = cylinders(x, model.connect, np.full(model.n_elements, 0.07), 24)
        p.add_mesh(pv.PolyData(points, faces), color=cmap(0.35 + 0.6 * n / (len(steps) - 1))[:3],
                   smooth_shading=True, specular=0.5)
        p.add_mesh(pv.PolyData(x), color=THEME["text"], point_size=9,
                   render_points_as_spheres=True)
        theta = 2 * np.pi * step / 20
        phi = np.linspace(0, theta, 200)  # exact deformed shape: arc of radius L / theta
        arc = np.column_stack([L / theta * np.sin(phi), np.full_like(phi, front),
                               L / theta * (1 - np.cos(phi))])
        p.add_mesh(pv.lines_from_points(arc), color="white", line_width=2.5)
    p.add_mesh(pv.Box((-0.25, 0, -0.6, 0.6, -0.6, 0.6)), color=THEME["well"])  # support
    p.enable_anti_aliasing("ssaa")
    p.enable_parallel_projection()
    p.reset_camera()
    p.camera.zoom(1.15)
    image = p.screenshot(return_img=True)
    p.close()

    # figure: 3D view + tip displacements (Figures 27-28)
    fig = plt.figure(figsize=(16, 7.2), dpi=100, facecolor=THEME["background"])
    fig.text(0.015, 0.94, "Cantilever rolled into a circle — Section 6.1", color=THEME["text"],
             fontsize=20, fontweight="bold", va="center")
    fig.text(0.015, 0.885, "End moment M = 2π EI / L in 20 steps · 10 corotational beam "
             "elements (tubes) against the exact circular arcs (white)",
             color=THEME["muted"], fontsize=11, va="center")
    ax3d = fig.add_axes([0.015, 0.04, 0.52, 0.8])
    ax3d.imshow(image)
    ax3d.set_axis_off()
    for n, step in enumerate(steps):
        ax3d.text(0.03, 0.95 - 0.055 * n, f"M = {step / 20:.1f} M_max",
                  transform=ax3d.transAxes, color=cmap(0.35 + 0.6 * n / (len(steps) - 1)),
                  fontsize=11, fontweight="bold")

    ax = fig.add_axes([0.6, 0.12, 0.38, 0.7])
    _style(ax, "Tip displacements")
    ratio = r.static_ratio
    fine = np.linspace(1e-3, 1, 300)
    th = 2 * np.pi * fine
    ax.plot(fine, L * (1 - np.sin(th) / th), color=THEME["accent"], lw=2, label="Ux exact (Eq. 65)")
    ax.plot(fine, L * (1 - np.cos(th)) / th, color=THEME["accent2"], lw=2,
            label="Uz exact (Eq. 66)")
    ax.plot(ratio, -r.history(tip, DX), "o", color=THEME["accent"], mfc=THEME["panel"], mew=2,
            label="-Ux finite elements")
    ax.plot(ratio, r.history(tip, DZ), "s", color=THEME["accent2"], mfc=THEME["panel"], mew=2,
            label="Uz finite elements")
    ax.set_xlabel("M / M_max")
    ax.set_ylabel("tip displacement (m)")
    ax.legend(loc="upper left", fontsize=10, facecolor=THEME["panel"], edgecolor=THEME["border"],
              labelcolor=THEME["muted"])
    fig.savefig(path, facecolor=THEME["background"])
    plt.close(fig)


def helical_buckling(path: Path, step: int = 4400) -> None:
    npz = ROOT / "results" / "helical_drillstring.npz"
    if not npz.exists():
        model, settings = cases.rotating_drillstring(t_rotation=30.0)  # WOB 150 kN
        settings.t_final, settings.save_every = 45.0, 2
        solve(model, settings).save(npz)
    dashboard_frame(Result.load(npz), path, step=step, z_range=(-4000, -3650),
                    title="Helical buckling — 4000 m drill string, WOB 150 kN, 80 rpm")


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    pure_bending(IMAGES / "pure_bending.png")
    helical_buckling(IMAGES / "helical_buckling.png")
    print(f"written to {IMAGES}")
