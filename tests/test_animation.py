"""Drawing of the deformed structure (PyVista is optional: tests are skipped without it)."""

import numpy as np
import pytest

from drillstring_fea import cases, solve
from drillstring_fea.animation import Scene, cylinders

pv = pytest.importorskip("pyvista")


def test_cylinder_rings_have_the_element_radius():
    coords = np.array([[0.0, 0, 0], [0, 0, -1], [0, 0, -2]])
    connect = np.array([[0, 1], [1, 2]])
    points, faces, owner = cylinders(coords, connect, np.array([0.1, 0.2]), n_panels=8)
    assert points.shape == (2 * 2 * 8, 3) and faces.size == 2 * 8 * 5
    distance = np.linalg.norm(points[:, :2] - coords[owner][:, :2], axis=1)
    np.testing.assert_allclose(distance, np.repeat([0.1, 0.2], 16))


def test_drill_bit_mesh_is_package_data():
    from drillstring_fea.animation import read_drill_bit

    points, faces = read_drill_bit()
    assert points.shape == (365, 3) and faces.size == 781 * 4
    assert faces.reshape(-1, 4)[:, 1:].max() < len(points)  # 0-based point numbers


@pytest.mark.parametrize("rings", ["tangent", "rotations"])
def test_rings_follow_the_nodes(rings):
    model, settings = cases.pure_bending(n_elements=4)
    result = solve(model, settings)
    scene = Scene(result, rings=rings)
    pipe, bit = scene.positions(len(result.steps) - 1)
    center = result.coords + result.displacements()[:, :3]
    # every ring point stays at one radius from its (displaced) node
    distance = np.linalg.norm(pipe - center[scene.pipe_owner], axis=1)
    np.testing.assert_allclose(distance, np.repeat(scene.radii, 2 * scene.n_panels), rtol=1e-9)
    assert bit is None  # no well -> no drill bit


def test_mp4_is_written(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from drillstring_fea.animation import animate

    model, settings = cases.pure_bending(n_elements=4)
    result = solve(model, settings)
    path = animate(result, tmp_path / "movie.mp4", every=5, fps=5, window_size=(320, 240))
    assert path.exists() and path.stat().st_size > 0


def test_dashboard_picture(tmp_path):
    pytest.importorskip("matplotlib")
    from drillstring_fea.dashboard import dashboard_frame

    model, settings = cases.tubing_in_well(case=2, n_elements=40)
    settings.t_final = 0.5
    result = solve(model, settings)
    path = dashboard_frame(result, tmp_path / "board.png", size=(960, 540))
    assert path.exists() and path.stat().st_size > 0
