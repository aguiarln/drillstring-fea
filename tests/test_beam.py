"""Unit tests of the beam element (rotations, matrices, corotational kinematics)."""

import numpy as np

from drillstring_fea import DX, RZ, Corrections, Model, NodalLoad, Settings, clamp, solve
from drillstring_fea.beam import (
    Beams,
    geometric_stiffness,
    initial_frames,
    linear_stiffness,
    lumped_mass,
    relative_rotations,
    rodrigues,
    skew,
)

E, G, A, I, L = 210e9, 210e9 / 2.6, 0.01, 2e-5, 2.0


def straight_beam(coords) -> Model:
    coords = np.asarray(coords, float)
    n = len(coords)
    return Model("test", coords, np.column_stack([np.arange(n - 1), np.arange(1, n)]),
                 E=E, G=G, A=A, I=I, rho=0.0, supports=clamp(0))


def test_skew_matrix_gives_cross_product():
    v, w = np.random.default_rng(0).normal(size=(2, 4, 3))
    np.testing.assert_allclose((skew(v) @ w[..., None])[..., 0], np.cross(v, w))


def test_rodrigues_is_a_rotation_and_identity_at_zero():
    R = rodrigues(np.random.default_rng(1).normal(size=(20, 3)))
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.broadcast_to(np.eye(3), R.shape),
                               atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(R), 1.0)
    np.testing.assert_array_equal(rodrigues(np.zeros(3)), np.eye(3))


def test_relative_rotation_of_a_small_rotation():
    nodal = rodrigues(np.array([0.01, -0.02, 0.03])).T  # frame stored by rows
    np.testing.assert_allclose(relative_rotations(nodal, np.eye(3)), [0.01, -0.02, 0.03],
                               atol=1e-5)


def test_linear_stiffness_is_symmetric_and_has_no_rigid_body_forces():
    K = linear_stiffness(E, G, A, I, np.array([L]))[0]
    np.testing.assert_allclose(K, K.T)
    for d in range(3):  # rigid translations
        u = np.zeros(12)
        u[d] = u[d + 6] = 1.0
        np.testing.assert_allclose(K @ u, 0.0, atol=1e-6)


def test_geometric_stiffness_is_symmetric():
    f = np.random.default_rng(2).normal(size=(1, 12))
    for full in (False, True):
        K = geometric_stiffness(f, np.array([A]), np.array([I]), np.array([L]), full)[0]
        np.testing.assert_allclose(K, K.T, atol=1e-12)


def test_lumped_mass_total():
    m = lumped_mass(np.array([7850.0]), np.array([A]), np.array([I]), np.array([L]), 1 / 420)[0]
    assert np.isclose(m[0] + m[6], 7850.0 * A * L)


def test_initial_frames_are_orthonormal_for_vertical_and_horizontal_elements():
    frames = initial_frames(np.zeros((2, 3)), np.array([[1.0, 0, 0], [0, 0, -1.0]]), False)
    np.testing.assert_allclose(frames @ frames.transpose(0, 2, 1),
                               np.broadcast_to(np.eye(3), frames.shape), atol=1e-15)


def test_rigid_body_motion_gives_no_internal_forces():
    """Key property of the corotational formulation (Section 4.2)."""
    model = straight_beam([[0, 0, 0], [L, 0, 0], [2 * L, 0, 0]])
    beams = Beams(model, Corrections())
    theta = np.array([0.3, -0.2, 0.5])
    new_x = model.coords @ rodrigues(theta).T + [1.0, 2.0, 3.0]
    du = np.zeros(model.n_dofs).reshape(-1, 6)
    du[:, :3] = new_x - model.coords
    du[:, 3:] = theta
    beams.update(du.ravel())
    np.testing.assert_allclose(beams.forces, 0.0, atol=1e-6)


def test_cantilever_tip_deflection_matches_beam_theory():
    n = 8
    coords = np.zeros((n + 1, 3))
    coords[:, 0] = np.linspace(0, L, n + 1)
    model = straight_beam(coords)
    model.loads = [NodalLoad(n, 1, 10.0)]
    r = solve(model, Settings())
    assert np.isclose(r.displacements()[-1, 1], 10.0 * L**3 / (3 * E * I), rtol=1e-6)


def test_slightly_inclined_column_needs_orthonormal_frames():
    """1 micrometre imperfection: with the original frames the bending stiffness halves."""
    n, h = 10, 10.0
    coords = np.zeros((n + 1, 3))
    coords[:, 2] = -np.linspace(0, h, n + 1)
    coords[:, 0] = 1e-6 * np.sin(np.pi * coords[:, 2] / h)

    def tip(corrections, load):
        model = straight_beam(coords)
        model.loads = [load]
        r = solve(model, Settings(), corrections)
        return r.displacements()[-1, load.dof] if r.converged else np.nan

    fixed, original = Corrections(), Corrections.dissertation()
    exact = h**3 / (3 * E * I)
    assert np.isclose(tip(fixed, NodalLoad(n, DX, 1.0)), exact, rtol=1e-6)
    assert np.isclose(tip(original, NodalLoad(n, DX, 1.0)), 2 * exact, rtol=0.01)
    assert np.isclose(tip(fixed, NodalLoad(n, RZ, 1.0)), h / (G * 2 * I), rtol=1e-6)
