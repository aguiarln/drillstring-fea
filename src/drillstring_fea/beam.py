"""Corotational 3D Euler-Bernoulli beam element (dissertation Sections 4.2-4.4, 4.7).

Idea of the corotational formulation (Section 4.2): each element carries a
local frame (X, Y, Z) that follows its rigid-body motion. Deformations - and
therefore internal forces - are computed only from what is left after removing
that rigid motion:

* the change of length of the chord (axial strain);
* the small rotations of each node relative to the element frame (bending, torsion).

All elements are processed together with NumPy arrays: an array with a
leading axis of size ``n_elements`` holds one 3x3 or 12x12 matrix per element.
Element DOF order: ``[u1 v1 w1 rx1 ry1 rz1  u2 v2 w2 rx2 ry2 rz2]`` (Eq. 12).
"""

from __future__ import annotations

import numpy as np

from .model import NDOF, Corrections, Model


# --------------------------------------------------------------------------
# Finite rotations (Section 4.3)
# --------------------------------------------------------------------------
def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix S(v) of Eq. 7, for vectors of shape (..., 3)."""
    s = np.zeros(v.shape[:-1] + (3, 3))
    s[..., 0, 1], s[..., 0, 2] = -v[..., 2], v[..., 1]
    s[..., 1, 0], s[..., 1, 2] = v[..., 2], -v[..., 0]
    s[..., 2, 0], s[..., 2, 1] = -v[..., 1], v[..., 0]
    return s


def rodrigues(theta: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation pseudo-vector theta (Eq. 5).

    R = I + sin(t)/t S + (1 - cos t)/t^2 S S, with t = |theta|.
    ``np.sinc`` gives sin(t)/t without dividing by zero when t = 0.
    """
    t = np.linalg.norm(theta, axis=-1)
    s = skew(theta)
    a = np.sinc(t / np.pi)[..., None, None]  # sin(t)/t
    b = (0.5 * np.sinc(t / (2 * np.pi)) ** 2)[..., None, None]  # (1 - cos t)/t^2
    return np.eye(3) + a * s + b * (s @ s)


def relative_rotations(node_frame: np.ndarray, elem_frame: np.ndarray) -> np.ndarray:
    """Rotation of a nodal frame (x, y, z) relative to the element frame (X, Y, Z), Eq. 43.

    2 sin(dtheta_x) = -z.Y + y.Z
    2 sin(dtheta_y) = -x.Z + z.X
    2 sin(dtheta_z) = -y.X + x.Y
    Frames are stored by rows: frame[..., 0, :] is the x axis, and so on.
    """
    x, y, z = node_frame[..., 0, :], node_frame[..., 1, :], node_frame[..., 2, :]
    X, Y, Z = elem_frame[..., 0, :], elem_frame[..., 1, :], elem_frame[..., 2, :]
    two_sin = np.stack(
        [
            np.sum(-z * Y + y * Z, axis=-1),
            np.sum(-x * Z + z * X, axis=-1),
            np.sum(-y * X + x * Y, axis=-1),
        ],
        axis=-1,
    )
    if np.any(np.abs(two_sin) > 2.0):
        raise RuntimeError("relative rotation could not be computed (frames degenerated); "
                           "try smaller load or time steps")
    return np.arcsin(two_sin / 2.0)


def initial_frames(x1: np.ndarray, x2: np.ndarray, normalize: bool) -> np.ndarray:
    """Element frames at t = 0 (rows X, Y, Z), shape (n_elements, 3, 3).

    X follows the element. Y = e_z x X and Z = X x Y; for vertical elements,
    where e_z x X = 0, Y and Z are built from e_y instead.

    ``normalize = False`` reproduces the dissertation implementation, where Y
    and Z are not normalized (see ``Corrections.orthonormal_initial_triads``).
    """
    X = (x2 - x1) / np.linalg.norm(x2 - x1, axis=1)[:, None]
    Y = np.cross([0.0, 0.0, 1.0], X)
    Z = np.cross(X, Y)
    vertical = np.linalg.norm(Y, axis=1) == 0.0
    Z[vertical] = np.cross(X[vertical], [0.0, 1.0, 0.0])
    Y[vertical] = np.cross(Z[vertical], X[vertical])
    if normalize:
        Y /= np.linalg.norm(Y, axis=1)[:, None]
        Z /= np.linalg.norm(Z, axis=1)[:, None]
    return np.stack([X, Y, Z], axis=1)


# --------------------------------------------------------------------------
# Element matrices in local axes (Section 4.4)
# --------------------------------------------------------------------------
def _matrices(rows: list[list], n: int) -> np.ndarray:
    """Build n matrices from a matrix 'literal' whose entries are scalars or (n,) arrays."""
    out = np.zeros((n, len(rows), len(rows[0])))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            out[:, i, j] = value
    return out


def linear_stiffness(E, G, A, I, L) -> np.ndarray:
    """Linear stiffness K_L of Euler-Bernoulli beams, shape (n, 12, 12).

    Closed form of the integral of Eq. 15 for a two-node element with cubic
    bending and linear axial/torsion interpolation (circular tube: Iy = Iz = I,
    J = 2 I). Rows and columns follow the element DOF order of Eq. 12.
    """
    a = E * A / L  # axial
    t = G * 2 * I / L  # torsion (J = 2 I)
    b = 12 * E * I / L**3  # bending terms
    c = 6 * E * I / L**2
    d = 4 * E * I / L
    e = 2 * E * I / L
    o = 0.0
    rows = [
        [ a,  o,  o,  o,  o,  o, -a,  o,  o,  o,  o,  o],
        [ o,  b,  o,  o,  o,  c,  o, -b,  o,  o,  o,  c],
        [ o,  o,  b,  o, -c,  o,  o,  o, -b,  o, -c,  o],
        [ o,  o,  o,  t,  o,  o,  o,  o,  o, -t,  o,  o],
        [ o,  o, -c,  o,  d,  o,  o,  o,  c,  o,  e,  o],
        [ o,  c,  o,  o,  o,  d,  o, -c,  o,  o,  o,  e],
        [-a,  o,  o,  o,  o,  o,  a,  o,  o,  o,  o,  o],
        [ o, -b,  o,  o,  o, -c,  o,  b,  o,  o,  o, -c],
        [ o,  o, -b,  o,  c,  o,  o,  o,  b,  o,  c,  o],
        [ o,  o,  o, -t,  o,  o,  o,  o,  o,  t,  o,  o],
        [ o,  o, -c,  o,  e,  o,  o,  o,  c,  o,  d,  o],
        [ o,  c,  o,  o,  o,  e,  o, -c,  o,  o,  o,  d],
    ]  # fmt: skip
    return _matrices(rows, len(L))


def geometric_stiffness(f: np.ndarray, A, I, L, full: bool = False) -> np.ndarray:
    """Geometric stiffness K_NL (Eq. 16) from the end forces f of node 1.

    F1 = axial force, F2/F3 = shear forces, F4 = torque, F5/F6 = bending moments.

    Closed form of the integral of Eq. 16 (Aguiar 2013, Eq. B.2). The C1..C4
    terms include the I/A contributions of the nonlinear strains.

    ``full = False``: matrix used in the dissertation.
    ``full = True``: complete matrix of Aguiar (2013), Eq. B.2 (12x12 part):
    l*F2/6 instead of I*F2/6 in entries (4,5)/(5,4) (1-based, i.e. rows 3/4
    below), plus the shear-force terms +-F2, +-F3 that couple the node-1
    translations with the node-2 rotations. Without these terms Newton-Raphson
    converges slowly (or not at all) when bending and torsion are coupled,
    e.g. in ``curved_cantilever``. See docs/theory.md, Section 2.2.
    """
    F1, F2, F3, F4, F5, F6 = (f[:, k] for k in range(6))
    J = 2 * I
    C1 = 6 / (5 * L) + 12 * I / (A * L**3)
    C2 = 1 / 10 + 6 * I / (A * L**2)
    C3 = 2 * L / 15 + 4 * I / (A * L)
    C4 = L / 30 - 2 * I / (A * L)
    p = (L if full else I) * F2 / 6
    o = 0.0
    rows = [
        [-F1/L,      o,      o,         o,  -F5/L,  -F6/L,  F1/L,      o,      o,         o,   F5/L,   F6/L],
        [    o, -C1*F1,      o,      F5/L,  -F4/L, -C2*F1,     o,  C1*F1,      o,     -F5/L,   F4/L, -C2*F1],
        [    o,      o, -C1*F1,      F6/L,  C2*F1,  -F4/L,     o,      o,  C1*F1,     -F6/L,  C2*F1,   F4/L],
        [    o,   F5/L,   F6/L, -J*F1/(A*L),    p, F3*L/6,     o,  -F5/L,  -F6/L, J*F1/(A*L), -F2*L/6, -F3*L/6],
        [-F5/L,  -F4/L,  C2*F1,         p, -C3*F1,      o,  F5/L,   F4/L, -C2*F1,   -F2*L/6,  C4*F1,  -F4/2],
        [-F6/L, -C2*F1,  -F4/L,    F3*L/6,      o, -C3*F1,  F6/L,  C2*F1,   F4/L,   -F3*L/6,   F4/2,  C4*F1],
        [ F1/L,      o,      o,         o,   F5/L,   F6/L, -F1/L,      o,      o,         o,  -F5/L,  -F6/L],
        [    o,  C1*F1,      o,     -F5/L,   F4/L,  C2*F1,     o, -C1*F1,      o,      F5/L,  -F4/L,  C2*F1],
        [    o,      o,  C1*F1,     -F6/L, -C2*F1,   F4/L,     o,      o, -C1*F1,      F6/L, -C2*F1,  -F4/L],
        [    o,  -F5/L,  -F6/L, J*F1/(A*L), -F2*L/6, -F3*L/6,  o,   F5/L,   F6/L, -J*F1/(A*L), F2*L/6, F3*L/6],
        [ F5/L,   F4/L,  C2*F1,   -F2*L/6,  C4*F1,   F4/2, -F5/L,  -F4/L, -C2*F1,    F2*L/6, -C3*F1,      o],
        [ F6/L, -C2*F1,   F4/L,   -F3*L/6,  -F4/2,  C4*F1, -F6/L,  C2*F1,  -F4/L,    F3*L/6,      o, -C3*F1],
    ]  # fmt: skip
    K = _matrices(rows, len(L))
    if full:
        # shear-force terms of Aguiar (2013), Eq. B.2 (symmetric); 0-based indices,
        # e.g. (0, 10) = row u1, column ry2
        for (i, j), value in {(0, 10): F3, (0, 11): -F2, (1, 9): -F3, (2, 9): F2,
                              (6, 10): -F3, (6, 11): F2, (7, 9): F3, (8, 9): -F2}.items():
            K[:, i, j] += value
            K[:, j, i] += value
    return K


def lumped_mass(rho, A, I, L, torsional_factor: float) -> np.ndarray:
    """Diagonal of the lumped mass matrix (Eq. 18), shape (n, 12).

    (rho A L / 2) [1, 1, 1, factor*J/A, L^2/210, L^2/210] for each node, J = 2 I.
    """
    one = np.ones_like(L)
    node = np.stack([one, one, one, torsional_factor * 2 * I / A, L**2 / 210, L**2 / 210], 1)
    return (rho * A * L / 2)[:, None] * np.hstack([node, node])


def block_rotation(frames: np.ndarray) -> np.ndarray:
    """12x12 transformation R_G = diag(R_e, R_e, R_e, R_e) of Eq. 47."""
    R = np.zeros((frames.shape[0], 12, 12))
    for b in range(4):
        R[:, 3 * b:3 * b + 3, 3 * b:3 * b + 3] = frames
    return R


# --------------------------------------------------------------------------
# All elements of a model
# --------------------------------------------------------------------------
class Beams:
    """Current state (geometry, frames, internal forces) of all elements."""

    def __init__(self, model: Model, corrections: Corrections) -> None:
        self.n1, self.n2 = model.connect[:, 0], model.connect[:, 1]
        self.A, self.I = model.A, model.I
        self.full_geometric = corrections.full_geometric_stiffness
        self.n_nodes = model.n_nodes

        # Global DOF numbers of the 12 element DOFs (used for assembly)
        self.dofs = np.hstack([NDOF * self.n1[:, None] + np.arange(NDOF),
                               NDOF * self.n2[:, None] + np.arange(NDOF)])

        self.x = model.coords.copy()  # current nodal positions
        self.L0 = model.element_lengths()  # initial (undeformed) lengths
        frames = initial_frames(self.x[self.n1], self.x[self.n2],
                                corrections.orthonormal_initial_triads)
        self.frame = frames.copy()  # element frame R_e (rows X, Y, Z)
        self.frame_node1 = frames.copy()  # nodal frames R_n1, R_n2 (start equal to R_e)
        self.frame_node2 = frames.copy()
        self.R = block_rotation(frames)  # R_G of Eq. 47
        self.forces = np.zeros((model.n_elements, 12))  # local end forces f_int

        self.K_L = linear_stiffness(model.E, model.G, model.A, model.I, self.L0)
        self.mass = lumped_mass(model.rho, model.A, model.I, self.L0,
                                model.torsional_inertia_factor)

    def update(self, du: np.ndarray) -> None:
        """Update geometry and internal forces with the increment ``du`` (steps 4.1-4.4)."""
        du = du.reshape(self.n_nodes, NDOF)

        # 4.1  new nodal positions, element axis X and deformed length
        self.x += du[:, :3]
        chord = self.x[self.n2] - self.x[self.n1]
        length = np.linalg.norm(chord, axis=1)
        X = chord / length[:, None]

        # 4.2.1  rotate the nodal frames by the incremental rotations (Eq. 41).
        # With frames stored by rows, R_new = R_inc R_old becomes rows @ R_inc^T.
        R_inc = rodrigues(du[:, 3:])
        self.frame_node1 = self.frame_node1 @ R_inc[self.n1].transpose(0, 2, 1)
        self.frame_node2 = self.frame_node2 @ R_inc[self.n2].transpose(0, 2, 1)

        # 4.2.2  element frame (Eq. 42): Y = mean of the nodal y axes, Z = X x Y,
        # then Y = Z x X makes the frame exactly orthogonal.
        Y = 0.5 * (self.frame_node1[:, 1, :] + self.frame_node2[:, 1, :])
        Z = np.cross(X, Y)
        Z /= np.linalg.norm(Z, axis=1)[:, None]
        Y = np.cross(Z, X)
        self.frame = np.stack([X, Y, Z], axis=1)
        self.R = block_rotation(self.frame)

        # 4.3  relative displacement vector u_r (Eq. 44)
        u_r = np.zeros((len(length), 12))
        u_r[:, 3:6] = relative_rotations(self.frame_node1, self.frame)
        u_r[:, 6] = length - self.L0
        u_r[:, 9:12] = relative_rotations(self.frame_node2, self.frame)

        # 4.4  internal forces in local axes f_int = K_L u_r (Eq. 45)
        self.forces = np.einsum("eij,ej->ei", self.K_L, u_r)

    def internal_force_vector(self) -> np.ndarray:
        """Global internal force vector F_int (Eq. 46 + assembly, step 4.4.2)."""
        f_global = np.einsum("eji,ej->ei", self.R, self.forces)  # R_G^T f_int
        return np.bincount(self.dofs.ravel(), f_global.ravel(), minlength=NDOF * self.n_nodes)

    def tangent_stiffness(self) -> np.ndarray:
        """Element tangent stiffness R_G^T (K_L + K_NL) R_G in global axes, (n, 12, 12)."""
        K_NL = geometric_stiffness(self.forces, self.A, self.I, self.L0, self.full_geometric)
        return self.R.transpose(0, 2, 1) @ (self.K_L + K_NL) @ self.R

    def mass_matrices(self) -> np.ndarray:
        """Element lumped mass R_G^T M R_G in global axes, (n, 12, 12)."""
        return self.R.transpose(0, 2, 1) @ (self.mass[:, :, None] * self.R)
