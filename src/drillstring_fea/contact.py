"""Well-bore contact with penalty springs (dissertation Section 4.8).

Wall contact (Eqs. 53-59): for each contact node, ``n_r`` pairs of orthogonal
springs are placed at angles theta = 0, pi/(2 n_r), ..., i.e. the local axes
x', y' rotated by theta. A spring is active when the node displacement along
its direction exceeds the radial gap; the set of springs approximates a
circular well by a polygon (Figures 22-24).

Bit/rock contact (Eq. 60): one vertical spring at the bit node, active when
the axial displacement exceeds the axial gap.

Displacements are measured from the initial nodal positions, which are
assumed to lie on the well axis. A gap equal to 0 means "no contact spring".
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .model import DX, DY, DZ, NDOF, Model


class Contact:
    """Contact springs of a model; ``forces_and_stiffness(U)`` is called at every iteration.

    The spring forces are added to the internal force vector F_int and their
    stiffness to the tangent matrix K_t (they act like extra elements).
    """

    def __init__(self, model: Model, symmetric_tangent: bool = False) -> None:
        self.n_dofs = model.n_dofs
        self.k_wall = model.wall_stiffness
        self.k_rock = model.rock_stiffness
        self.symmetric_tangent = symmetric_tangent

        # spring directions theta_i = i * pi / (2 n_r), i = 0 .. n_r - 1
        theta = np.arange(model.n_spring_directions) * np.pi / (2 * model.n_spring_directions)
        self.c = np.cos(theta)[:, None]  # shape (n_r, 1) -> broadcasts over nodes
        self.s = np.sin(theta)[:, None]

        on = model.wall_gaps != 0
        self.gap = model.wall_gaps[on]
        self.ix = NDOF * model.wall_nodes[on] + DX  # global DOF of U_x of each node
        self.iy = NDOF * model.wall_nodes[on] + DY

        on = model.bit_gaps != 0
        self.bit_gap = model.bit_gaps[on]
        self.iz = NDOF * model.bit_nodes[on] + DZ

    @property
    def active(self) -> bool:
        return self.ix.size > 0 or self.iz.size > 0

    def forces_and_stiffness(self, u: np.ndarray) -> tuple[np.ndarray, sp.csr_matrix]:
        """Contact contribution to the internal force vector and to the tangent matrix."""
        f = np.zeros(self.n_dofs)
        rows, cols, vals = [], [], []

        if self.ix.size:
            c, s, k, gap = self.c, self.s, self.k_wall, self.gap
            ux, uy = u[self.ix], u[self.iy]
            # Eq. 54: displacement in the rotated spring axes, u_r = T^T U_G
            u1 = c * ux + s * uy
            u2 = -s * ux + c * uy
            # Eq. 57: f = k (1 - gap/|u|) u for the springs beyond the gap
            on1, on2 = np.abs(u1) > gap, np.abs(u2) > gap
            with np.errstate(divide="ignore", invalid="ignore"):
                f1 = np.where(on1, k * (1 - gap / np.abs(u1)) * u1, 0.0)
                f2 = np.where(on2, k * (1 - gap / np.abs(u2)) * u2, 0.0)
            k1, k2 = k * on1, k * on2  # Eq. 58: spring stiffness when active
            # Eq. 59: back to global axes, F = T f and K = T K_r T^T, summed over directions
            np.add.at(f, self.ix, (c * f1 - s * f2).sum(axis=0))
            np.add.at(f, self.iy, (s * f1 + c * f2).sum(axis=0))
            kxx = (c * c * k1 + s * s * k2).sum(axis=0)
            kyy = (s * s * k1 + c * c * k2).sum(axis=0)
            kxy = (c * s * (k1 - k2)).sum(axis=0)
            rows += [self.ix, self.iy, self.ix]
            cols += [self.ix, self.iy, self.iy]
            vals += [kxx, kyy, kxy]
            if self.symmetric_tangent:  # the dissertation code omits this term
                rows.append(self.iy)
                cols.append(self.ix)
                vals.append(kxy)

        if self.iz.size:
            uz = u[self.iz]
            hit = np.abs(uz) > self.bit_gap
            # Eq. 60: F(z) = F(z) - k_r (|U_z| - gap) and K(z,z) = K(z,z) + k_r
            np.add.at(f, self.iz, -np.where(hit, self.k_rock * (np.abs(uz) - self.bit_gap), 0.0))
            rows.append(self.iz)
            cols.append(self.iz)
            vals.append(self.k_rock * hit)

        if not rows:
            return f, sp.csr_matrix((self.n_dofs, self.n_dofs))
        K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.n_dofs, self.n_dofs))
        return f, K.tocsr()
