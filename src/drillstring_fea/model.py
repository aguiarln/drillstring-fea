"""Input data of an analysis: the structure (Model) and how to analyze it (Settings).

Everything a user can change lives in this file:

* ``Model``         geometry, cross sections, materials, supports, loads, contact;
* ``Settings``      number of static steps, time step, Newmark parameters, tolerances;
* ``Corrections``   optional fixes to the implementation used in the dissertation.

Conventions
-----------
* SI units (N, m, kg, s, rad, Pa).
* Nodes and elements are numbered from 0.
* Each node has 6 degrees of freedom (DOFs), in this order (dissertation Eq. 12):
  ``DX, DY, DZ`` (translations) and ``RX, RY, RZ`` (rotations).
  DOF ``d`` of node ``n`` has the global index ``6 * n + d``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.80665  # m/s^2

NDOF = 6  # degrees of freedom per node
DX, DY, DZ, RX, RY, RZ = range(NDOF)
DOF_NAMES = ("DX", "DY", "DZ", "RX", "RY", "RZ")


# --------------------------------------------------------------------------
# Supports and loads
# --------------------------------------------------------------------------
@dataclass
class PrescribedDisplacement:
    """Imposed displacement/rotation: ``value * factor(history, step)``.

    Use ``value = 0`` for a fixed support. The available histories are listed
    in ``drillstring_fea.loads`` (e.g. "linear", "constant", "ramp").
    """

    node: int
    dof: int
    value: float = 0.0
    history: str = "linear"
    t_switch: float = 0.0  # switching time (s) of the "hold" / "start" histories


@dataclass
class NodalLoad:
    """Concentrated force/moment: ``value * factor(history, step)``."""

    node: int
    dof: int
    value: float
    history: str = "linear"
    t_switch: float = 0.0


def clamp(node: int, dofs=range(NDOF)) -> list[PrescribedDisplacement]:
    """Fixed support: zero displacement on the given DOFs (default: all six)."""
    return [PrescribedDisplacement(node, d, 0.0) for d in dofs]


# --------------------------------------------------------------------------
# Cross sections
# --------------------------------------------------------------------------
def tube_section(inner_radius: float, outer_radius: float) -> dict:
    """Area A, inertia I and outer/inner areas Ae/Ai of a circular tube.

    The beam element assumes Iy = Iz = I and polar inertia J = 2 I.
    """
    ri, ro = inner_radius, outer_radius
    return {
        "A": math.pi * (ro**2 - ri**2),
        "I": math.pi * (ro**4 - ri**4) / 4.0,
        "Ae": math.pi * ro**2,
        "Ai": math.pi * ri**2,
    }


def shear_modulus(E, poisson: float = 0.3):
    """G = E / (2 (1 + nu)), isotropic material (nu = 0.3 for steel)."""
    return np.asarray(E) / (2.0 * (1.0 + poisson))


# --------------------------------------------------------------------------
# The structural model
# --------------------------------------------------------------------------
@dataclass
class Model:
    """Finite element model of a 3D frame or drill string.

    Element properties (``E, G, A, I, rho, ...``) may be scalars (same value for
    every element) or arrays with one value per element.

    Self-weight acts along -Z (Section 6.4/6.5 strings hang along -Z):
    ``W = -(rho A + rho_if Ai - rho_ef Ae) g L`` per element, half to each node,
    where ``rho_if`` / ``rho_ef`` are the internal / external fluid densities.
    """

    name: str
    coords: np.ndarray  # (n_nodes, 3) nodal coordinates
    connect: np.ndarray  # (n_elements, 2) first and second node of each element
    E: np.ndarray  # Young's modulus
    G: np.ndarray  # shear modulus
    A: np.ndarray  # cross-section area
    I: np.ndarray  # second moment of area (Iy = Iz = I, J = 2 I)
    rho: np.ndarray  # material density
    Ae: np.ndarray = 0.0  # outer area (buoyancy)
    Ai: np.ndarray = 0.0  # inner area (internal fluid)
    rho_if: np.ndarray = 0.0  # internal fluid density
    rho_ef: np.ndarray = 0.0  # external fluid density
    pipe_diam: np.ndarray = 0.0  # outer diameter (only for post-processing)
    well_diam: float | None = None  # well diameter (only for drawings; gaps are given below)

    supports: list[PrescribedDisplacement] = field(default_factory=list)
    loads: list[NodalLoad] = field(default_factory=list)
    self_weight_history: str = "constant"  # load history of the self-weight (loads.py)

    # Well-bore contact (Section 4.8). ``wall_gaps`` are radial clearances of the
    # ``wall_nodes``; ``bit_nodes`` touch the rock after an axial gap ``bit_gaps``.
    wall_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    wall_gaps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bit_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    bit_gaps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wall_stiffness: float = 1.0e6  # k_s, Eq. 57-58
    rock_stiffness: float = 1.0e13  # k_r, Eq. 60
    n_spring_directions: int = 4  # n_r, Figure 23

    # Rayleigh damping D = alpha M + beta K_t (Eq. 20)
    rayleigh_alpha: float = 1.133636568737845
    rayleigh_beta: float = 2.908670114430112e-04

    # Lumped torsional mass = (m/2) * factor * (J/A) (Eq. 18 uses factor 1/420;
    # 0 removes the torsional inertia, as in the "N.I." runs of Section 6.5).
    torsional_inertia_factor: float = 1.0 / 420.0

    description: str = ""  # one line printed by the command line
    info: dict = field(default_factory=dict)  # free data of the case (used by tests/scripts)

    def __post_init__(self) -> None:
        self.coords = np.atleast_2d(np.asarray(self.coords, dtype=float))
        self.connect = np.atleast_2d(np.asarray(self.connect, dtype=int))
        # Turn scalar properties into one value per element.
        for name in ("E", "G", "A", "I", "rho", "Ae", "Ai", "rho_if", "rho_ef", "pipe_diam"):
            value = np.asarray(getattr(self, name), dtype=float)
            setattr(self, name, np.broadcast_to(value, (self.n_elements,)).copy())
        for name in ("wall_nodes", "bit_nodes"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=int))
        for name in ("wall_gaps", "bit_gaps"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.check()

    @property
    def n_nodes(self) -> int:
        return self.coords.shape[0]

    @property
    def n_elements(self) -> int:
        return self.connect.shape[0]

    @property
    def n_dofs(self) -> int:
        return NDOF * self.n_nodes

    def check(self) -> None:
        if self.coords.shape[1] != 3 or self.connect.shape[1] != 2:
            raise ValueError("coords must be (n_nodes, 3) and connect (n_elements, 2)")
        if self.connect.min() < 0 or self.connect.max() >= self.n_nodes:
            raise ValueError("connectivity refers to a node that does not exist")
        for bc in [*self.supports, *self.loads]:
            if not (0 <= bc.node < self.n_nodes and 0 <= bc.dof < NDOF):
                raise ValueError(f"invalid node/DOF in {bc}")

    def element_lengths(self) -> np.ndarray:
        n1, n2 = self.connect.T
        return np.linalg.norm(self.coords[n2] - self.coords[n1], axis=1)

    def self_weight(self) -> np.ndarray:
        """Global vector of self-weight + fluid forces (along Z)."""
        L = self.element_lengths()
        w = -(self.rho * self.A + self.rho_if * self.Ai - self.rho_ef * self.Ae) * GRAVITY * L
        f = np.zeros(self.n_dofs)
        n1, n2 = self.connect.T
        np.add.at(f, NDOF * n1 + DZ, w / 2.0)
        np.add.at(f, NDOF * n2 + DZ, w / 2.0)
        return f


# --------------------------------------------------------------------------
# Analysis settings
# --------------------------------------------------------------------------
@dataclass
class Settings:
    """How the analysis is run (Section 4.7).

    The analysis applies ``n_static_steps`` static load steps and then, if
    ``dynamic`` is true, Newmark time steps of size ``dt`` up to ``t_final``.
    Steps are numbered 1, 2, ... (static steps first), as in the dissertation.
    """

    n_static_steps: int = 1
    dynamic: bool = False
    t_final: float = 5.0  # duration of the dynamic phase (s)
    dt: float = 0.01  # time step (s)
    beta: float = 0.25  # Newmark beta (Eq. 24)
    gamma: float = 0.50  # Newmark gamma (Eq. 23)
    max_iter: int = 100  # maximum Newton-Raphson iterations per step
    tol_disp: float = 1.0e-4  # displacement tolerance, Eq. 34
    tol_res: float = 1.0e-2  # residual tolerance, Eq. 35
    ramp_time: float = 5.0  # duration of the "ramp" history (s)
    save_every: int = 1  # store every n-th step in the results

    @property
    def n_dynamic_steps(self) -> int:
        # t = 0, dt, 2 dt, ..., t_final  ->  t_final/dt + 1 steps
        return int(math.floor(self.t_final / self.dt + 1e-9)) + 1 if self.dynamic else 0

    @property
    def n_steps(self) -> int:
        return self.n_static_steps + self.n_dynamic_steps


@dataclass(frozen=True)
class Corrections:
    """Fixes to the implementation used to produce the dissertation results.

    ``Corrections()``               default: the first two fixes are on;
    ``Corrections.dissertation()``  all off, reproduces the published results;
    ``Corrections.all()``           all on.

    Each item is explained in ``docs/theory.md``, Part 2.
    """

    orthonormal_initial_triads: bool = True
    """Normalize the initial element axes Y, Z (Mourelle 1993, Sec. III.6: the
    nodal reference systems are orthonormal). Without it, slightly inclined
    elements get scaled relative rotations (half bending, ~zero torsion stiffness)."""

    full_geometric_stiffness: bool = True
    """Use the complete geometric stiffness of Aguiar (2013), Eq. B.2: l*F2/6 in
    entries (4,5)/(5,4) and the shear-force terms omitted in the dissertation.
    Restores fast Newton-Raphson convergence when bending and torsion couple."""

    symmetric_contact_tangent: bool = False
    """Add the K(y, x) coupling term of the rotated contact springs (Eq. 59)."""

    update_geometry_after_convergence: bool = False
    """Apply the last converged increment to the element geometry (step 4.1)
    before the next time step, so element geometry and displacements agree."""

    @classmethod
    def dissertation(cls) -> Corrections:
        """Implementation exactly as used for the results of the dissertation."""
        return cls(False, False, False, False)

    @classmethod
    def all(cls) -> Corrections:
        return cls(True, True, True, True)

    @classmethod
    def preset(cls, name: str) -> Corrections:
        """'default', 'dissertation' or 'all' (used by the command line)."""
        presets = {"default": cls, "dissertation": cls.dissertation, "all": cls.all}
        if name not in presets:
            raise ValueError(f"unknown preset {name!r}; choose one of {list(presets)}")
        return presets[name]()
