"""Building a drill string (or tubing) from pipe sections (Chapter 2, Table 8).

A string is a list of ``PipeSection`` from top to bottom (drill pipes, HWDP,
drill collars, bit). ``build_string`` creates the nodes along -Z (node 0 at the
top, z = 0), the elements and the per-element properties, and computes the
radial gap between each pipe and the well wall (Eq. 53: gap = r_w - r_p).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Oilfield units -> SI
INCH = 0.0254  # m
FOOT = 0.3048  # m
PSI = 6894.757293168361  # Pa
LBM_PER_FT3 = 0.45359237 / FOOT**3  # kg/m^3


@dataclass(frozen=True)
class PipeSection:
    """A run of identical joints, in SI units."""

    od: float  # outer diameter
    id: float  # inner diameter
    E: float  # Young's modulus
    n_joints: float  # number of joints
    joint_length: float  # length of one joint
    density: float  # steel density
    n_elements: int  # number of finite elements for the whole section

    @classmethod
    def from_oilfield_units(cls, od_in, id_in, e_psi, n_joints, joint_ft, density_lbft3,
                            n_elements):
        """Same data in inches, psi, feet and lbm/ft^3."""
        return cls(od_in * INCH, id_in * INCH, e_psi * PSI, n_joints, joint_ft * FOOT,
                   density_lbft3 * LBM_PER_FT3, int(n_elements))

    @property
    def length(self) -> float:
        return self.n_joints * self.joint_length

    @property
    def area(self) -> float:
        return np.pi * (self.od**2 - self.id**2) / 4

    @property
    def inertia(self) -> float:
        return np.pi * (self.od**4 - self.id**4) / 64


@dataclass
class PipeString:
    """Nodes, elements and per-element properties of a discretized string."""

    coords: np.ndarray
    connect: np.ndarray
    E: np.ndarray
    A: np.ndarray
    I: np.ndarray
    rho: np.ndarray
    Ae: np.ndarray  # outer area (buoyancy)
    Ai: np.ndarray  # inner area (internal fluid)
    pipe_diam: np.ndarray
    node_gaps: np.ndarray  # radial gap at each node (gap of the element below it)

    @property
    def bottom_node(self) -> int:
        return len(self.coords) - 1


def build_string(sections: list[PipeSection], well_diam: float) -> PipeString:
    """Discretize the sections, from the top (z = 0) downwards."""
    z = [0.0]
    for sec in sections:
        dz = sec.length / sec.n_elements
        z += list(z[-1] - dz * np.arange(1, sec.n_elements + 1))
    coords = np.zeros((len(z), 3))
    coords[:, 2] = z
    n_el = len(z) - 1

    def per_element(value_of):  # repeat a section value for each of its elements
        return np.concatenate([np.full(s.n_elements, value_of(s)) for s in sections])

    gaps = per_element(lambda s: (well_diam - s.od) / 2)
    return PipeString(
        coords=coords,
        connect=np.column_stack([np.arange(n_el), np.arange(1, n_el + 1)]),
        E=per_element(lambda s: s.E),
        A=per_element(lambda s: s.area),
        I=per_element(lambda s: s.inertia),
        rho=per_element(lambda s: s.density),
        Ae=per_element(lambda s: np.pi * s.od**2 / 4),
        Ai=per_element(lambda s: np.pi * s.id**2 / 4),
        pipe_diam=per_element(lambda s: s.od),
        node_gaps=np.append(gaps, gaps[-1]),
    )


def sinusoidal_imperfection(z, amplitude: float, half_wavelength: float, n_waves: int = 1):
    """Initial warping x(z) = amplitude * sin(n pi z / half_wavelength) (Eq. 62).

    Eq. 62 uses amplitude = d / (k L) and half_wavelength = L (string length).
    """
    return amplitude * np.sin(n_waves * np.pi * np.asarray(z) / half_wavelength)
