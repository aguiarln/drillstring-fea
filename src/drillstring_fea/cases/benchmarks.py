"""Verification examples for large displacements and rotations (Chapter 6 and Appendix A)."""

from __future__ import annotations

import numpy as np

from ..model import DX, DZ, RY, Model, NodalLoad, Settings, clamp, shear_modulus, tube_section


def _chain(n_nodes: int) -> np.ndarray:
    """Connectivity of a straight sequence of elements 0-1, 1-2, ..."""
    return np.column_stack([np.arange(n_nodes - 1), np.arange(1, n_nodes)])


def appendix_a_cantilever():
    """Appendix A: one-element cantilever under a 10 kN tip load (converges in 3 iterations)."""
    E, I, A = 1.25e8, 0.005, 0.08 * np.pi
    model = Model(
        name="appendix_a_cantilever",
        coords=[[0, 0, 0], [1, 0, 0]],
        connect=[[0, 1]],
        E=E, G=shear_modulus(E), A=A, I=I, rho=0.0,
        supports=clamp(0),
        loads=[NodalLoad(1, DZ, -1.0e4)],
        description="Appendix A - cantilever beam under point load",
    )
    return model, Settings(n_static_steps=1)


def pure_bending(n_elements: int = 10):
    """Section 6.1: cantilever rolled into a full circle by an end moment M = 2 pi EI / L."""
    L = 5.0
    section = tube_section(0.12, 0.14)
    E = 207e9
    coords = np.zeros((n_elements + 1, 3))
    coords[:, 0] = np.linspace(0.0, L, n_elements + 1)
    moment = 2 * np.pi * E * section["I"] / L  # Eq. 64
    model = Model(
        name="pure_bending",
        coords=coords,
        connect=_chain(n_elements + 1),
        E=E, G=shear_modulus(E), rho=100.0, **section,
        supports=clamp(0),
        loads=[NodalLoad(n_elements, RY, -moment)],  # applied in 20 equal steps
        description="Section 6.1 - straight cantilever beam under constant bending",
        info={"length": L, "moment": moment},
    )
    return model, Settings(n_static_steps=20)


def column_buckling(n_elements: int = 10, load_ratio: float = 1.0):
    """Section 6.2: slender cantilever column under compression P = load_ratio * P_crit.

    P_crit = pi^2 EI / (4 L^2); a lateral load Q = 0.1 % P_crit triggers buckling.
    The dissertation applies P_crit (load_ratio = 1) in 400 steps; use
    load_ratio > 1 to follow the post-buckling path.
    """
    L = 100.0
    section = tube_section(8.4e-3 / 2, 25.4e-3 / 2)
    E = 7.07e3 / section["I"]  # EI = 7070 N m^2
    p_crit = np.pi**2 * E * section["I"] / (4 * L**2)
    coords = np.zeros((n_elements + 1, 3))
    coords[:, 2] = np.linspace(0.0, L, n_elements + 1)
    model = Model(
        name="column_buckling",
        coords=coords,
        connect=_chain(n_elements + 1),
        E=E, G=shear_modulus(E), A=section["A"], I=section["I"], rho=0.0,
        supports=clamp(0),
        loads=[
            NodalLoad(n_elements, DX, 1e-3 * p_crit, "static+dynamic"),
            NodalLoad(n_elements, DZ, -load_ratio * p_crit),
        ],
        description="Section 6.2 - slender column subjected to compressive load",
        info={"length": L, "p_crit": p_crit, "load_ratio": load_ratio},
    )
    return model, Settings(n_static_steps=400)


def composite_column():
    """Section 6.3: composite column under eccentric axial load (Nunes et al. 2003).

    22 elements (Figure 32): ten in each vertical segment, one horizontal
    element joining the segments at mid-height (offset 0.25 m) and one at the
    top leading to the load point (offset 0.50 m). Load applied in 100 steps.
    """
    z = np.arange(0.0, 10.5, 0.5)
    coords = np.array([[0.0, 0.0, zz] for zz in z[:11]]  # lower segment
                      + [[0.25, 0.0, zz] for zz in z[10:]]  # upper segment (offset 0.25)
                      + [[0.50, 0.0, 10.0]])  # load application point (eccentricity)
    section = tube_section(0.052 / 2, 0.3425 / 2)
    E = 10e6
    model = Model(
        name="composite_column",
        coords=coords,
        connect=_chain(len(coords)),
        E=E, G=shear_modulus(E), A=section["A"], I=section["I"], rho=0.0,
        supports=clamp(0),
        loads=[NodalLoad(len(coords) - 1, DZ, -1000.0)],  # applied in 100 steps
        description="Section 6.3 - composite column subjected to eccentric load",
    )
    return model, Settings(n_static_steps=100)


def curved_cantilever(n_elements: int = 8, load_ratio: float = 7.2):
    """45-degree curved cantilever with an out-of-plane tip load (Bathe & Bolourchi 1979).

    Radius 100 in, E = 1e7 psi, A = 1 in^2, I = 1/12 in^4 (consistent units).
    Load P = load_ratio * EI / R^2 in 20 steps. Reference tip positions:
    Aguiar (2013), Table 3.
    """
    R, angle = 100.0, np.pi / 4
    phi = np.linspace(0.0, angle, n_elements + 1)
    coords = np.column_stack([R - R * np.cos(phi), R * np.sin(phi), np.zeros_like(phi)])
    section = tube_section(0.173 / 2, 1.142 / 2)  # A ~ 1, I ~ 1/12
    E = 1e7
    model = Model(
        name="curved_cantilever",
        coords=coords,
        connect=_chain(n_elements + 1),
        E=E, G=shear_modulus(E), A=section["A"], I=section["I"], rho=0.0,
        supports=clamp(0),
        loads=[NodalLoad(n_elements, DZ, load_ratio * E * section["I"] / R**2)],
        description="45-degree curved cantilever under out-of-plane load",
        info={"radius": R, "load_ratio": load_ratio},
    )
    return model, Settings(n_static_steps=20)
