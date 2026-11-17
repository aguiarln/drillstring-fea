"""Tubing and drill-string examples inside a vertical well (Sections 6.4 and 6.5)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..model import (
    DX,
    DY,
    DZ,
    RX,
    RY,
    RZ,
    Model,
    NodalLoad,
    PrescribedDisplacement,
    Settings,
    clamp,
    shear_modulus,
)
from ..pipes import PipeSection, build_string, sinusoidal_imperfection


def _string_model(name, string, well_diam, coords, **kwargs) -> Model:
    """Model of a pipe string with wall contact springs at every node."""
    return Model(
        name=name,
        coords=coords,
        connect=string.connect,
        E=string.E, G=shear_modulus(string.E), A=string.A, I=string.I, rho=string.rho,
        Ae=string.Ae, Ai=string.Ai, pipe_diam=string.pipe_diam, well_diam=well_diam,
        wall_nodes=np.arange(len(coords)), wall_gaps=string.node_gaps,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Section 6.4 - tubing under compression inside a well
# --------------------------------------------------------------------------
def tubing_in_well(case: int = 1, n_waves: int = 1, k: float = 10.0, n_elements: int = 200):
    """Section 6.4: 100 m tubing compressed inside a well (Tables 3 and 4).

    case 1: initial sinusoidal warping; top pushed down at 5 mm/s; the bottom
            rests on the rock (vertical contact spring after a 2 mm gap).
    case 2: initial sinusoidal warping; upward bottom force growing at 5 kN/s.
    case 3: lateral force F_x = 1e-2 N at mid-length; bottom force 5 kN/s.
    case 4: case 3 plus a bottom moment M_z = pi G J / L (Eq. 68), simulating
            the bit/rock friction.

    Warping (Eq. 62): x(z) = d / (k L) * sin(n pi z / L), with n = ``n_waves``.
    In all cases the top is clamped (except the imposed axial displacement of
    case 1) and the bottom is held laterally.
    """
    if case not in (1, 2, 3, 4):
        raise ValueError("case must be 1, 2, 3 or 4")
    od, id_, length = 88.9e-3, 75.9968e-3, 100.0  # Table 4
    rho = 15.2 / (np.pi * (od**2 - id_**2) / 4)  # 15.2 kg/m -> 9095.87 kg/m^3 (Table 4)
    tube = PipeSection(od=od, id=id_, E=2.06843e11, n_joints=1, joint_length=length,
                       density=rho, n_elements=n_elements)
    well = od + 2 * 13.3e-3  # radial gap 13.3 mm (Table 4)
    string = build_string([tube], well)
    coords = string.coords.copy()
    if case in (1, 2):
        coords[:, 0] = sinusoidal_imperfection(coords[:, 2], od / (k * length), length, n_waves)
    top, bottom = 0, string.bottom_node

    supports = clamp(top, (DX, DY, RX, RY, RZ)) + clamp(bottom, (DX, DY))
    loads = []
    bit_nodes, bit_gaps = [], []
    if case == 1:
        supports.append(PrescribedDisplacement(top, DZ, -0.005, "linear"))  # 5 mm/s
        # z_gap of Table 4 (2 mm), with the digits of the input file of the dissertation
        bit_nodes, bit_gaps = [bottom], [0.00200562106531493085]
    else:
        supports.append(PrescribedDisplacement(top, DZ, 0.0))
        loads.append(NodalLoad(bottom, DZ, 5.0e3, "linear"))  # 5 kN/s
    if case in (3, 4):
        mid = len(coords) // 2
        loads.append(NodalLoad(mid, DX, 1e-2, "constant"))
    if case == 4:
        G = shear_modulus(tube.E)
        loads.append(NodalLoad(bottom, RZ, np.pi * G * 2 * tube.inertia / length, "static"))

    model = _string_model(
        "tubing_in_well", string, well, coords,
        supports=supports, loads=loads, bit_nodes=bit_nodes, bit_gaps=bit_gaps,
        wall_stiffness=1e4, rock_stiffness=1e11,  # Table 4
        description=f"Section 6.4 - tubing under compression inside well, case {case}",
        info={"case": case},
    )
    return model, Settings(n_static_steps=10, dynamic=True, dt=0.01, t_final=30.0)


# --------------------------------------------------------------------------
# Section 6.5 - long rotating drill string
# --------------------------------------------------------------------------
DRILLSTRING_SECTIONS = [  # Table 8: drill pipes, HWDP, drill collars and bit
    PipeSection(139.7e-3, 121.0e-3, 210e9, 372, 9.97916, 7860.0, 379),  # DP
    PipeSection(139.7e-3, 121.0e-3, 210e9, 10, 9.97916, 7860.0, 20),  # DP (finer mesh)
    PipeSection(139.7e-3, 82.0e-3, 210e9, 16, 10.56625, 7860.0, 180),  # HWDP
    PipeSection(165.0e-3, 82.0e-3, 210e9, 2, 9.28, 7860.0, 50),  # DC
    PipeSection(215.9e-3, 0.0, 210e9, 1, 0.34, 7860.0, 1),  # bit
]

STABILIZER_NODES = {15.0: 589, 30.0: 568, 45.0: 551}
"""Nodes used for stabilizers 15, 30 and 45 m above the bit in Section 6.5."""


def rotating_drillstring(
    rpm: float = 80.0,
    wob_rate: float = 5.0e3,
    t_rotation: float = 16.1,
    stabilizers: Sequence[float] = (),
    torsional_inertia_factor: float = 1.0 / 420.0,
    wall_stiffness: float = 1.0e7,
):
    """Section 6.5: 4000 m drill string, rotated at the top (Table 8).

    1. Static phase (150 steps): self-weight applied gradually.
    2. Dynamic phase: the weight on bit grows at ``wob_rate`` (N/s) until
       ``t_rotation`` (16.1 s -> 80.5 kN at step 1760), then stays constant
       while the top rotates at ``rpm``.

    The weight on bit is an upward force at the bit node, as in the input of
    the dissertation, where the bit/rock spring (Eq. 60) was switched off for
    this example; the wall springs (``wall_stiffness``, Table 8) act at every node.
    WOB = ``wob_rate * t_rotation``; e.g. ``t_rotation=30`` gives 150 kN and a
    helically buckled string (README, "Helical buckling").

    ``stabilizers``: heights above the bit (m) of lateral supports (Figs. 49-50).
    ``torsional_inertia_factor = 0`` removes the torsional inertia ("N.I." runs).
    The initial imperfection is x = (d/1000) sin(pi z / l_joint), with the length
    of one drill-pipe joint, as in the analyses of the dissertation.
    """
    well = 2 * 111.2e-3
    string = build_string(DRILLSTRING_SECTIONS, well)
    coords = string.coords.copy()
    dp = DRILLSTRING_SECTIONS[0]
    coords[:, 0] = sinusoidal_imperfection(coords[:, 2], dp.od / 1000, dp.joint_length)
    top, bottom = 0, string.bottom_node

    # top: clamped, except the rotation about Z, imposed from t_rotation on
    omega = rpm * 2 * np.pi / 60  # rad/s
    supports = clamp(top, (DX, DY, DZ, RX, RY))
    supports.append(PrescribedDisplacement(top, RZ, -omega, "start", t_rotation))
    nodes = []
    for h in stabilizers:
        node = STABILIZER_NODES.get(h)
        if node is None:  # any other height: closest node
            node = int(np.argmin(np.abs(coords[:, 2] - (coords[bottom, 2] + h))))
        nodes.append(node)
        supports += clamp(node, (DX, DY))

    model = _string_model(
        "rotating_drillstring", string, well, coords,
        supports=supports,
        loads=[NodalLoad(bottom, DZ, wob_rate, "hold", t_rotation)],
        self_weight_history="static",
        wall_stiffness=wall_stiffness,
        torsional_inertia_factor=torsional_inertia_factor,
        description="Section 6.5 - long rotating drill string",
        info={"rpm": rpm, "wob": wob_rate * t_rotation, "stabilizer_nodes": nodes},
    )
    # t_final = 36.09 s -> last step 3760, as in the figures of Section 6.5
    return model, Settings(n_static_steps=150, dynamic=True, dt=0.01, t_final=36.09)
