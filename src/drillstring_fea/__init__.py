"""drillstring_fea - nonlinear dynamic finite element analysis of oil-well drill strings.

Implementation of the formulation of the MSc dissertation "Nonlinear Dynamic
Behavior of Oil-Well Drill Strings" (L. P. de Aguiar Neto, PUC-Rio, 2021,
https://doi.org/10.17771/PUCRio.acad.56800).

Typical use::

    from drillstring_fea import cases, solve
    model, settings = cases.pure_bending(n_elements=10)
    result = solve(model, settings)
    print(result.displacements()[-1])          # tip displacements at the last step
"""

from . import cases
from .model import (
    DOF_NAMES,
    DX,
    DY,
    DZ,
    RX,
    RY,
    RZ,
    Corrections,
    Model,
    NodalLoad,
    PrescribedDisplacement,
    Settings,
    clamp,
    tube_section,
)
from .results import Result
from .solver import solve

__version__ = "1.0.0"

__all__ = [
    "DOF_NAMES", "DX", "DY", "DZ", "RX", "RY", "RZ",
    "Corrections", "Model", "NodalLoad", "PrescribedDisplacement", "Result", "Settings",
    "cases", "clamp", "solve", "tube_section",
]
