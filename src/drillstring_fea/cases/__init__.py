"""Ready-made analysis cases. Each function returns ``(model, settings)``.

==========================  ==========================================  ==========
name                        description                                 section
==========================  ==========================================  ==========
appendix_a_cantilever       1-element cantilever, iteration by iteration  App. A
pure_bending                cantilever rolled into a circle              6.1
column_buckling             post-buckling of a slender column            6.2
composite_column            eccentrically loaded composite column        6.3
curved_cantilever           45-degree curved cantilever, 3D bending      (Aguiar 2013)
tubing_in_well              100 m tubing buckling inside a well          6.4
rotating_drillstring        4000 m rotating drill string                 6.5
==========================  ==========================================  ==========
"""

from .benchmarks import (
    appendix_a_cantilever,
    column_buckling,
    composite_column,
    curved_cantilever,
    pure_bending,
)
from .drillstrings import rotating_drillstring, tubing_in_well

CASES = {
    "appendix_a_cantilever": appendix_a_cantilever,
    "pure_bending": pure_bending,
    "column_buckling": column_buckling,
    "composite_column": composite_column,
    "curved_cantilever": curved_cantilever,
    "tubing_in_well": tubing_in_well,
    "rotating_drillstring": rotating_drillstring,
}

__all__ = ["CASES", *CASES]
