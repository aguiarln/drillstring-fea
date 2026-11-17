"""Load histories: how loads and prescribed displacements evolve along the steps.

Every load (or prescribed displacement) is ``value * factor``, where ``factor``
depends on the chosen history and on the current step:

=================  =========================================================
history            factor
=================  =========================================================
"linear"           analyses with a dynamic phase: time t, so ``value`` is a
                   rate per second (0 during the static steps);
                   static analyses: k / n_static (load ratio 0 -> 1)
"constant"         1
"static"           ramps 0 -> 1 in the static phase, then stays at 1
"ramp"             0 in the static phase; t / ramp_time (max 1) in the dynamic phase
"static+dynamic"   "static" plus t in the dynamic phase
"hold"             t until ``t_switch``, then ``t_switch`` (e.g. WOB growing at
                   5 kN/s up to 16.1 s and then constant, Section 6.5)
"start"            0 until ``t_switch``, then (t - t_switch) (e.g. a rotation that
                   starts at 16.1 s with constant speed ``value`` rad/s)
"cosine", "sine"   cos(2 pi f t), sin(2 pi f t) with f = 3 Hz
=================  =========================================================

Time t starts at 0 in the first dynamic step: t = (step - n_static) * dt.
In static analyses "cosine" and "sine" also use the load ratio k / n_static
in place of t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

OSCILLATION_FREQUENCY = 3.0  # Hz, used by "cosine" and "sine"

HISTORIES = ("linear", "constant", "static", "ramp", "static+dynamic", "hold", "start",
             "cosine", "sine")


@dataclass(frozen=True)
class StepInfo:
    """Load variables of the current step (Section 4.7, steps 1-3)."""

    step: int  # 1, 2, ... (static steps first)
    n_static: int  # number of static steps
    dt: float  # time step of the dynamic phase
    time: float  # t in the dynamic phase (0 in the static phase)
    static_ratio: float  # k / n_static in the static phase, 1 afterwards
    ramp_ratio: float  # t / ramp_time (max 1) in the dynamic phase
    has_dynamic_phase: bool

    @property
    def is_dynamic(self) -> bool:
        return self.step > self.n_static

    @property
    def k_dynamic(self) -> int:
        """Index of the dynamic step (time = k_dynamic * dt); <= 0 while static."""
        return self.step - self.n_static


def load_factor(history: str, t_switch: float, s: StepInfo) -> float:
    """Multiplier of a load or prescribed displacement at step ``s``."""
    # "linear" follows time in dynamic analyses and the load ratio in static ones
    t = s.time if s.has_dynamic_phase else s.static_ratio

    if history == "linear":
        return t
    if history == "constant":
        return 1.0
    if history == "static":
        return s.static_ratio
    if history == "ramp":
        return s.ramp_ratio
    if history == "static+dynamic":
        return s.static_ratio + (s.time if s.is_dynamic else 0.0)
    if history == "hold":
        # integer step comparison avoids round-off in t == t_switch
        return t_switch if s.k_dynamic > round(t_switch / s.dt) else t
    if history == "start":
        k0 = round(t_switch / s.dt)
        return (s.k_dynamic - k0) * s.dt if s.k_dynamic > k0 else 0.0
    if history == "cosine":
        return math.cos(2.0 * math.pi * OSCILLATION_FREQUENCY * t)
    if history == "sine":
        return math.sin(2.0 * math.pi * OSCILLATION_FREQUENCY * t)
    raise ValueError(f"unknown history {history!r}; choose one of {HISTORIES}")
