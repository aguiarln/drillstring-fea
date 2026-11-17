"""Quick start: run a case, read the results and plot them.

    python examples/quickstart.py          (needs matplotlib: pip install -e ".[plot]")

Section 6.1 of the dissertation: a cantilever rolled into a full circle by an
end moment applied in 20 steps.
"""

import matplotlib.pyplot as plt
import numpy as np

from drillstring_fea import DX, DZ, cases, solve

# 1. Build the model and the analysis settings of a ready-made case.
model, settings = cases.pure_bending(n_elements=10)

# 2. Run the analysis.
result = solve(model, settings)
print(result.summary())

# 3. Read results.
tip = model.n_nodes - 1
print("tip displacement at the last step [DX DY DZ RX RY RZ]:", result.displacements()[tip])
ux = result.history(tip, DX)  # one value per saved step
uz = result.history(tip, DZ)

# Exact solution (Eqs. 65-66)
L = model.info["length"]
theta = 2 * np.pi * result.static_ratio
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(result.static_ratio, np.abs(ux), "o", label="|Ux| numerical")
ax1.plot(result.static_ratio, L * (1 - np.sin(theta) / theta), "-", label="Ux exact")
ax1.plot(result.static_ratio, np.abs(uz), "s", label="|Uz| numerical")
ax1.plot(result.static_ratio, L * (1 - np.cos(theta)) / theta, "-", label="Uz exact")
ax1.set_xlabel("M / M_max")
ax1.set_ylabel("tip displacement (m)")
ax1.legend()

# Deformed shapes (cf. Figure 26)
for step in (5, 10, 15, 20):
    u = result.displacements(step)
    ax2.plot(model.coords[:, 0] + u[:, DX], model.coords[:, 2] + u[:, DZ], "o-", ms=3,
             label=f"step {step}")
ax2.set_aspect("equal")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("z (m)")
ax2.legend()
plt.tight_layout()
plt.show()
