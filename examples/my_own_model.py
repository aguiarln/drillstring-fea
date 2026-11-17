"""Building a model from scratch: a 50 m pipe in a well, compressed from the bottom.

    python examples/my_own_model.py
"""

import numpy as np

from drillstring_fea import DX, DY, DZ, Model, NodalLoad, Settings, clamp, solve, tube_section

# Geometry: 51 nodes along -Z, from z = 0 (top) to z = -50 m (bottom)
n_elements = 50
coords = np.zeros((n_elements + 1, 3))
coords[:, 2] = -np.linspace(0.0, 50.0, n_elements + 1)
connect = np.column_stack([np.arange(n_elements), np.arange(1, n_elements + 1)])

section = tube_section(inner_radius=0.038, outer_radius=0.0445)  # A, I, Ae, Ai
top, bottom = 0, n_elements

model = Model(
    name="compressed_pipe",
    coords=coords,
    connect=connect,
    E=2.07e11, G=2.07e11 / 2.6, rho=7850.0, **section,  # steel, G = E / (2 (1 + 0.3))
    # top clamped; bottom held laterally (it can still move axially and rotate)
    supports=clamp(top) + clamp(bottom, (DX, DY)),
    loads=[
        NodalLoad(bottom, DZ, 2.0e3, "linear"),  # upward force growing 2 kN/s (dynamic phase)
        NodalLoad(n_elements // 2, DX, 1.0, "constant"),  # tiny lateral force: triggers buckling
    ],
    # contact with the well wall: 13 mm radial gap at every node
    wall_nodes=np.arange(n_elements + 1), wall_gaps=np.full(n_elements + 1, 0.013),
    wall_stiffness=1e4,
)

# 5 static steps (self-weight), then 20 s of dynamic analysis with dt = 0.01 s
settings = Settings(n_static_steps=5, dynamic=True, t_final=20.0, dt=0.01, save_every=10)
result = solve(model, settings)
print(result.summary())

N_bottom = result.section_forces()[-1, 0]  # axial force, positive in compression
max_lateral = np.max(np.hypot(result.displacements()[:, DX], result.displacements()[:, DY]))
print(f"compression at the bottom: {N_bottom / 1e3:.2f} kN, max lateral displacement: "
      f"{max_lateral * 1e3:.1f} mm")
result.save("results/compressed_pipe.npz")
result.step_to_csv("results/compressed_pipe_last_step.csv")
