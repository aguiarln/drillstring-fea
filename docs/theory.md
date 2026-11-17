# Theory, implementation notes and validation

Companion of the dissertation for readers of the code. Equation, section,
table and figure numbers refer to the dissertation.

1. [From the equations to the code](#1-from-the-equations-to-the-code)
2. [Implementation notes](#2-implementation-notes): where the implementation of
   the published results differs from the theory (the `Corrections` options)
3. [Validation](#3-validation): results against the dissertation and the literature

## 1. From the equations to the code

This part maps the formulation of the dissertation (Chapters 4-6) to the
functions of the package. Equation numbers refer to the dissertation.

### Suggested reading order

1. `model.py`: what defines an analysis (nodes, elements, supports, loads,
   contact, settings) and the DOF numbering.
2. `loads.py`: how loads and imposed displacements evolve along the steps.
3. `solver.py`: the loop of Section 4.7, numbered as in the dissertation
   (steps 2-5). It calls the element and the contact routines below.
4. `beam.py`: rotations (Section 4.3), element matrices (Section 4.4) and the
   corotational update of each iteration (steps 4.1-4.4).
5. `contact.py`: the well-wall and rock springs (Section 4.8).
6. `cases/`: the examples of Chapter 6, built from the pieces above.

A good first exercise is to follow Appendix A in the code: run
`tests/test_dissertation.py::test_appendix_a_iteration_by_iteration` (or
`solve(model, settings, Corrections.dissertation(), on_iteration=print)` with
`cases.appendix_a_cantilever()`) and compare each iteration with Sections A.2-A.4.

### Unknowns and conventions

* 6 DOFs per node (Eq. 12): translations `DX DY DZ`, rotations `RX RY RZ`.
  DOF `d` of node `n` is entry `6 n + d` of the global vectors.
* Element DOFs: `[u1 v1 w1 rx1 ry1 rz1 u2 v2 w2 rx2 ry2 rz2]`.
* Frames (element and nodal triads) are 3x3 matrices stored **by rows**:
  row 0 = local x axis, row 1 = y, row 2 = z, written in global components.
* Assumptions (Section 4.1): two-node element, large displacements and
  rotations, small strains, Euler-Bernoulli, no warping, circular tube
  (Iy = Iz = I, J = 2 I).
* Element end forces (Eq. 45) are stored in element axes; the axial force of
  `Result.section_forces` is **positive in compression** (an element in
  tension has N1 < 0 at its first node).
* Drill strings hang along -Z from the top node (z = 0); gravity acts along -Z.

### Element (`beam.py`)

| Dissertation | Code |
|---|---|
| Rodrigues rotation matrix, Eqs. 4-7 | `rodrigues`, `skew` |
| Initial frames (X along the element, Y = e_z × X, Z = X × Y) | `initial_frames` |
| Linear stiffness K_L, Eq. 15 | `linear_stiffness` |
| Geometric stiffness K_NL, Eq. 16 (closed form: Aguiar 2013, Eq. B.2) | `geometric_stiffness` |
| Lumped mass, Eq. 18 (Mourelle 1993, Eq. IV.50) | `lumped_mass` |
| R_G = diag(R_e, R_e, R_e, R_e), Eq. 47 | `block_rotation` |
| Step 4.1: nodal positions, chord, deformed length | `Beams.update` |
| Step 4.2.1: R_n = R_inc R_n, Eq. 41 | `Beams.update` |
| Step 4.2.2: element frame, Eq. 42 | `Beams.update` |
| Step 4.3: relative rotations (Crisfield), Eq. 43; u_r, Eq. 44 | `relative_rotations`, `Beams.update` |
| Step 4.4: f_int = K_L u_r, Eq. 45; global f = R_G^T f_int, Eq. 46 | `Beams.update`, `Beams.internal_force_vector` |
| Element tangent R_G^T (K_L + K_NL) R_G | `Beams.tangent_stiffness` |

### Global solution (`solver.py`)

| Dissertation | Code |
|---|---|
| Newmark constants a0 … a7, Eq. 31 and Eq. 33 | `newmark_constants` |
| Rayleigh damping D = αM + βK, Eq. 20 | `D = alpha * M + beta * K_t` |
| Effective stiffness, Eqs. 28/48 | `K_ef = K_t + a0 M + a1 D` |
| Effective force, Eqs. 29/49-50 | `F_ef = R + M G + D E` |
| Residual, Eq. 37; solution, Eq. 51; update, Eq. 52 | `residual`, `spsolve`, `U = U + du` |
| Convergence, Eqs. 34-35 | `disp_err`, `force_err` |
| Velocities and accelerations, Eqs. 32-33 | after the iteration loop |

Prescribed displacements are imposed in the first iteration of each step
(`du[fixed] = target - U[fixed]`); the reduced system is
`K_ff du_f = R_f - K_fs du_s`.

### Contact (`contact.py`)

| Dissertation | Code |
|---|---|
| gap = r_w - r_p, Eq. 53 | `pipes.build_string` (`node_gaps`) |
| u_r = T^T U_G, Eqs. 54-56 | `u1`, `u2` |
| f = k_s (1 - gap/|u|) u, K_r = diag(k_s), Eqs. 57-58 | `f1`, `f2`, `k1`, `k2` |
| F = T f, K = T K_r T^T, Eq. 59 | `kxx`, `kyy`, `kxy` |
| n_r rotated spring pairs, Figures 22-24 | `Model.n_spring_directions` |
| Bit/rock spring, Eq. 60 | bit contact block |

### Loads and drill strings

| Dissertation | Code |
|---|---|
| load steps and time steps, Section 4.7 | `loads.StepInfo`, `loads.load_factor` |
| self-weight, buoyancy and internal fluid | `Model.self_weight` |
| geometric imperfection, Eq. 62 | `pipes.sinusoidal_imperfection` |
| drill string components and design (Ch. 2, Table 8) | `pipes.PipeSection`, `pipes.build_string` |

### Buckling and examples (Chapters 5 and 6)

| Dissertation | Code |
|---|---|
| geometric imperfection, Section 5.1, Eq. 62 | `pipes.sinusoidal_imperfection`, `tubing_in_well(n_waves, k)` |
| critical loads of Table 1 (Lubinski, Wu) and helix pitch, Eq. 61 | README, "Helical buckling" |
| 6.1: end moment M = 2π EI / L, Eqs. 63-64; exact tip displacements, Eqs. 65-66; error, Eq. 67 | `cases.pure_bending`, `tests/test_dissertation.py` |
| 6.2: P_crit = π² EI / (4 L²), lateral force Q = 0.1 % P_crit | `cases.column_buckling` |
| 6.3: composite column, Figure 32 | `cases.composite_column` |
| 6.4: Tables 3-4; bottom moment M_z = π G J / L, Eq. 68 | `cases.tubing_in_well` |
| 6.5: Table 8; WOB and rotation histories, Figure 41 | `cases.rotating_drillstring`, `loads` ("hold", "start") |
| Appendix A: one-element cantilever | `cases.appendix_a_cantilever` |

## 2. Implementation notes

This part explains the points where the implementation used for the
dissertation results differs from the underlying theory, checked against the
references of the dissertation:

* Aguiar, L. L. (2013), *A Three-Dimensional Pipe Beam Finite Element for
  Nonlinear Analysis of Multilayered Risers and Pipelines*, D.Sc. thesis, PUC-Rio
  (dissertation ref. [6]);
* Mourelle, M. M. (1993), *Análise Dinâmica de Sistemas Estruturais Constituídos
  por Linhas Marítimas*, D.Sc. thesis, COPPE/UFRJ (dissertation ref. [7]).

Items 2.1-2.4 are switched with `Corrections` (`--corrections` in the command line).
Items 2.1 and 2.2 are **on by default**; `Corrections.dissertation()` switches all
of them off and reproduces the published results exactly.

### 2.1 Initial element frames — `orthonormal_initial_triads`

**Theory.** The nodal reference systems are rigid, orthonormal frames
(Mourelle 1993, Sec. III.6; Aguiar 2013, Sec. 2 and Figure 1).

**Implementation.** The initial frame uses Y = e_z × X and Z = X × Y without
normalizing them. For an element inclined by a small angle φ from the
vertical, |Y| = |Z| = sin φ. The nodal frames start equal to this frame and
are only rotated afterwards, so they keep the wrong length. Eq. 43 then
measures about half of the bending rotations and almost none of the torsion.

**Who is affected.** Elements that are neither vertical nor perpendicular to Z:
every model with an initial sinusoidal imperfection (Sections 6.4 cases 1-2 and 6.5).

**Evidence** (`tests/test_beam.py`, Part 3):

* vertical cantilever with a 1 µm imperfection: tip deflection 2× the beam
  theory value; under a tip torque Newton-Raphson does not converge;
* 6.4 case 1, step 525: 80.1 kN without the correction, 87.2 kN with it;
  the published value is 86.7 kN;
* 6.5, first step of rotation: 45 iterations without, 3 with the correction.

### 2.2 Geometric stiffness — `full_geometric_stiffness`

**Theory.** Aguiar (2013), Eq. B.2, gives the closed-form K_NL.

**Implementation.** Two differences: entries (4,5)/(5,4) are `I F2/6`
instead of `l F2/6`, and the terms ±F2, ±F3 that couple the node-1
translations with the node-2 rotations are omitted.

**Effect.** The tangent matrix is not the derivative of the internal forces
when bending and torsion are coupled, so Newton-Raphson converges slowly.
For the 45° curved cantilever (a standard 3D benchmark) the analysis stops at
λ = 3.6 without the correction; with it, it converges in 5-7 iterations per
step and the tip positions agree with Crisfield within 0.3 %. In dynamic
analyses the tangent also enters the Rayleigh damping (D = αM + βK_t), so
results change slightly.

### 2.3 Contact tangent — `symmetric_contact_tangent`

**Theory.** Eq. 59: K = T K_r T^T is a full symmetric 2x2 matrix.

**Implementation.** The term K(y, x) is not added (only K(x, y)).

**Effect.** Only the tangent (and, in dynamics, the damping) changes.

### 2.4 Last increment of each step — `update_geometry_after_convergence`

**Implementation.** The increment of the converged iteration is added to the
displacement vector U but is not applied to the element geometry before the
next step (step 4.1 of the next iteration uses a zero increment). Element
geometry and U therefore differ by the accumulated last increments, which
are small because they are below the convergence tolerance.

### 2.5 Torsional lumped mass (parameter `torsional_inertia_factor`)

Eq. 18 copies Mourelle (1993), Eq. IV.50: torsional term
(ρAL/2)·I_x/(420 A). The lumping rule described in the same references
(sum of the off-diagonal terms of the consistent matrix with the same unit as
the diagonal term) applied to the consistent mass of Aguiar (2013), Eq. B.3
(torsion: 2I/3A on the diagonal, I/3A off the diagonal) gives
(ρAL/2)·I_x/A, i.e. 420 times larger. The bending terms (L²/210) agree with
that rule. The factor is a model parameter (default 1/420, as published);
`torsional_inertia_factor=1` gives the lumped value from the rule and
`0` removes the torsional inertia (runs "N.I." of Section 6.5).

### 2.6 Other points kept as published

* **Convergence test (Eqs. 34-35):** `min(absolute, relative)` norms over
  vectors that mix translations and rotations (and forces and moments).
  Mourelle (1993, Sec. IV.8) uses relative norms only, computed separately for
  translations and rotations, with ε_F = 10 ε_D.
* **Rayleigh coefficients:** α = 1.1336, β = 2.9087e-4 are fixed numbers;
  the dissertation (Eq. 20) and Aguiar (2013, Eq. 45) derive them from damping
  ratios of two modes. They are model parameters (`rayleigh_alpha`, `rayleigh_beta`).
* **Total rotations in U** are the algebraic sum of the increments (the
  element frames themselves are updated exactly with Eq. 41). Acknowledged in
  Chapter 7 of the dissertation.
* **Imperfection of Section 6.5:** x = (d/1000) sin(π z / l_joint) uses the
  length of one drill-pipe joint instead of the string length of Eq. 62.
* **Contact** measures displacements from the initial node positions, assumed
  on the well axis (the imperfection amplitudes are < 0.15 mm).
* **Bit/rock spring (Eq. 60)** is activated by |U_z| > z_gap, so it would also
  react if the bit moved *up* by more than the gap. In the published cases
  the bit only moves down (6.4, case 1), and the 6.5 string has no rock spring
  (the weight on bit is an applied force).
* **Wall/rock stiffnesses** follow Table 4 (6.4: 10⁴ and 10¹¹ N/m) and Table 8
  (6.5: 10⁷ N/m).

## 3. Validation

Results of this program (Python 3.14.3, NumPy 2.5.3, SciPy 1.18.1, Windows 11,
2026-11-17), with the reference values in brackets. They are checked by the
tests: `pytest` (fast) and `pytest -m slow` (Sections 6.4 and 6.5).
"dissertation" = `Corrections.dissertation()` (implementation of the published
results); "default" = `Corrections()` (orthonormal initial frames and full
geometric stiffness). Appendix A, 6.1 and 6.2 use "dissertation".

### Appendix A — residual norm per iteration

```
iteration 1: 10000  [10000]
iteration 2: 393.47  [393.47]
iteration 3: 0.1253  [0.125]
```

Internal forces, increments and both convergence measures of every iteration
are also checked in `tests/test_dissertation.py`.

### Section 6.1 — Table 2 (relative errors, %)

```
 5 elements: Ux 0.3209 [0.3209]  Uz 1.1901 [1.1902]
10 elements: Ux 0.0802 [0.0802]  Uz 0.2945 [0.2945]
15 elements: Ux 0.0362 [0.0362]  Uz 0.1307 [0.1307]
20 elements: Ux 0.0210 [0.021]   Uz 0.0735 [0.0735]
```

Eq. 67 is evaluated with |·| = Euclidean norm of the tip displacements over
the 20 load steps (this is the only aggregation that reproduces the table).

### Section 6.2 — post-buckling vs the exact elastica

The exact elastica (complete elliptic integrals) is used as reference, with
the load raised to 3 P_crit (`load_ratio=3`).

```
P/Pcr = 1.10: x 0.5105 [0.5135]  z 0.1808 [0.1837]
P/Pcr = 1.50: x 0.7891 [0.7886]  z 0.6338 [0.6364]
P/Pcr = 2.00: x 0.7984 [0.7968]  z 0.9279 [0.9302]
P/Pcr = 3.00: x 0.7098 [0.7074]  z 1.2018 [1.2041]
```

### 45° curved cantilever — tip position (Crisfield, from Aguiar 2013, Table 3)

```
dissertation: no convergence at step 10, iterations per step [6, 7, 7, 8, 13, 17, 25, 36, 58]
default: analysis finished, iterations per step [5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, ..., 7]
   step 10: [22.09, 58.41, 40.65]  [22.16, 58.53, 40.53]
   step 15: [18.36, 51.82, 48.85]  [18.43, 51.93, 48.79]
   step 20: [15.57, 46.74, 53.73]  [15.61, 46.84, 53.71]
```

### Section 6.4, case 1 — bottom force (Table 5: 38.2 kN at step 229, 86.7 kN at step 525)

```
dissertation         38.32 kN  80.06 kN  (4.5 s, 1383 iterations)
default              38.33 kN  87.22 kN  (4.1 s, 1293 iterations)
all corrections      38.33 kN  87.22 kN  (4.4 s, 1292 iterations)
```

The top displacement at step 229 is -0.01095 m, as in Table 5. With the
original frames ("dissertation") the value at step 525 is sensitive to round-off (80.06 kN
here, 79.98 kN with a different summation order of the sparse matrices).

### Section 6.5 — step 1760 (WOB 80 kN, neutral point 92.0 m above the bit, Figure 45)

```
dissertation         WOB 80.02 kN, neutral point 92.16 m, iterations at step 1761: 45  (33.3 s)
default              WOB 80.02 kN, neutral point 92.16 m, iterations at step 1761: 3  (36.1 s)
```

### Frame correction, isolated

10-element vertical cantilever, tip loads 1 N and 1 N·m, imperfection 1 µm
(`tests/test_beam.py::test_slightly_inclined_column_needs_orthonormal_frames`):

| | tip deflection | tip twist |
|---|---|---|
| beam theory | 1.8290e-4 m | 7.1330e-6 rad |
| dissertation frames | 3.6437e-4 m (2×) | no convergence |
| orthonormal frames (default) | 1.8290e-4 m | 7.1330e-6 rad |
