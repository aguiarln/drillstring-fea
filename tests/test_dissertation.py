"""Reproduction of results published in the dissertation and in the literature."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk

from drillstring_fea import Corrections, cases, solve

DISSERTATION = Corrections.dissertation()  # implementation used for the published results


def test_appendix_a_iteration_by_iteration():
    model, settings = cases.appendix_a_cantilever()
    log = []
    result = solve(model, settings, DISSERTATION, on_iteration=log.append)
    assert result.converged and result.iterations.tolist() == [3]
    # Iteration 1 (A.2): linear solution; the residual is the applied load
    assert np.isclose(np.linalg.norm(log[0].residual[6:]), 1e4)
    np.testing.assert_allclose(log[0].du[[8, 10]], [-0.0053, 0.0080], atol=5e-5)
    # Iteration 2 (A.3)
    np.testing.assert_allclose(log[1].internal_force[[6, 8, 10]], [393.46, -10002, 0.1896],
                               rtol=2e-4)
    assert np.isclose(np.linalg.norm(log[1].residual[6:]), 393.47, atol=0.01)
    assert np.isclose(log[1].disp_error, 1.25e-5, rtol=5e-3)
    # Iteration 3 (A.4): converged
    assert np.isclose(np.linalg.norm(log[2].residual[6:]), 0.125, rtol=3e-3)
    assert np.isclose(log[2].disp_error, 2.24e-7, rtol=5e-3)
    assert np.isclose(log[2].force_error, 1.25e-5, rtol=5e-3)


@pytest.mark.parametrize("n, err_x, err_z", [(5, 0.3209, 1.1902), (10, 0.0802, 0.2945),
                                             (15, 0.0362, 0.1307), (20, 0.0210, 0.0735)])
def test_section_6_1_table_2(n, err_x, err_z):
    """Eq. 67 with |.| = Euclidean norm of the tip displacements over the 20 steps."""
    model, settings = cases.pure_bending(n_elements=n)
    r = solve(model, settings, DISSERTATION)
    L = model.info["length"]
    theta = 2 * np.pi * r.static_ratio
    exact_x = L * (1 - np.sin(theta) / theta)  # Eq. 65
    exact_z = L * (1 - np.cos(theta)) / theta  # Eq. 66

    def error(exact, numeric):
        norm_exact = np.linalg.norm(exact)
        return 100 * abs(norm_exact - np.linalg.norm(np.abs(numeric))) / norm_exact

    assert error(exact_x, r.history(n, 0)) == pytest.approx(err_x, abs=1e-4)
    assert error(exact_z, r.history(n, 2)) == pytest.approx(err_z, abs=1.5e-4)


def test_section_6_2_post_buckling_follows_the_exact_elastica():
    model, settings = cases.column_buckling(load_ratio=3.0)
    r = solve(model, settings)
    L, n = model.info["length"], model.n_elements
    load_ratio = 3.0 * r.static_ratio
    for target in (1.2, 1.5, 2.0, 3.0):
        i = int(np.argmin(np.abs(load_ratio - target)))
        m = brentq(lambda m, p=load_ratio[i]: ellipk(m) - np.pi / 2 * np.sqrt(p), 0, 1 - 1e-15)
        K, Ek = ellipk(m), ellipe(m)
        assert r.U[i, 6 * n] / L == pytest.approx(2 * np.sqrt(m) / K, abs=4e-3)
        assert -r.U[i, 6 * n + 2] / L == pytest.approx(2 * (1 - Ek / K), abs=4e-3)


def test_curved_cantilever_against_literature():
    """Tip positions of Crisfield (1990), as listed in Aguiar (2013), Table 3."""
    model, settings = cases.curved_cantilever()
    r = solve(model, settings)  # default corrections
    tip = model.coords[-1] + r.U[:, 6 * model.n_elements: 6 * model.n_elements + 3]
    crisfield = {10: (22.16, 58.53, 40.53), 15: (18.43, 51.93, 48.79), 20: (15.61, 46.84, 53.71)}
    for step, xyz in crisfield.items():  # load factor 3.6, 5.4, 7.2
        np.testing.assert_allclose(tip[r.index(step)], xyz, rtol=0.01)
    assert r.iterations.max() <= 7


def test_curved_cantilever_needs_the_full_geometric_stiffness():
    """With the geometric stiffness of the dissertation, Newton-Raphson stalls."""
    model, settings = cases.curved_cantilever()
    assert not solve(model, settings, DISSERTATION).converged


@pytest.mark.slow
@pytest.mark.parametrize("corrections, f525", [
    (DISSERTATION, 80.06),
    (Corrections(), 87.22),
])
def test_section_6_4_case_1_bottom_force(corrections, f525):
    """Table 5 (n = 1, k = 10): 38.2 kN at step 229 and 86.7 kN at step 525."""
    model, settings = cases.tubing_in_well(case=1)
    settings.t_final = 5.14  # up to step 525
    r = solve(model, settings, corrections)
    assert r.displacements(229)[0, 2] == pytest.approx(-0.01095)
    assert abs(r.section_forces(229)[-1, 0]) / 1e3 == pytest.approx(38.2, rel=0.01)
    assert abs(r.section_forces(525)[-1, 0]) / 1e3 == pytest.approx(f525, abs=0.2)


@pytest.mark.slow
def test_section_6_5_weight_on_bit_and_neutral_point():
    """Figure 45: WOB 80 kN and neutral point 92.0 m above the bit at step 1760."""
    model, settings = cases.rotating_drillstring()
    settings.t_final = 16.1  # up to step 1761, the first step of the rotation
    r = solve(model, settings)
    z = model.coords[:, 2]
    N = r.section_forces(1760)[:, 0]  # positive in compression
    k = np.flatnonzero(np.diff(np.sign(N)) != 0)[-1]  # last change tension -> compression
    assert N[-1] / 1e3 == pytest.approx(80.0, abs=0.1)
    assert z[k] - z[-1] == pytest.approx(92.0, abs=0.5)
    assert r.iterations[-1] <= 5  # orthonormal frames: the rotation starts smoothly
