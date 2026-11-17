"""Cases catalogue, load histories, result files and command line."""

import numpy as np
import pytest

from drillstring_fea import Result, Settings, cases, solve
from drillstring_fea.cases import CASES
from drillstring_fea.cli import main
from drillstring_fea.loads import StepInfo, load_factor


@pytest.mark.parametrize("name", list(CASES))
def test_every_case_builds(name):
    model, settings = CASES[name]()
    assert model.n_elements > 0 and settings.n_steps > 0


def test_rotating_drillstring_matches_table_8():
    model, settings = cases.rotating_drillstring(stabilizers=(15.0, 30.0, 45.0))
    assert (model.n_elements, model.n_nodes) == (630, 631)
    assert -model.coords[-1, 2] == pytest.approx(3812.0 + 169.06 + 18.56 + 0.34, abs=0.05)
    assert model.info["wob"] == pytest.approx(80.5e3)
    assert settings.n_steps == 3760
    assert [n + 1 for n in model.info["stabilizer_nodes"]] == [590, 569, 552]  # 1-based


def test_tubing_case_4_moment_is_eq_68():
    model, _ = cases.tubing_in_well(case=4)
    assert model.loads[-1].value == pytest.approx(7.14e3, rel=1e-3)


def test_hold_and_start_histories_switch_at_step_1760():
    def info(step):
        return StepInfo(step, 150, 0.01, (step - 150) * 0.01, 1.0, 1.0, True)

    assert load_factor("hold", 16.1, info(1760)) == pytest.approx(16.1)
    assert load_factor("hold", 16.1, info(3000)) == 16.1
    assert load_factor("start", 16.1, info(1760)) == 0.0
    assert load_factor("start", 16.1, info(1761)) == pytest.approx(0.01)


def test_save_load_and_csv_export(tmp_path):
    model, settings = cases.pure_bending(n_elements=4)
    r = solve(model, settings)
    r2 = Result.load(r.save(tmp_path / "r.npz"))
    np.testing.assert_array_equal(r.U, r2.U)
    np.testing.assert_array_equal(r.forces, r2.forces)
    table = np.loadtxt(r2.step_to_csv(tmp_path / "s.csv", step=20), delimiter=",", skiprows=1)
    assert table.shape == (5, 16)
    hist = np.loadtxt(r2.history_to_csv(tmp_path / "h.csv", node=4, dof=2), delimiter=",",
                      skiprows=1)
    assert hist.shape == (20, 3)


def test_save_every():
    model, _ = cases.pure_bending(n_elements=4)
    r = solve(model, Settings(n_static_steps=10, save_every=4))
    assert r.steps.tolist() == [4, 8, 10]


def test_command_line_run_and_export(tmp_path, capsys):
    out = tmp_path / "pb.npz"
    assert main(["run", "pure_bending", "n_elements=4", "n_static_steps=5", "-o", str(out)]) == 0
    assert main(["info", str(out)]) == 0
    assert main(["run", "appendix_a_cantilever", "--corrections", "dissertation", "-o",
                 str(tmp_path / "a.npz")]) == 0
    assert main(["export", str(out), "--node", "4", "--dof", "DX", "-o",
                 str(tmp_path / "tip.csv")]) == 0
    assert "pure_bending" in capsys.readouterr().out
