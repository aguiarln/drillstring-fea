"""Results of an analysis: storage, saving/loading (.npz) and CSV export.

What is stored for every saved step
-----------------------------------
* ``U``      (n_saved, n_dofs)          displacements/rotations of all DOFs;
* ``forces`` (n_saved, n_elements, 12)  element end forces in local axes,
  ``[N1 Vy1 Vz1 T1 My1 Mz1  N2 Vy2 Vz2 T2 My2 Mz2]`` (N axial, V shear,
  T torque, M bending moment), i.e. f_int of Eq. 45;
* ``steps``, ``time``, ``static_ratio``, ``iterations`` for each saved step.

Sign convention of the forces
-----------------------------
The forces are the end forces f_int = K_L u_r of each element (Eq. 45), in
the element axes (X along the element, from node 1 to node 2). At node 1 an
element in tension has N1 < 0 and at node 2 N2 > 0, so ``section_forces``
(node-1 values) gives **N > 0 in compression** (e.g. the weight on bit).

File format
-----------
``Result.save("file.npz")`` writes a NumPy ``.npz`` file: a zip archive with
one ``.npy`` array per field. It can be read without this package with
``numpy.load("file.npz")`` (the field names are the attribute names below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import DOF_NAMES, NDOF, Model, Settings


@dataclass
class Result:
    """Saved steps of an analysis (returned by ``solve``, read with ``Result.load``)."""

    model_name: str
    coords: np.ndarray  # initial nodal coordinates (n_nodes, 3)
    connect: np.ndarray  # element connectivity (n_elements, 2)
    steps: np.ndarray  # step numbers (1, 2, ...; static steps first)
    time: np.ndarray  # time of the dynamic phase (0 for static steps)
    static_ratio: np.ndarray  # static load ratio (1 after the static phase)
    iterations: np.ndarray  # Newton-Raphson iterations of each step
    U: np.ndarray  # (n_saved, 6 n_nodes) displacements; DOF d of node n is column 6 n + d
    forces: np.ndarray  # (n_saved, n_elements, 12) element end forces in element axes
    converged: bool = True
    message: str = ""
    elapsed: float = 0.0  # wall-clock time of the analysis (s)
    pipe_diam: np.ndarray | None = None  # outer diameter of each element (for drawings)
    well_diam: float = float("nan")  # well diameter (nan if there is no well)
    _count: int = field(default=0, repr=False)
    _save_every: int = field(default=1, repr=False)
    _n_steps: int = field(default=0, repr=False)

    # ------------------------------------------------------------- building
    @classmethod
    def empty(cls, model: Model, settings: Settings) -> Result:
        n_steps = settings.n_steps
        saved = sum(1 for k in range(1, n_steps + 1)
                    if k % settings.save_every == 0 or k == n_steps)
        return cls(
            model_name=model.name,
            coords=model.coords.copy(),
            connect=model.connect.copy(),
            steps=np.zeros(saved, int),
            time=np.zeros(saved),
            static_ratio=np.zeros(saved),
            iterations=np.zeros(saved, int),
            U=np.zeros((saved, model.n_dofs)),
            forces=np.zeros((saved, model.n_elements, 12)),
            pipe_diam=model.pipe_diam.copy(),
            well_diam=float("nan") if model.well_diam is None else model.well_diam,
            _save_every=settings.save_every,
            _n_steps=n_steps,
        )

    def store(self, step, time, static_ratio, iterations, U, forces) -> None:
        """Keep a converged step if it is a multiple of ``save_every`` (or the last one)."""
        if step % self._save_every == 0 or step == self._n_steps:
            i = self._count
            self.steps[i], self.time[i], self.static_ratio[i] = step, time, static_ratio
            self.iterations[i] = iterations
            self.U[i], self.forces[i] = U, forces
            self._count += 1

    def finish(self, converged: bool, message: str, elapsed: float) -> None:
        """Trim unused rows (if the analysis stopped early) and record the status."""
        k = self._count
        for name in ("steps", "time", "static_ratio", "iterations", "U", "forces"):
            setattr(self, name, getattr(self, name)[:k])
        self.converged, self.message, self.elapsed = converged, message, elapsed

    # ------------------------------------------------------------- reading
    @property
    def n_nodes(self) -> int:
        return self.coords.shape[0]

    def index(self, step: int) -> int:
        """Row of ``step`` in the stored arrays."""
        rows = np.flatnonzero(self.steps == step)
        if rows.size == 0:
            raise KeyError(f"step {step} was not saved (saved: {self.steps[0]}..{self.steps[-1]})")
        return int(rows[0])

    def displacements(self, step: int | None = None) -> np.ndarray:
        """(n_nodes, 6) displacements [DX DY DZ RX RY RZ] at ``step`` (default: last)."""
        i = -1 if step is None else self.index(step)
        return self.U[i].reshape(self.n_nodes, NDOF)

    def history(self, node: int, dof: int) -> np.ndarray:
        """Displacement of one node/DOF at every saved step."""
        return self.U[:, NDOF * node + dof]

    def section_forces(self, step: int | None = None) -> np.ndarray:
        """Internal forces at each node of a chain of elements, (n_nodes, 6) = [N Vy Vz T My Mz].

        Node k (k < n_elements) takes the node-1 end forces of element k; the
        last node takes the node-2 end forces of the last element with the sign
        changed, so that both follow the same convention (N > 0 in compression,
        see the module docstring). Valid for strings numbered 0-1, 1-2, 2-3, ...
        """
        i = -1 if step is None else self.index(step)
        f = self.forces[i]
        return np.vstack([f[:, :NDOF], -f[-1:, NDOF:]])

    # ------------------------------------------------------------- export
    def history_to_csv(self, path: str | Path, node: int, dof: int) -> Path:
        """CSV with columns step, time, value for one node/DOF."""
        data = np.column_stack([self.steps, self.time, self.history(node, dof)])
        header = f"step,time,{DOF_NAMES[dof]}_node{node}"
        return _write_csv(path, data, header)

    def step_to_csv(self, path: str | Path, step: int | None = None) -> Path:
        """CSV with one row per node: coordinates, displacements and section forces."""
        data = np.hstack([np.arange(self.n_nodes)[:, None], self.coords,
                          self.displacements(step), self.section_forces(step)])
        header = "node,x,y,z," + ",".join(DOF_NAMES) + ",N,Vy,Vz,T,My,Mz"
        return _write_csv(path, data, header)

    def summary(self) -> str:
        return (f"model {self.model_name}: {self.n_nodes} nodes, {len(self.connect)} elements\n"
                f"saved steps {self.steps[0]}..{self.steps[-1]} ({len(self.steps)} stored), "
                f"time {self.time[0]:.3f}..{self.time[-1]:.3f} s\n"
                f"status: {self.message} (converged={self.converged}), "
                f"{int(self.iterations.sum())} iterations, {self.elapsed:.2f} s")

    # ------------------------------------------------------------- files
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, model_name=self.model_name, coords=self.coords, connect=self.connect,
            steps=self.steps, time=self.time, static_ratio=self.static_ratio,
            iterations=self.iterations, U=self.U, forces=self.forces,
            converged=self.converged, message=self.message, elapsed=self.elapsed,
            pipe_diam=self.pipe_diam, well_diam=self.well_diam,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Result:
        with np.load(path) as d:
            r = cls(**{k: d[k] for k in ("coords", "connect", "steps", "time", "static_ratio",
                                         "iterations", "U", "forces")},
                    model_name=str(d["model_name"]), converged=bool(d["converged"]),
                    message=str(d["message"]), elapsed=float(d["elapsed"]),
                    pipe_diam=d["pipe_diam"], well_diam=float(d["well_diam"]))
        r._count = len(r.steps)
        return r


def _write_csv(path, data, header) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10g")
    return path
