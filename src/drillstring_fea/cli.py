"""Command line interface.

    drillstring-fea list                                  # cases and their parameters
    drillstring-fea run CASE [name=value ...] [-o FILE] [--corrections PRESET]
    drillstring-fea info FILE                             # summary of a result file
    drillstring-fea export FILE --step N -o out.csv       # one step, all nodes
    drillstring-fea export FILE --node N --dof DZ -o out.csv   # one DOF, all steps
    drillstring-fea animate FILE -o movie.mp4 [--scale 250] # MP4 video (needs PyVista)
    drillstring-fea show FILE [--step N] [--png out.png]    # 3D view of one step
    drillstring-fea dashboard FILE -o board.mp4             # drill-string dashboard video

In ``run``, each ``name=value`` is either a parameter of the case function
(e.g. ``case=2``, ``rpm=120``) or a field of ``Settings`` (e.g. ``t_final=10``,
``dt=0.005``, ``save_every=10``).
"""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
from dataclasses import fields, replace

import numpy as np

from .cases import CASES
from .model import DOF_NAMES, Corrections, Settings
from .results import Result
from .solver import solve


def _value(text: str):
    """'2' -> 2, '0.5' -> 0.5, '(15, 30)' -> (15, 30), 'abc' -> 'abc'."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def cmd_list(_args) -> int:
    print("Cases (parameters and default values):\n")
    for name, func in CASES.items():
        params = ", ".join(f"{p.name}={p.default!r}"
                           for p in inspect.signature(func).parameters.values())
        print(f"  {name}\n      {inspect.getdoc(func).splitlines()[0]}\n      {params or '-'}")
    print("\nSettings that can be changed in every case:")
    print("  " + ", ".join(f"{f.name}={f.default!r}" for f in fields(Settings)))
    print("\nCorrections (--corrections PRESET):")
    print("  default      = " + ", ".join(f"{f.name}={f.default}" for f in fields(Corrections)))
    print("  dissertation = all off (reproduces the published results)")
    print("  all          = all on")
    return 0


def cmd_run(args) -> int:
    func = CASES[args.case]
    case_params = set(inspect.signature(func).parameters)
    setting_names = {f.name for f in fields(Settings)}
    case_kw, setting_kw = {}, {}
    for item in args.params:
        name, sep, text = item.partition("=")
        if not sep:
            raise SystemExit(f"expected name=value, got {item!r}")
        if name in case_params:
            case_kw[name] = _value(text)
        elif name in setting_names:
            setting_kw[name] = _value(text)
        else:
            raise SystemExit(f"unknown parameter {name!r} (see 'drillstring-fea list')")

    model, settings = func(**case_kw)
    settings = replace(settings, **setting_kw)
    corrections = Corrections.preset(args.corrections)

    every = max(1, settings.n_steps // 20)  # print progress about 20 times

    def progress(step, total, iterations):
        if step % every == 0 or step == total:
            print(f"  step {step}/{total} ({iterations} iterations)", flush=True)

    print(f"{model.description}: {model.n_elements} elements, {settings.n_steps} steps")
    result = solve(model, settings, corrections, on_step=progress)
    print(result.summary())
    out = args.output or f"results/{args.case}.npz"
    print(f"saved to {result.save(out)}")
    return 0 if result.converged else 1


def cmd_info(args) -> int:
    print(Result.load(args.file).summary())
    return 0


def cmd_export(args) -> int:
    result = Result.load(args.file)
    if args.node is not None:
        path = result.history_to_csv(args.output, args.node, DOF_NAMES.index(args.dof))
    else:
        path = result.step_to_csv(args.output, args.step)
    print(f"written {path}")
    return 0


def _z_range(args) -> tuple[float, float] | None:
    """(z_min, z_max) of the part of the string to draw, or None for all of it."""
    if args.z_min is None and args.z_max is None:
        return None
    return (-np.inf if args.z_min is None else args.z_min,
            np.inf if args.z_max is None else args.z_max)


def _scene_options(args) -> dict:
    return {"displacement_scale": args.scale, "axial_scale": args.axial_scale,
            "pipe_scale": args.pipe_scale, "well_scale": args.well_scale, "rings": args.rings,
            "color_by": args.color_by, "z_range": _z_range(args)}


def cmd_animate(args) -> int:
    from .animation import animate

    result = Result.load(args.file)
    path = animate(result, args.output, start=args.start, end=args.end, every=args.every,
                   fps=args.fps, **_scene_options(args))
    print(f"written {path}")
    return 0


def cmd_dashboard(args) -> int:
    from .dashboard import dashboard, dashboard_frame

    result = Result.load(args.file)
    options = {"color_by": args.color_by, "node": args.node, "z_range": _z_range(args),
               "displacement_scale": args.scale, "pipe_scale": args.pipe_scale,
               "well_scale": args.well_scale, "title": args.title}
    if args.output.lower().endswith(".png"):
        path = dashboard_frame(result, args.output, step=args.step, **options)
    else:
        path = dashboard(result, args.output, start=args.start, end=args.end, every=args.every,
                         fps=args.fps, **options)
    print(f"written {path}")
    return 0


def cmd_show(args) -> int:
    from .animation import show

    show(Result.load(args.file), args.step, screenshot=args.png, **_scene_options(args))
    if args.png:
        print(f"written {args.png}")
    return 0


def _add_scene_arguments(parser) -> None:
    parser.add_argument("--scale", type=float, default=1.0,
                        help="magnification of the lateral displacements (250 for drill strings)")
    parser.add_argument("--axial-scale", type=float, default=1.0,
                        help="magnification of the axial (z) displacements (default 1)")
    parser.add_argument("--pipe-scale", type=float, default=1.0,
                        help="magnification of the pipe diameter (20 for drill strings)")
    parser.add_argument("--well-scale", type=float, default=1.0,
                        help="magnification of the well diameter (125 for drill strings)")
    parser.add_argument("--rings", choices=["tangent", "rotations"], default="tangent",
                        help="rings perpendicular to the drawn axis (default) or turned by "
                             "the node rotations")
    parser.add_argument("--color-by", choices=["N", "Vy", "Vz", "T", "My", "Mz", "lateral"],
                        help="color the pipe by a section force or the lateral displacement")
    parser.add_argument("--z-min", type=float, help="draw only nodes with z >= z_min")
    parser.add_argument("--z-max", type=float, help="draw only nodes with z <= z_max")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="drillstring-fea",
                                     description="Nonlinear dynamic FEA of drill strings")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list cases and parameters").set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="run a case")
    run.add_argument("case", choices=list(CASES))
    run.add_argument("params", nargs="*", help="name=value (case parameter or setting)")
    run.add_argument("-o", "--output", help="result file (default results/CASE.npz)")
    run.add_argument("--corrections", default="default", choices=["default", "dissertation", "all"],
                     help="'dissertation' reproduces the published results")
    run.set_defaults(func=cmd_run)

    info = sub.add_parser("info", help="summary of a result file")
    info.add_argument("file")
    info.set_defaults(func=cmd_info)

    export = sub.add_parser("export", help="export results to CSV")
    export.add_argument("file")
    export.add_argument("-o", "--output", required=True, help="CSV file")
    export.add_argument("--step", type=int, help="step to export (default: last)")
    export.add_argument("--node", type=int, help="export the history of this node")
    export.add_argument("--dof", default="DZ", choices=DOF_NAMES, help="DOF of --node")
    export.set_defaults(func=cmd_export)

    anim = sub.add_parser("animate", help="MP4 animation of a result file (needs PyVista)")
    anim.add_argument("file")
    anim.add_argument("-o", "--output", required=True, help="video file (.mp4)")
    anim.add_argument("--start", type=int, help="first step (default: first saved)")
    anim.add_argument("--end", type=int, help="last step (default: last saved)")
    anim.add_argument("--every", type=int, default=1, help="use every n-th saved step")
    anim.add_argument("--fps", type=int, default=25, help="frames per second")
    _add_scene_arguments(anim)
    anim.set_defaults(func=cmd_animate)

    view = sub.add_parser("show", help="3D view of one step (needs PyVista)")
    view.add_argument("file")
    view.add_argument("--step", type=int, help="step to show (default: last saved)")
    view.add_argument("--png", help="save a picture instead of opening a window")
    _add_scene_arguments(view)
    view.set_defaults(func=cmd_show)

    board = sub.add_parser(
        "dashboard", help="dashboard (3D + top view + profiles + histories) of a drill string")
    board.add_argument("file")
    board.add_argument("-o", "--output", required=True, help=".mp4 video or .png picture")
    board.add_argument("--step", type=int, help="step of the .png picture (default: last)")
    board.add_argument("--start", type=int, help="first step of the video")
    board.add_argument("--end", type=int, help="last step of the video")
    board.add_argument("--every", type=int, default=1, help="use every n-th saved step")
    board.add_argument("--fps", type=int, default=25, help="frames per second")
    board.add_argument("--color-by", default="lateral",
                       choices=["lateral", "N", "T", "My", "Mz"], help="color of the 3D view")
    board.add_argument("--node", type=int,
                       help="node whose path is drawn in the top view (default: largest motion)")
    board.add_argument("--z-min", type=float, help="draw only nodes with z >= z_min")
    board.add_argument("--z-max", type=float, help="draw only nodes with z <= z_max")
    board.add_argument("--scale", type=float, default=250.0,
                       help="lateral displacement magnification")
    board.add_argument("--pipe-scale", type=float, default=20.0, help="pipe diameter magnification")
    board.add_argument("--well-scale", type=float, default=125.0,
                       help="well diameter magnification")
    board.add_argument("--title", help="title of the dashboard")
    board.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
