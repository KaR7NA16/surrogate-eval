"""Command-line workflows for independent evaluation and optional baselines."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from . import __version__
from .metrics import compare, evaluate
from .report import write_markdown
from .schema import load_prediction, load_reference, save_prediction, sha256, write_json


def _outputs(paths):
    paths = [Path(p).resolve() for p in paths if p is not None]
    if len(paths) != len(set(paths)):
        raise ValueError("Output paths must be distinct")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing overwrite: {path}")


def _provenance(result, paths):
    result["provenance"] = {
        "software_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [{"file": Path(p).name, "sha256": sha256(p)} for p in paths],
    }
    return result


def _emit(result, output=None, markdown=None):
    if output:
        write_json(output, result)
    if markdown:
        write_markdown(markdown, result)
    if output or markdown:
        print(
            json.dumps(
                {
                    "output": str(output) if output else None,
                    "markdown": str(markdown) if markdown else None,
                }
            )
        )
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


def parser():
    ap = argparse.ArgumentParser(
        prog="surrogate-eval",
        description="Evaluate paired perturbation response of scientific surrogate models",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate a small independent CPU example")
    demo.add_argument("--system", choices=["linear", "pendulum"], default="linear")
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--output", type=Path, required=True)
    for command in ["score", "compare"]:
        p = sub.add_parser(command)
        p.add_argument("--reference", type=Path, required=True)
        p.add_argument("--output", type=Path)
        p.add_argument("--markdown", type=Path)
        if command == "score":
            p.add_argument("--prediction", type=Path, required=True)
        else:
            p.add_argument(
                "--model", action="append", required=True, metavar="NAME=PREDICTIONS.npz"
            )
            p.add_argument("--baseline", required=True)
            p.add_argument("--bootstrap-samples", type=int, default=2000)
            p.add_argument("--seed", type=int, default=0)
    for command in ["diagnose", "noise"]:
        p = sub.add_parser(
            command, help="Optional pointwise analysis requiring supplied reference Jacobians"
        )
        p.add_argument("--input", type=Path, required=True, help="NPZ with A,B and optionally M")
        p.add_argument("--cutoff", type=float, default=1e-10)
        p.add_argument("--output", type=Path)
        p.add_argument("--markdown", type=Path)
        if command == "noise":
            p.add_argument("--sigma", type=float, nargs="+", required=True)
    fit = sub.add_parser(
        "fit-head",
        help="Fit a known fixed-feature ridge response baseline using train/validation only",
    )
    for name in [
        "train-reference",
        "train-features",
        "validation-reference",
        "validation-features",
        "output",
        "nominal-output",
        "report",
    ]:
        fit.add_argument("--" + name, type=Path, required=True)
    fit.add_argument("--ridge", nargs="+", type=float, default=[1e-8, 1e-6, 1e-4, 0.01, 1])
    fit.add_argument("--response-multipliers", nargs="+", type=float, default=[0, 0.1, 1, 10])
    fit.add_argument("--nominal-cap", type=float, default=1.05)
    predict = sub.add_parser("predict-head")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--features", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    old = sub.add_parser(
        "l96-export", help="Convert the optional frozen L96 evidence to the generic data format"
    )
    old.add_argument("--root", type=Path, required=True)
    old.add_argument("--output", type=Path, required=True)
    old.add_argument("--study", choices=["intervention", "finetune"], default="finetune")
    old.add_argument("--forcing", type=int, choices=[16, 24], default=24)
    old.add_argument("--block", choices=["history", "current_wide"], default="history")
    old.add_argument("--initialization", type=int)
    evidence = sub.add_parser("verify-evidence")
    evidence.add_argument("--root", type=Path, required=True)
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        if args.command == "demo":
            from .examples import make_demo

            result = make_demo(args.output, args.system, args.seed)
            write_markdown(args.output / "comparison.md", result)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "system": args.system,
                        "report": str(args.output / "comparison.md"),
                    }
                )
            )
        elif args.command in ["score", "compare"]:
            _outputs([args.output, args.markdown])
            ref = load_reference(args.reference)
            if args.command == "score":
                result = evaluate(ref, load_prediction(args.prediction))
                paths = [args.reference, args.prediction]
            else:
                models = {}
                paths = [args.reference]
                for item in args.model:
                    if "=" not in item:
                        raise ValueError("--model must be NAME=PATH")
                    name, path = item.split("=", 1)
                    if name in models or not name.strip():
                        raise ValueError("Model names must be nonempty and unique")
                    models[name] = load_prediction(path)
                    paths.append(path)
                result = compare(ref, models, args.baseline, args.bootstrap_samples, args.seed)
            _emit(_provenance(result, paths), args.output, args.markdown)
        elif args.command in ["diagnose", "noise"]:
            from .geometry import decompose, oracle_risk

            _outputs([args.output, args.markdown])
            with np.load(args.input, allow_pickle=False) as z:
                result = (
                    decompose(z["A"], z["B"], z["M"] if "M" in z else None, args.cutoff)
                    if args.command == "diagnose"
                    else oracle_risk(z["A"], z["B"], args.sigma, args.cutoff)
                )
            _emit(_provenance(result, [args.input]), args.output, args.markdown)
        elif args.command == "fit-head":
            from .repair import fit_heads, load_features, save_head

            _outputs([args.output, args.nominal_output, args.report])
            models, result = fit_heads(
                load_reference(args.train_reference),
                load_features(args.train_features),
                load_reference(args.validation_reference),
                load_features(args.validation_features),
                args.ridge,
                args.response_multipliers,
                args.nominal_cap,
            )
            _provenance(
                result,
                [
                    args.train_reference,
                    args.train_features,
                    args.validation_reference,
                    args.validation_features,
                ],
            )
            save_head(args.output, models["response"], result)
            save_head(args.nominal_output, models["nominal"], result)
            write_json(args.report, result)
            print(
                json.dumps(
                    {
                        "model": str(args.output),
                        "nominal_model": str(args.nominal_output),
                        "report": str(args.report),
                    }
                )
            )
        elif args.command == "predict-head":
            from .repair import load_features, load_head

            _outputs([args.output])
            pred = load_head(args.model).predict(load_features(args.features))
            save_prediction(args.output, pred)
            print(json.dumps({"output": str(args.output), "rows": len(pred.predictions)}))
        elif args.command == "verify-evidence":
            from .legacy import verify_evidence

            print(json.dumps(verify_evidence(args.root)))
        elif args.command == "l96-export":
            from .legacy import export_l96

            print(
                json.dumps(
                    export_l96(
                        args.root,
                        args.output,
                        args.study,
                        args.forcing,
                        args.block,
                        args.initialization,
                    )
                )
            )
    except (ValueError, OSError, KeyError, ArithmeticError) as exc:
        print(f"surrogate-eval: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
