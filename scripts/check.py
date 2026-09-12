"""Run local checks with a unique workspace-owned pytest temporary directory."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="also test external L96 evidence (requires --evidence-root or RFL_L96_ROOT)",
    )
    ap.add_argument("--evidence-root", type=Path, help="External complete L96 evidence directory")
    args = ap.parse_args()
    evidence = args.evidence_root or os.environ.get("RFL_L96_ROOT")
    if args.legacy and (not evidence or not Path(evidence).is_dir()):
        ap.error("--legacy requires an existing --evidence-root or RFL_L96_ROOT directory")
    if args.evidence_root and not args.legacy:
        ap.error("--evidence-root requires --legacy")
    root = Path(__file__).resolve().parents[1]
    output_root = (root / "outputs").resolve()
    run = (output_root / ("check-" + uuid.uuid4().hex[:12])).resolve()
    temporary = run / "tmp"
    if not run.is_relative_to(output_root) or run.exists() or temporary.exists():
        raise RuntimeError("Expected a fresh temporary path within this repository's outputs")
    run.mkdir(parents=True)
    env = os.environ.copy()
    if args.legacy:
        env["RFL_L96_ROOT"] = str(Path(evidence).resolve())
    else:
        env.pop("RFL_L96_ROOT", None)
    commands = [
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--basetemp",
            str(temporary),
            "--junitxml",
            str(run / "junit.xml"),
        ],
    ]
    outcomes = []
    for i, command in enumerate(commands):
        proc = subprocess.run(
            command,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        (run / f"check-{i}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        print(proc.stdout, end="", flush=True)
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr, flush=True)
        outcomes.append({"command": command[1:], "returncode": proc.returncode})
    files = [root / "pyproject.toml"]
    for directory in ["src", "tests", "scripts"]:
        files.extend((root / directory).rglob("*.py"))
    result = {
        "passed": all(r["returncode"] == 0 for r in outcomes),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "legacy_enabled": args.legacy,
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["numpy", "pytest", "ruff", "response-fidelity-lab"]
        },
        "commands": outcomes,
        "source_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)
        },
    }
    (run / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "validation": str(run / "validation.json")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
