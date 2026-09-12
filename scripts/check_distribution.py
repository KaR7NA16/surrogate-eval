"""Verify an installed wheel from a fresh working directory and exercise documented flows."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import uuid
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True, help="Clean wheel environment Python")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    interpreter = args.python.resolve()
    output = root / "outputs" / ("distribution-" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    files = list((root / "dist").glob("*.whl")) + list((root / "dist").glob("*.tar.gz"))
    assert len(files) == 2, "Expected one wheel and one source distribution"
    for path in files:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        else:
            with tarfile.open(path) as archive:
                names = archive.getnames()
        assert not any("reproduction/l96/" in n or n.endswith(".npz") for n in names)
    missing = []
    documents = [
        root / "README.md",
        root / "README.zh-CN.md",
        root / "CONTRIBUTING.md",
        root / "reproduction/README.md",
    ]
    documents += list((root / "docs").glob("*.md"))
    for doc in documents:
        for link in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if "://" not in link and not link.startswith("#"):
                if not (doc.parent / link.split("#")[0]).exists():
                    missing.append(f"{doc.name}: {link}")
    assert not missing, missing
    commands = [
        [
            "-c",
            "import response_fidelity, importlib.util; "
            "assert 'site-packages' in response_fidelity.__file__; "
            "assert importlib.util.find_spec('torch') is None; "
            "print(response_fidelity.__file__)",
        ],
        ["-m", "response_fidelity", "demo", "--system", "linear", "--output", "linear"],
        ["-m", "response_fidelity", "demo", "--system", "pendulum", "--output", "pendulum"],
        [
            "-m",
            "response_fidelity",
            "diagnose",
            "--input",
            "linear/geometry.npz",
            "--output",
            "geometry.json",
            "--markdown",
            "geometry.md",
        ],
        [
            "-m",
            "response_fidelity",
            "noise",
            "--input",
            "linear/geometry.npz",
            "--sigma",
            "0",
            "0.001",
            "0.01",
            "--output",
            "noise.json",
        ],
    ]
    tutorial = (
        (root / "docs/your_model.md")
        .read_text(encoding="utf-8")
        .split("```python\n")[1]
        .split("```")[0]
    )
    (output / "tutorial.py").write_text(tutorial, encoding="utf-8")
    commands.append(["tutorial.py"])
    commands.append(
        [
            "-m",
            "response_fidelity",
            "score",
            "--reference",
            "outputs/external-model/reference.npz",
            "--prediction",
            "outputs/external-model/model.npz",
            "--output",
            "tutorial-cli-score.json",
            "--markdown",
            "tutorial-score.md",
        ]
    )
    outcomes = []
    for i, command in enumerate(commands):
        proc = subprocess.run(
            [str(interpreter), *command],
            cwd=output,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        (output / f"command-{i}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        outcomes.append({"arguments": command, "returncode": proc.returncode})
        if proc.returncode:
            print(proc.stdout + proc.stderr)
    result = {
        "passed": all(x["returncode"] == 0 for x in outcomes),
        "python": str(interpreter),
        "commands": outcomes,
        "missing_document_links": missing,
        "distributions": {
            p.name: {
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in files
        },
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "report": str(output / "validation.json")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
