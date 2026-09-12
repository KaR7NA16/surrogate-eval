import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from surrogate_eval.examples import make_demo, pendulum_flow
from surrogate_eval.schema import load_reference


def run(*args, cwd=None, check=True):
    return subprocess.run(
        [sys.executable, "-m", "surrogate_eval", *map(str, args)],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def test_core_import_has_no_training_dependency():
    code = "import surrogate_eval,sys; assert not any(x in sys.modules for x in ['torch','scipy','matplotlib'])"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("system", ["linear", "pendulum"])
def test_demo_external_scoring_and_report(tmp_path, system):
    folder = tmp_path / system
    run("demo", "--system", system, "--output", folder, cwd=tmp_path)
    run(
        "score",
        "--reference",
        folder / "test_reference.npz",
        "--prediction",
        folder / "response_head_predictions.npz",
        "--output",
        tmp_path / "score.json",
        "--markdown",
        tmp_path / "score.md",
        cwd=tmp_path,
    )
    result = json.loads((tmp_path / "score.json").read_text())
    assert result["n_targets"] == 1
    assert result["n_clusters"] == 4
    assert result["provenance"]["inputs"][0]["file"] == "test_reference.npz"
    assert "Interpretation limits" in (tmp_path / "score.md").read_text()
    bad = run("demo", "--system", system, "--output", folder, check=False)
    assert bad.returncode == 2


def test_compare_and_geometry_workflows(tmp_path):
    folder = tmp_path / "linear"
    make_demo(folder)
    run(
        "compare",
        "--reference",
        folder / "test_reference.npz",
        "--model",
        f"base={folder / 'persistence_predictions.npz'}",
        "--model",
        f"fit={folder / 'response_head_predictions.npz'}",
        "--baseline",
        "base",
        "--output",
        tmp_path / "result.json",
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["comparisons"][0]["response_mse"]["relative_reduction"] > 0.99
    run("diagnose", "--input", folder / "geometry.npz", "--output", tmp_path / "geometry.json")
    run(
        "noise",
        "--input",
        folder / "geometry.npz",
        "--sigma",
        "0",
        ".01",
        "1",
        "--output",
        tmp_path / "noise.json",
    )
    assert json.loads((tmp_path / "geometry.json").read_text())["mean_invisible"] < 1e-24


def test_fit_command_requires_no_test_labels(tmp_path):
    folder = tmp_path / "linear"
    make_demo(folder)
    run(
        "fit-head",
        "--train-reference",
        folder / "train_reference.npz",
        "--train-features",
        folder / "train_features.npz",
        "--validation-reference",
        folder / "validation_reference.npz",
        "--validation-features",
        folder / "validation_features.npz",
        "--output",
        tmp_path / "model.npz",
        "--nominal-output",
        tmp_path / "nominal.npz",
        "--report",
        tmp_path / "fit.json",
    )
    run(
        "predict-head",
        "--model",
        tmp_path / "model.npz",
        "--features",
        folder / "test_features.npz",
        "--output",
        tmp_path / "pred.npz",
    )
    assert len(load_reference(folder / "test_reference.npz").reference) == 64


def test_pendulum_has_reference_refinement_check():
    x = np.array([[0.6, -0.2], [-1.0, 0.3]])
    np.testing.assert_allclose(
        pendulum_flow(x, 0.4), pendulum_flow(x, 0.4, 0.0005), rtol=1e-11, atol=1e-12
    )


@pytest.mark.skipif(
    not os.environ.get("SURROGATE_EVAL_L96_ROOT"), reason="optional frozen L96 data not configured"
)
def test_optional_l96_migration_reproduces_original_summary(tmp_path):
    root = Path(os.environ["SURROGATE_EVAL_L96_ROOT"])
    run("l96-export", "--root", root, "--output", tmp_path / "l96")
    folder = tmp_path / "l96"
    out = run(
        "compare",
        "--reference",
        folder / "reference.npz",
        "--model",
        f"original={folder / 'original.npz'}",
        "--model",
        f"trained={folder / 'response_trained.npz'}",
        "--baseline",
        "original",
        "--bootstrap-samples",
        "0",
    )
    report = json.loads(out.stdout)
    old = json.loads((root / "RESP_finetune_results_20260912.json").read_text())
    expected = next(
        s
        for s in old["summary"]
        if s["split"] == "fresh_test"
        and s["forcing"] == 24
        and s["block"] == "history"
        and s["arm"] == "response_head"
    )
    got = report["comparisons"][0]["response_mse"]["median_cluster_relative_reduction"]
    assert got == pytest.approx(expected["median_response_reduction"], abs=1e-12)
    assert report["models"]["original"]["n_clusters"] == 8
