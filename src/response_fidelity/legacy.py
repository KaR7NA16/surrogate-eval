"""Read-only adapter for the complete frozen L96 release.

Experimental dates and network geometry are confined to this adapter, never
required by the generic reference/prediction scoring interface.
"""

import json
from pathlib import Path
import numpy as np

from .schema import PairedReference, Prediction, save_reference, save_prediction, sha256, write_json

MANIFEST = "RESP_response_lab_release_manifest_20260912.json"


def _path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Historical file points outside evidence root")
    return path


def verify_evidence(root):
    root = Path(root)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = _path(root, name)
        if not path.is_file():
            raise ValueError(
                f"Missing optional evidence file: {name}. Restore the original local release before this operation."
            )
        if sha256(path) != expected:
            raise ValueError(f"Historical evidence hash mismatch: {name}")
    return {
        "passed": True,
        "verified_files": len(manifest["files"]),
        "manifest_sha256": sha256(root / MANIFEST),
    }


def _arrays(root, record):
    path = _path(root, record["file"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Historical array hash mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name].copy() for name in z.files}


def _predict(history, z, name, block, arm, study):
    mean, scale = float(z["state_mean"]), float(z["state_scale"])
    q = (history - mean) / scale
    current = q[:, :, 2]
    if block == "history":
        fields, offsets = [current, q[:, :, 0] - current, q[:, :, 1] - current], range(-2, 3)
    else:
        fields, offsets = [current], range(-7, 8)
    x = np.stack([np.roll(f, -i, axis=-1) for f in fields for i in offsets], axis=-1)
    x = (x - z[name + "_center"]) / z[name + "_scale"]
    prefix = name if study == "intervention" else name + "_" + arm
    h = np.tanh(x @ z[prefix + "_W1"].T + z[prefix + "_b1"])
    h = np.tanh(h @ z[prefix + "_W2"].T + z[prefix + "_b2"])
    if study == "intervention":
        coef = z[name + "_" + arm]
        out = h @ coef[1:] + coef[0]
    else:
        out = h @ z[prefix + "_W3"].T + z[prefix + "_b3"]
    return out * scale + history[:, :, 2, :, None]


def export_l96(root, output, study="finetune", forcing=24, block="history", initialization=None):
    """Export all eight dataset clusters as generic version-1 reference/predictions.

    By default all three initialization errors are represented as repeated rows
    within the same dataset cluster. They never create independent clusters.
    """
    if (
        study not in ["intervention", "finetune"]
        or forcing not in [16, 24]
        or block not in ["history", "current_wide"]
    ):
        raise ValueError("Unsupported historical study, forcing or input block")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Export output must be new or empty")
    verify = verify_evidence(root)
    root = Path(root)
    result = json.loads((root / f"RESP_{study}_results_20260912.json").read_text(encoding="utf-8"))
    arms = {
        "incumbent": "original",
        "nominal_head": "nominal_refit" if study == "intervention" else "nominal_continuation",
        "response_selection": "response_selected",
        "response_head": "response_trained",
    }
    values, scales, ids, clusters = [], [], [], []
    predictions = {name: [] for name in arms.values()}
    physical_starts = 0
    for row in result["replicates"]:
        if row["forcing"] != forcing:
            continue
        z = _arrays(root, row["arrays"])
        fresh = _arrays(root, row["fresh_source"])
        lo, hi = row["fresh_source"]["slice"]
        obs = fresh["observations"][lo:hi]
        physical_starts += len(obs)
        runs = [
            r
            for r in row["runs"]
            if r["block"] == block
            and (initialization is None or r["initialization_seed"] == initialization)
        ]
        if not runs:
            raise ValueError("Initialization not available in historical model set")
        for run in runs:
            seed = run["initialization_seed"]
            name = f"{block}_{seed}"
            values.append(obs[:, :, 3:].swapaxes(-2, -1))
            scales.append(np.full((len(obs), obs.shape[-1]), float(z["state_scale"])))
            ids.extend(
                [
                    f"F{forcing}-data{row['dataset_seed']}-init{seed}-start{i}"
                    for i in range(len(obs))
                ]
            )
            clusters.extend([str(row["dataset_seed"])] * len(obs))
            for arm, public in arms.items():
                predictions[public].append(_predict(obs[:, :, :3], z, name, block, arm, study))
    if not values:
        raise ValueError("No matching historical data")
    ref = PairedReference(
        np.concatenate(values),
        0.01,
        np.concatenate(scales),
        np.array([0.05, 0.1, 0.2]),
        np.array(ids),
        np.array(clusters),
        "Lorenz-96 state units",
    )
    output.mkdir(parents=True, exist_ok=True)
    save_reference(output / "reference.npz", ref)
    for name, arrays in predictions.items():
        save_prediction(
            output / f"{name}.npz",
            Prediction(np.concatenate(arrays), ref.sample_ids, ref.horizons, ref.units),
        )
    metadata = {
        "study": study,
        "forcing": forcing,
        "block": block,
        "physical_starts": physical_starts,
        "evaluation_rows": len(ref.reference),
        "clusters": len(set(clusters)),
        "initialization": initialization or "all three",
        "aggregation": "initialization rows stay inside the same training-dataset cluster; do not count them as independent datasets",
        "source_verification": verify,
    }
    write_json(output / "export_metadata.json", metadata)
    return metadata
