"""Cluster-weighted response errors and paired model comparisons."""

import numpy as np

from .schema import PairedReference, Prediction


def evaluate(reference: PairedReference, prediction: Prediction):
    """Score physical predictions; independent clusters receive equal weight.

    Outputs are mean squared errors in the supplied, fixed scale. Spatial
    coordinates and initializations are not promoted to independent replicates.
    """
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            return _evaluate(reference, prediction)
    except FloatingPointError as exc:
        raise ValueError(
            "Scoring overflowed; check units, scales and perturbation amplitudes"
        ) from exc


def _evaluate(reference, prediction):
    pred = prediction.aligned(reference)
    scales = np.broadcast_to(reference.scale, (len(pred), pred.shape[2]))
    scale = scales[:, :, None]
    eps = reference.epsilon[:, None, None]
    error = (pred - reference.reference) / scale[:, None, :, :]
    nominal = error[:, 0] ** 2
    response_error = ((error[:, 1] - error[:, 2]) / (2 * eps)) ** 2
    actual = (reference.reference[:, 1] - reference.reference[:, 2]) / (2 * eps * scale)
    estimated = (pred[:, 1] - pred[:, 2]) / (2 * eps * scale)
    # Exploratory, separate scalar decisions: minimize each target using +/- perturbation.
    chosen = np.where(estimated <= 0, actual, -actual)
    regret = chosen + np.abs(actual)
    rows = []
    for cluster in sorted(set(reference.cluster_ids)):
        mask = reference.cluster_ids == cluster
        row = {
            "cluster_id": str(cluster),
            "n_samples": int(mask.sum()),
            "nominal_by_target_and_horizon": nominal[mask].mean(axis=0).tolist(),
            "response_by_target_and_horizon": response_error[mask].mean(axis=0).tolist(),
            "nominal_by_horizon": nominal[mask].mean(axis=(0, 1)).tolist(),
            "response_by_horizon": response_error[mask].mean(axis=(0, 1)).tolist(),
            "true_response_second_moment_by_horizon": (actual[mask] ** 2)
            .mean(axis=(0, 1))
            .tolist(),
            "predicted_response_second_moment_by_horizon": (estimated[mask] ** 2)
            .mean(axis=(0, 1))
            .tolist(),
            "wrong_direction_rate_by_horizon": (chosen[mask] > 0).mean(axis=(0, 1)).tolist(),
            "decision_regret_by_horizon": regret[mask].mean(axis=(0, 1)).tolist(),
            "random_expected_regret_by_horizon": np.abs(actual[mask]).mean(axis=(0, 1)).tolist(),
        }
        row["nominal_mse"] = float(np.mean(row["nominal_by_horizon"]))
        row["response_mse"] = float(np.mean(row["response_by_horizon"]))
        rows.append(row)
    keys = [k for k in rows[0] if k not in ("cluster_id", "n_samples")]
    aggregate = {}
    for key in keys:
        value = np.mean([row[key] for row in rows], axis=0)
        aggregate[key] = float(value) if np.ndim(value) == 0 else value.tolist()
    return {
        "report_version": 1,
        "kind": "evaluation",
        "n_samples": len(pred),
        "n_clusters": len(rows),
        "n_targets": pred.shape[2],
        "target_ids": reference.target_ids.tolist(),
        "horizons": reference.horizons.tolist(),
        "units": reference.units,
        "epsilon_range": [float(reference.epsilon.min()), float(reference.epsilon.max())],
        "scale_layout": "per_target" if reference.scale.ndim == 1 else "per_sample_target",
        "scale_range_by_target": [scales.min(axis=0).tolist(), scales.max(axis=0).tolist()],
        "weighting": "equal cluster means; equal target/horizon weights within a cluster",
        "aggregate": aggregate,
        "clusters": rows,
        "limits": [
            "Scores require matched physical perturbation branches and reference outputs.",
            "Cluster labels and training-only scales are supplied by the caller; independence and absence of leakage are not certified.",
            "Directional errors measure supplied perturbations, not an unmeasured full Jacobian.",
            "Direction-choice regret is an exploratory one-step scalar task, not closed-loop control. Prediction ties choose the plus branch.",
        ],
    }


def _ratio_reduction(candidate, baseline):
    return None if baseline == 0 else float(1 - candidate / baseline)


def compare(reference, predictions, baseline, bootstrap_samples=2000, seed=0):
    """Paired cluster comparisons; positive reductions mean lower error.

    Percentile intervals resample the supplied clusters. They are descriptive,
    conditional on fixed models/protocol, and cannot establish unseen-benchmark
    confirmation or account for earlier adaptive model development.
    """
    if baseline not in predictions or len(predictions) < 2:
        raise ValueError("comparison requires a named baseline and at least one candidate")
    if any(not isinstance(name, str) or not name.strip() for name in predictions):
        raise ValueError("model names must be nonempty strings")
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    reports = {name: evaluate(reference, pred) for name, pred in predictions.items()}
    base = reports[baseline]
    n = base["n_clusters"]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, (bootstrap_samples, n)) if n > 1 and bootstrap_samples else None
    comparisons = []
    for name, report in reports.items():
        if name == baseline:
            continue
        row = {"candidate": name, "baseline": baseline}
        for metric in ["nominal_mse", "response_mse"]:
            b = np.array([r[metric] for r in base["clusters"]])
            c = np.array([r[metric] for r in report["clusters"]])
            absolute = float((b - c).mean())
            interval = None
            ratio_interval = None
            if indices is not None:
                db = b[indices].mean(axis=1)
                dc = c[indices].mean(axis=1)
                interval = np.quantile(db - dc, [0.025, 0.975]).tolist()
                if np.all(db > 0):
                    ratio_interval = np.quantile(1 - dc / db, [0.025, 0.975]).tolist()
            row[metric] = {
                "absolute_reduction": absolute,
                "relative_reduction": _ratio_reduction(float(c.mean()), float(b.mean())),
                "median_cluster_relative_reduction": float(np.median(1 - c / b))
                if np.all(b > 0)
                else None,
                "positive_clusters": int(np.sum(c < b)),
                "n_clusters": n,
                "absolute_reduction_interval_95": interval,
                "relative_reduction_interval_95": ratio_interval,
                "relative_reduction_by_horizon": [
                    _ratio_reduction(cc, bb)
                    for cc, bb in zip(
                        report["aggregate"][metric.replace("_mse", "_by_horizon")],
                        base["aggregate"][metric.replace("_mse", "_by_horizon")],
                        strict=True,
                    )
                ],
            }
        comparisons.append(row)
    return {
        "report_version": 1,
        "kind": "comparison",
        "baseline": baseline,
        "models": reports,
        "comparisons": comparisons,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "unit": "paired supplied clusters",
            "method": "percentile 95%; no interval for one cluster",
            "scope": "descriptive uncertainty for fixed models; not a significance claim or adaptive-selection correction",
        },
        "limits": [
            "Zero baseline error makes relative improvement undefined (null); absolute errors remain available.",
            "A lower average error does not imply improvement at every horizon or in every cluster.",
        ],
    }
