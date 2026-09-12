"""Optional pointwise tangent diagnostics with explicit numerical-rank semantics.

These functions need reference derivatives. They do not estimate observability
from ordinary time-series samples and do not prove global reconstructability.
"""

import numpy as np

from .schema import finite_array


def _inputs(a, b, cutoff):
    a = finite_array(a, "input Jacobian A")
    b = finite_array(b, "target Jacobian B")
    if a.ndim < 2 or b.ndim != a.ndim or a.shape[:-2] != b.shape[:-2]:
        raise ValueError("A and B must have matching batch axes and explicit row/direction axes")
    if any(n == 0 for n in a.shape + b.shape) or a.shape[-1] != b.shape[-1]:
        raise ValueError("A and B must have nonempty, equal state-direction dimensions")
    if not np.isscalar(cutoff) or not np.isfinite(cutoff) or not 0 <= cutoff < 1:
        raise ValueError("cutoff must lie in [0,1)")
    _, singular, vectors = np.linalg.svd(a, full_matrices=False)
    keep = singular > cutoff * singular[..., :1]
    vectors = vectors * keep[..., :, None]
    projected = (b @ vectors.swapaxes(-2, -1)) @ vectors
    return a, b, singular, vectors, keep, projected


def decompose(a, b, predicted=None, cutoff=1e-10, chain_tolerance=1e-8):
    """Split an input-realizable fitted Jacobian's mean squared tangent error.

    A: [..., input coordinates, full-state directions].
    B and predicted M: [..., target outputs, full-state directions].
    Perturbation covariance is I/d. Row spaces use a relative SVD cutoff.
    If M is not in row(A), the decomposition is rejected instead of misreported.
    """
    a, b, singular, vectors, keep, projected = _inputs(a, b, cutoff)
    if not np.isscalar(chain_tolerance) or not np.isfinite(chain_tolerance) or chain_tolerance <= 0:
        raise ValueError("chain_tolerance must be positive")
    d = a.shape[-1]
    invisible = np.sum((b - projected) ** 2, axis=-1) / d
    result = {
        "report_version": 1,
        "kind": "geometry",
        "state_dimension": d,
        "rank": keep.sum(axis=-1).tolist(),
        "singular_values": singular.tolist(),
        "cutoff": float(cutoff),
        "invisible_by_target": invisible.tolist(),
        "mean_invisible": float(invisible.mean()),
        "assumptions": "Pointwise linearization, Euclidean metric, isotropic unit perturbation with covariance I/d",
        "limits": [
            "Local row-space access does not establish a globally realizable or learnable predictor.",
            "Visible error can contain model, estimation, optimization and global-input-ambiguity effects.",
            "The numerical rank threshold is part of the reported calculation.",
        ],
    }
    if predicted is not None:
        m = finite_array(predicted, "predicted Jacobian M")
        if m.shape != b.shape:
            raise ValueError("M must have the same shape as B")
        mp = (m @ vectors.swapaxes(-2, -1)) @ vectors
        denominator = np.maximum(
            np.maximum(np.linalg.norm(m, axis=(-2, -1)), np.linalg.norm(b, axis=(-2, -1))),
            np.finfo(float).tiny,
        )
        leakage = float(np.max(np.linalg.norm(m - mp, axis=(-2, -1)) / denominator))
        if leakage > chain_tolerance:
            raise ValueError(
                f"M is not realizable in the retained input row space (relative leakage {leakage:g})"
            )
        total = np.sum((m - b) ** 2, axis=-1) / d
        visible = np.sum((m - projected) ** 2, axis=-1) / d
        error = float(
            np.max(np.abs(total - visible - invisible))
            / max(float(total.max()), np.finfo(float).tiny)
        )
        result.update(
            {
                "total_by_target": total.tolist(),
                "visible_by_target": visible.tolist(),
                "mean_total": float(total.mean()),
                "mean_visible": float(visible.mean()),
                "chain_leakage": leakage,
                "identity_relative_error": error,
            }
        )
    return result


def oracle_risk(a, b, noise, cutoff=1e-10):
    """Unrestricted linear oracle for z=A v+eta, Cov(v)=I/d, Cov(eta)=sigma²I.

    Noise is in the supplied input coordinate basis. Lag-difference coordinates
    generally do not have independent noise when they share a raw measurement.
    """
    a, b, singular, vectors, keep, projected = _inputs(a, b, cutoff)
    noise = finite_array(noise, "noise")
    if noise.ndim != 1 or not len(noise) or np.any(noise < 0) or np.any(np.diff(noise) <= 0):
        raise ValueError("noise must be a nonempty strictly increasing nonnegative vector")
    d = a.shape[-1]
    coordinates = b @ vectors.swapaxes(-2, -1)
    invisible = np.sum((b - projected) ** 2, axis=-1) / d
    risks = []
    for sigma in noise:
        if sigma == 0:
            remaining = np.zeros_like(singular)
        else:
            # Squared ratio only in the stable <=1 direction; avoid overflow.
            t = np.sqrt(d) * sigma
            large = singular > t
            remaining = np.zeros_like(singular)
            ratio = t / singular[large]
            remaining[large] = ratio * ratio / (1 + ratio * ratio)
            ratio = singular[~large] / t
            remaining[~large] = 1 / (1 + ratio * ratio)
            remaining *= keep
        # Null/removed directions retain their full risk at sigma=0 too.
        risk = invisible + np.sum(coordinates * coordinates * remaining[..., None, :], axis=-1) / d
        if not np.isfinite(risk).all():
            raise ValueError("Oracle calculation overflowed; rescale the supplied Jacobians")
        risks.append(risk)
    return {
        "report_version": 1,
        "kind": "noise-oracle",
        "noise": noise.tolist(),
        "risk_by_noise_and_target": np.stack(risks).tolist(),
        "mean_risk_by_noise": [float(r.mean()) for r in risks],
        "cutoff": float(cutoff),
        "rank": keep.sum(axis=-1).tolist(),
        "limits": [
            "Unrestricted pointwise linear oracle, not performance of a trained nonlinear predictor.",
            "Independent noise is assumed in the supplied A coordinates; specify raw measurement coordinates when required.",
        ],
    }
