"""Optional NumPy fixed-feature response-aware ridge baseline.

This is an established least-squares/Sobolev-style baseline, not a new algorithm.
The fitting interface consumes separate training and validation records only.
"""

from dataclasses import dataclass
import json

import numpy as np

from .metrics import evaluate
from .schema import (
    Prediction,
    finite_array,
    identifiers,
    positive_vector,
    write_npz,
)


@dataclass(frozen=True)
class PairedFeatures:
    """Features [sample,3 branches,feature], with declared order and construction."""

    features: np.ndarray
    sample_ids: np.ndarray
    feature_names: np.ndarray
    feature_spec: str

    def __post_init__(self):
        x = finite_array(self.features, "features")
        if x.ndim != 3 or x.shape[1] != 3 or not x.shape[0] or not x.shape[2]:
            raise ValueError("features must have shape [n>0,3 branches,features>0]")
        ids = identifiers(self.sample_ids, len(x), "sample_ids", unique=True)
        names = identifiers(self.feature_names, x.shape[-1], "feature_names", unique=True)
        if not isinstance(self.feature_spec, str) or not self.feature_spec.strip():
            raise ValueError("feature_spec must describe feature construction")
        for key, value in [("features", x), ("sample_ids", ids), ("feature_names", names)]:
            value.setflags(write=False)
            object.__setattr__(self, key, value)

    def aligned(self, reference):
        if not np.array_equal(self.sample_ids, reference.sample_ids):
            raise ValueError("feature/reference sample identity or order mismatch")
        return self.features


def save_features(path, features):
    write_npz(path, schema_version=np.array(1), **features.__dict__)


def load_features(path):
    with np.load(path, allow_pickle=False) as z:
        if z["schema_version"].shape != () or z["schema_version"].item() != 1:
            raise ValueError("Unsupported feature schema_version")
        return PairedFeatures(
            z["features"], z["sample_ids"], z["feature_names"], str(z["feature_spec"].item())
        )


@dataclass(frozen=True)
class HeadModel:
    coefficients: np.ndarray
    center: np.ndarray
    spread: np.ndarray
    target_scale: np.ndarray
    horizons: np.ndarray
    feature_names: np.ndarray
    feature_spec: str
    units: str
    target_ids: np.ndarray | None = None

    def __post_init__(self):
        coef = finite_array(self.coefficients, "coefficients")
        center = finite_array(self.center, "center")
        if center.ndim != 1 or not len(center):
            raise ValueError("center must be a nonempty feature vector")
        spread = positive_vector(self.spread, len(center), "spread")
        scale = finite_array(self.target_scale, "target_scale")
        h = finite_array(self.horizons, "horizons")
        if scale.ndim != 1 or not len(scale) or np.any(scale <= 0):
            raise ValueError("target_scale must be positive")
        if h.ndim != 1 or not len(h) or np.any(h <= 0) or np.any(np.diff(h) <= 0):
            raise ValueError("Invalid model horizons")
        if coef.shape != (len(center) + 1, len(scale) * len(h)):
            raise ValueError("Model coefficient dimensions disagree with its schema")
        names = identifiers(self.feature_names, len(center), "feature_names", unique=True)
        targets = identifiers(
            self.target_ids if self.target_ids is not None else np.arange(len(scale)),
            len(scale),
            "target_ids",
            unique=True,
        )
        if not self.feature_spec or not self.units:
            raise ValueError("Model feature specification and units are required")
        for key, value in [
            ("coefficients", coef),
            ("center", center),
            ("spread", spread),
            ("target_scale", scale),
            ("horizons", h),
            ("feature_names", names),
            ("target_ids", targets),
        ]:
            value.setflags(write=False)
            object.__setattr__(self, key, value)

    def predict(self, features):
        if features.feature_spec != self.feature_spec or not np.array_equal(
            features.feature_names, self.feature_names
        ):
            raise ValueError("Feature construction or coordinate order differs from fitted model")
        phi = _design(features.features, self.center, self.spread)
        pred = (phi @ self.coefficients).reshape(
            len(phi), 3, len(self.target_scale), len(self.horizons)
        )
        pred = pred * self.target_scale[None, None, :, None]
        return Prediction(pred, features.sample_ids, self.horizons, self.units, self.target_ids)


def _design(x, center, spread):
    z = (x - center) / spread
    return np.concatenate([np.ones_like(z[..., :1]), z], axis=-1)


def _weights(reference):
    clusters, counts = np.unique(reference.cluster_ids, return_counts=True)
    count = dict(zip(clusters, counts, strict=True))
    return np.array([1 / (len(clusters) * count[c]) for c in reference.cluster_ids])


def fit_heads(
    train,
    train_features,
    validation,
    validation_features,
    ridge=(1e-8, 1e-6, 1e-4, 0.01, 1),
    response_multipliers=(0, 0.1, 1, 10),
    nominal_cap=1.05,
):
    """Fit on train; freeze a nominal baseline and response-selected candidate on validation.

    Intercept is unpenalized. Normalizers and response-loss balancing use train
    only. Objective is equal-cluster squared normalized value/response error
    (sum over output columns), plus ridge. The selection cap is relative to the
    nominal-validation-selected zero-response-weight head, not a test guarantee.
    """
    x = train_features.aligned(train)
    validation_features.aligned(validation)
    if set(train.sample_ids) & set(validation.sample_ids):
        raise ValueError("Training and validation sample identities must be disjoint")
    train_scale = np.broadcast_to(train.scale, (len(train.reference), train.reference.shape[2]))
    val_scale = np.broadcast_to(
        validation.scale, (len(validation.reference), validation.reference.shape[2])
    )
    if (
        not np.all(train_scale == train_scale[0])
        or not np.all(val_scale == train_scale[0])
        or not np.array_equal(train.horizons, validation.horizons)
        or train.units != validation.units
        or not np.array_equal(train.target_ids, validation.target_ids)
    ):
        raise ValueError("Train/validation target scales, horizons and units must match")
    if train_features.feature_spec != validation_features.feature_spec or not np.array_equal(
        train_features.feature_names, validation_features.feature_names
    ):
        raise ValueError("Train/validation feature definitions must match")
    ridge = finite_array(ridge, "ridge")
    multipliers = finite_array(response_multipliers, "response_multipliers")
    if ridge.ndim != 1 or not len(ridge) or np.any(ridge < 0) or len(set(ridge)) != len(ridge):
        raise ValueError("ridge must be a nonempty distinct nonnegative grid")
    if (
        multipliers.ndim != 1
        or not len(multipliers)
        or 0 not in multipliers
        or np.any(multipliers < 0)
        or len(set(multipliers)) != len(multipliers)
    ):
        raise ValueError("response_multipliers must be distinct nonnegative values including zero")
    if not np.isfinite(nominal_cap) or nominal_cap < 1:
        raise ValueError("nominal_cap must be finite and at least 1")
    center = x[:, 0].mean(axis=0)
    spread = np.maximum(x[:, 0].std(axis=0), 1e-8)
    phi = _design(x, center, spread)
    target_scale = train_scale[0]
    y = (train.reference / target_scale[None, None, :, None]).reshape(len(x), 3, -1)
    eps = train.epsilon[:, None]
    dx = (phi[:, 1] - phi[:, 2]) / (2 * eps)
    dy = (y[:, 1] - y[:, 2]) / (2 * eps)
    sample_weights = _weights(train)
    value_energy = float(
        np.sum(sample_weights[:, None, None] * (y - y[:, 0].mean(axis=0)) ** 2) / (3 * y.shape[-1])
    )
    response_energy = float(np.sum(sample_weights[:, None] * dy**2) / dy.shape[-1])
    balance = value_energy / response_energy if response_energy > 0 else 1.0
    if not np.isfinite(balance):
        raise ValueError("Loss balancing is nonfinite; rescale training data")
    nominal_weight = np.repeat(np.sqrt(sample_weights / 3), 3)
    base_a = phi.reshape(-1, phi.shape[-1]) * nominal_weight[:, None]
    base_b = y.reshape(-1, y.shape[-1]) * nominal_weight[:, None]
    derivative_weight = np.sqrt(sample_weights)[:, None]
    penalty = np.eye(phi.shape[-1])
    penalty[0, 0] = 0
    candidates = []
    models = {}
    for multiplier in multipliers:
        for alpha in ridge:
            weight = float(multiplier * balance)
            a = np.concatenate(
                [base_a, np.sqrt(weight) * dx * derivative_weight, np.sqrt(alpha) * penalty]
            )
            b = np.concatenate(
                [
                    base_b,
                    np.sqrt(weight) * dy * derivative_weight,
                    np.zeros((len(penalty), y.shape[-1])),
                ]
            )
            coef, _, rank, _ = np.linalg.lstsq(a, b, rcond=1e-12)
            model = HeadModel(
                coef,
                center,
                spread,
                target_scale,
                train.horizons,
                train_features.feature_names,
                train_features.feature_spec,
                train.units,
                train.target_ids,
            )
            cid = f"candidate-{len(candidates)}"
            scores = evaluate(validation, model.predict(validation_features))
            models[cid] = model
            candidates.append(
                {
                    "id": cid,
                    "ridge": float(alpha),
                    "response_multiplier": float(multiplier),
                    "response_weight": weight,
                    "solve_rank": int(rank),
                    "validation": scores["aggregate"],
                }
            )
    nominal = min(
        (c for c in candidates if c["response_multiplier"] == 0),
        key=lambda c: c["validation"]["nominal_mse"],
    )
    cap = nominal_cap * nominal["validation"]["nominal_mse"]
    eligible = [c for c in candidates if c["validation"]["nominal_mse"] <= cap]
    chosen = min(eligible, key=lambda c: c["validation"]["response_mse"])
    report = {
        "report_version": 1,
        "kind": "head-fit",
        "nominal_candidate": nominal["id"],
        "selected_candidate": chosen["id"],
        "nominal_validation_cap": float(cap),
        "nominal_cap_ratio": float(nominal_cap),
        "training_value_energy": value_energy,
        "training_response_energy": response_energy,
        "response_balance": balance,
        "candidates": candidates,
        "limits": [
            "Known fixed-feature ridge baseline; no claimed new optimization method.",
            "Only training/validation data were used by this API. The caller must keep evaluation data separate.",
            "Validation nominal cap does not guarantee a cap on evaluation error.",
            "The loss sums output columns; changing target count changes effective regularization.",
        ],
    }
    return {"nominal": models[nominal["id"]], "response": models[chosen["id"]]}, report


def save_head(path, model, metadata=None):
    write_npz(
        path,
        schema_version=np.array(1),
        **model.__dict__,
        metadata=np.array(json.dumps(metadata or {}, allow_nan=False)),
    )


def load_head(path):
    with np.load(path, allow_pickle=False) as z:
        if z["schema_version"].shape != () or z["schema_version"].item() != 1:
            raise ValueError("Unsupported model schema_version")
        fields = {k: z[k].copy() for k in HeadModel.__dataclass_fields__}
    for key in ["feature_spec", "units"]:
        fields[key] = str(fields[key].item())
    return HeadModel(**fields)
