"""Versioned, identity-checked reference and prediction interchange.

Reference and prediction values are physical units with shape
``(sample, branch, target, horizon)``; branch order is nominal, plus, minus.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def finite_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values; no cases may be dropped")
    return array


def identifiers(value, n, name, unique=False):
    array = np.asarray(value)
    if array.shape != (n,) or array.dtype.kind not in "USiu":
        raise ValueError(f"{name} must be a one-dimensional string/integer array of length {n}")
    array = array.astype(str)
    if any(not x.strip() for x in array):
        raise ValueError(f"{name} contains an empty identifier")
    if unique and len(set(array)) != n:
        raise ValueError(f"{name} must be unique")
    return array


def positive_vector(value, n, name):
    array = finite_array(value, name)
    if array.ndim == 0:
        array = np.full(n, float(array))
    if array.shape != (n,) or np.any(array <= 0):
        raise ValueError(f"{name} must be positive, scalar or length {n}")
    return array


@dataclass(frozen=True)
class PairedReference:
    """Truth and protocol; scale must be fixed independently of evaluation targets.

    ``epsilon`` is scalar or one amplitude per sample. ``scale`` is scalar,
    one scale per target, or [sample,target] for independently normalized datasets.
    ``cluster_ids`` identify independent replication units;
    the software cannot infer independence from the identifiers.
    """

    reference: np.ndarray
    epsilon: np.ndarray
    scale: np.ndarray
    horizons: np.ndarray
    sample_ids: np.ndarray
    cluster_ids: np.ndarray
    units: str = "unspecified"
    target_ids: np.ndarray | None = None

    def __post_init__(self):
        values = finite_array(self.reference, "reference")
        if values.ndim != 4 or values.shape[1] != 3 or any(n == 0 for n in values.shape):
            raise ValueError("reference must have shape [n>0,3 branches,targets>0,horizons>0]")
        n, _, targets, h = values.shape
        times = finite_array(self.horizons, "horizons")
        if times.shape != (h,) or np.any(times <= 0) or np.any(np.diff(times) <= 0):
            raise ValueError(
                "horizons must be positive and strictly increasing, one per output horizon"
            )
        scales = finite_array(self.scale, "scale")
        if scales.ndim == 0:
            scales = np.full(targets, float(scales))
        if scales.shape not in [(targets,), (n, targets)] or np.any(scales <= 0):
            raise ValueError("scale must be positive scalar, [target], or [sample,target]")
        normalized = {
            "reference": values,
            "epsilon": positive_vector(self.epsilon, n, "epsilon"),
            "scale": scales,
            "horizons": times,
            "sample_ids": identifiers(self.sample_ids, n, "sample_ids", unique=True),
            "cluster_ids": identifiers(self.cluster_ids, n, "cluster_ids"),
            "target_ids": identifiers(
                self.target_ids if self.target_ids is not None else np.arange(targets),
                targets,
                "target_ids",
                unique=True,
            ),
        }
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a nonempty string")
        for key, array in normalized.items():
            array.setflags(write=False)
            object.__setattr__(self, key, array)


@dataclass(frozen=True)
class Prediction:
    """Physical-unit predictions with identity and output-horizon metadata."""

    predictions: np.ndarray
    sample_ids: np.ndarray
    horizons: np.ndarray
    units: str = "unspecified"
    target_ids: np.ndarray | None = None

    def __post_init__(self):
        array = finite_array(self.predictions, "predictions")
        if array.ndim != 4 or array.shape[1] != 3 or any(n == 0 for n in array.shape):
            raise ValueError("predictions must have shape [n>0,3 branches,targets>0,horizons>0]")
        ids = identifiers(self.sample_ids, len(array), "sample_ids", unique=True)
        horizons = finite_array(self.horizons, "horizons")
        if (
            horizons.shape != (array.shape[-1],)
            or np.any(horizons <= 0)
            or np.any(np.diff(horizons) <= 0)
        ):
            raise ValueError("prediction horizons must be positive and strictly increasing")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a nonempty string")
        target_ids = identifiers(
            self.target_ids if self.target_ids is not None else np.arange(array.shape[2]),
            array.shape[2],
            "target_ids",
            unique=True,
        )
        for key, value in [
            ("predictions", array),
            ("sample_ids", ids),
            ("horizons", horizons),
            ("target_ids", target_ids),
        ]:
            value.setflags(write=False)
            object.__setattr__(self, key, value)

    def aligned(self, reference):
        if self.predictions.shape != reference.reference.shape:
            raise ValueError("prediction/reference shapes differ")
        if not np.array_equal(self.sample_ids, reference.sample_ids):
            raise ValueError("sample identity/order mismatch; align explicitly before scoring")
        if not np.array_equal(self.horizons, reference.horizons):
            raise ValueError("prediction/reference horizons differ")
        if not np.array_equal(self.target_ids, reference.target_ids):
            raise ValueError("prediction/reference target identities or order differ")
        if self.units != reference.units:
            raise ValueError("prediction/reference physical units differ")
        return self.predictions


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path, required):
    with np.load(path, allow_pickle=False) as file:
        missing = required - set(file.files)
        if missing:
            raise ValueError(f"Missing NPZ fields: {', '.join(sorted(missing))}")
        version = np.asarray(file["schema_version"])
        if (
            version.shape != ()
            or version.dtype.kind not in "iu"
            or version.item() != SCHEMA_VERSION
        ):
            raise ValueError("Unsupported schema_version; expected integer 1")
        data = {key: file[key].copy() for key in required - {"schema_version"}}
    units = data["units"]
    if units.shape != () or units.dtype.kind not in "US":
        raise ValueError("units must be a scalar string")
    data["units"] = str(units.item())
    return data


def load_reference(path):
    fields = {
        "schema_version",
        "reference",
        "epsilon",
        "scale",
        "horizons",
        "sample_ids",
        "cluster_ids",
        "units",
        "target_ids",
    }
    return PairedReference(**_read(path, fields))


def load_prediction(path):
    fields = {"schema_version", "predictions", "sample_ids", "horizons", "units", "target_ids"}
    return Prediction(**_read(path, fields))


def write_npz(path, **arrays):
    """Write a new NPZ, refusing overwrite and ambiguous automatic extensions."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("NPZ output must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def save_reference(path, reference):
    write_npz(path, schema_version=np.array(SCHEMA_VERSION), **reference.__dict__)


def save_prediction(path, prediction):
    write_npz(path, schema_version=np.array(SCHEMA_VERSION), **prediction.__dict__)


def write_json(path, value):
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
