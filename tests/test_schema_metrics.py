from dataclasses import replace
import json

import numpy as np
import pytest

from surrogate_eval import Prediction, compare, evaluate
from surrogate_eval.schema import (
    load_prediction,
    load_reference,
    save_prediction,
    save_reference,
    write_npz,
)


def test_exact_errors_and_heterogeneous_epsilon(paired):
    ref, pred = paired
    result = evaluate(ref, pred)
    assert result["aggregate"]["nominal_mse"] == pytest.approx(1)
    assert result["aggregate"]["response_mse"] == pytest.approx(0.625)
    assert result["aggregate"]["response_by_horizon"] == pytest.approx([0.25, 1])
    assert result["aggregate"]["decision_regret_by_horizon"] == [0.0, 0.0]
    json.dumps(result, allow_nan=False)


def test_cluster_weighting_is_not_row_weighting(paired):
    ref, pred = paired
    ref = replace(ref, cluster_ids=np.array(["a", "b", "b", "b", "b", "b"]))
    out = ref.reference.copy()
    out[0] += 1
    out[1:] += 2
    result = evaluate(ref, replace(pred, predictions=out))
    assert result["aggregate"]["nominal_mse"] == pytest.approx(2.5)
    assert result["n_clusters"] == 2


def test_units_and_per_sample_scales(paired):
    ref, pred = paired
    expected = evaluate(ref, pred)["aggregate"]
    factor = np.arange(1, 7, dtype=float)[:, None, None, None]
    scaled = replace(ref, reference=ref.reference * factor, scale=factor[:, 0, :, 0])
    got = evaluate(scaled, replace(pred, predictions=pred.predictions * factor))["aggregate"]
    for key in expected:
        np.testing.assert_allclose(got[key], expected[key], atol=1e-10)


@pytest.mark.parametrize(
    "field,value",
    [
        ("epsilon", 0),
        ("epsilon", -1),
        ("scale", 0),
        ("scale", float("nan")),
        ("horizons", [0.7, 0.2]),
        ("horizons", [0.2, 0.2]),
        ("units", ""),
        ("sample_ids", np.array(["x"] * 6)),
        ("cluster_ids", np.array([""] * 6)),
    ],
)
def test_invalid_reference_metadata(paired, field, value):
    with pytest.raises(ValueError):
        replace(paired[0], **{field: value})


def test_nonfinite_and_complex_rejected(paired):
    ref, pred = paired
    with pytest.raises(ValueError):
        replace(pred, predictions=np.full_like(pred.predictions, np.inf))
    with pytest.raises(ValueError):
        replace(ref, reference=ref.reference.astype(complex))


def test_sample_and_protocol_mismatch_rejected(paired):
    ref, pred = paired
    for bad in [
        replace(pred, sample_ids=pred.sample_ids[::-1]),
        replace(pred, horizons=[0.3, 0.8]),
        replace(pred, units="meters"),
        replace(pred, target_ids=np.array(["a-different-target"])),
    ]:
        with pytest.raises(ValueError):
            evaluate(ref, bad)


def test_roundtrip_and_refuse_overwrite(tmp_path, paired):
    ref, pred = paired
    save_reference(tmp_path / "reference.npz", ref)
    save_prediction(tmp_path / "prediction.npz", pred)
    result = evaluate(
        load_reference(tmp_path / "reference.npz"), load_prediction(tmp_path / "prediction.npz")
    )
    assert result == evaluate(ref, pred)
    with pytest.raises(FileExistsError):
        save_reference(tmp_path / "reference.npz", ref)
    with pytest.raises(ValueError):
        save_prediction(tmp_path / "wrong.npy", pred)


def test_bad_version_and_pickle_rejected(tmp_path, paired):
    path = tmp_path / "data.npz"
    write_npz(path, schema_version=np.array(1.0), **paired[0].__dict__)
    with pytest.raises(ValueError, match="schema_version"):
        load_reference(path)
    obj = tmp_path / "object.npz"
    write_npz(obj, schema_version=np.array(1), reference=np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        load_reference(obj)


def test_zero_baseline_and_reproducible_uncertainty(paired):
    ref, pred = paired
    exact = Prediction(ref.reference, ref.sample_ids, ref.horizons, ref.units)
    a = compare(ref, {"truth": exact, "biased": pred}, "truth", seed=42)
    assert a == compare(ref, {"truth": exact, "biased": pred}, "truth", seed=42)
    metric = a["comparisons"][0]["response_mse"]
    assert metric["relative_reduction"] is None
    assert metric["absolute_reduction"] == pytest.approx(-0.625)
    assert metric["relative_reduction_interval_95"] is None
    b = compare(ref, {"truth": exact, "biased": pred}, "biased", seed=42)
    assert b["comparisons"][0]["response_mse"]["relative_reduction"] == pytest.approx(1)


def test_one_cluster_has_no_interval(paired):
    ref, pred = paired
    ref = replace(ref, cluster_ids=np.array(["one"] * 6))
    result = compare(ref, {"a": pred, "b": pred}, "a")
    assert result["comparisons"][0]["nominal_mse"]["absolute_reduction_interval_95"] is None


def test_duplicate_or_absent_baseline_invalid(paired):
    ref, pred = paired
    with pytest.raises(ValueError):
        compare(ref, {"a": pred}, "a")
    with pytest.raises(ValueError):
        compare(ref, {"a": pred, "b": pred}, "missing")


def test_numeric_overflow_is_not_returned_as_a_score(paired):
    ref, pred = paired
    huge = replace(pred, predictions=np.full_like(pred.predictions, 1e200))
    with pytest.raises(ValueError, match="overflowed"):
        evaluate(ref, huge)
