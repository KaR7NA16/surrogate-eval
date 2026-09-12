from dataclasses import replace
import numpy as np
import pytest

from surrogate_eval import PairedReference, evaluate
from surrogate_eval.repair import PairedFeatures, fit_heads, load_head, save_head


def data(n, seed, prefix):
    rng = np.random.default_rng(seed)
    nominal = rng.normal(size=(n, 2))
    direction = rng.normal(size=(n, 2))
    eps = np.linspace(0.001, 0.002, n)
    features = np.stack(
        [nominal, nominal + eps[:, None] * direction, nominal - eps[:, None] * direction], axis=1
    )
    coefficients = np.array([[2.0, -0.3, 0.7, 0.2], [0.4, 1.0, -0.2, 0.5]])
    truth = (features @ coefficients + np.array([0.2, -0.1, 0.3, 0.1])).reshape(n, 3, 2, 2)
    ids = np.array([f"{prefix}{i}" for i in range(n)])
    ref = PairedReference(
        truth,
        eps,
        [1.3, 0.7],
        [0.1, 0.4],
        ids,
        np.array([f"g{i % 3}" for i in range(n)]),
        "dimensionless",
    )
    x = PairedFeatures(features, ids, np.array(["a", "b"]), "two measured coordinates")
    return ref, x


def test_exact_fit_predict_and_reload(tmp_path):
    train, tx = data(24, 1, "train")
    val, vx = data(12, 2, "val")
    test, xx = data(12, 3, "test")
    models, report = fit_heads(train, tx, val, vx, ridge=[0], response_multipliers=[0, 1])
    pred = models["response"].predict(xx)
    assert evaluate(test, pred)["aggregate"]["response_mse"] < 1e-20
    save_head(tmp_path / "head.npz", models["response"], report)
    np.testing.assert_array_equal(
        load_head(tmp_path / "head.npz").predict(xx).predictions, pred.predictions
    )


def test_normalization_and_fit_do_not_use_validation_values():
    train, tx = data(24, 1, "train")
    val, vx = data(12, 2, "val")
    first, _ = fit_heads(train, tx, val, vx, ridge=[0.1], response_multipliers=[0])
    second, _ = fit_heads(
        train,
        tx,
        replace(val, reference=val.reference * 100),
        replace(vx, features=vx.features + 100),
        ridge=[0.1],
        response_multipliers=[0],
    )
    np.testing.assert_array_equal(first["nominal"].center, tx.features[:, 0].mean(axis=0))
    np.testing.assert_array_equal(first["nominal"].coefficients, second["nominal"].coefficients)


def test_selected_coefficient_matches_independent_normal_equations():
    train, tx = data(24, 5, "train")
    val, vx = data(12, 6, "val")
    models, report = fit_heads(train, tx, val, vx, ridge=[0.1], response_multipliers=[0, 1])
    model = models["response"]
    selected = next(c for c in report["candidates"] if c["id"] == report["selected_candidate"])
    x = (tx.features - model.center) / model.spread
    phi = np.concatenate([np.ones_like(x[..., :1]), x], axis=-1)
    y = (train.reference / train.scale[None, None, :, None]).reshape(24, 3, 4)
    dx = (phi[:, 1] - phi[:, 2]) / (2 * train.epsilon[:, None])
    dy = (y[:, 1] - y[:, 2]) / (2 * train.epsilon[:, None])
    x0, y0 = phi.reshape(-1, 3), y.reshape(-1, 4)
    gram = (
        x0.T @ x0 / len(x0)
        + selected["response_weight"] * dx.T @ dx / len(dx)
        + 0.1 * np.diag([0.0, 1, 1])
    )
    rhs = x0.T @ y0 / len(x0) + selected["response_weight"] * dx.T @ dy / len(dx)
    np.testing.assert_allclose(model.coefficients, np.linalg.solve(gram, rhs), atol=1e-12)


def test_inconsistent_split_and_features_fail():
    train, tx = data(24, 1, "train")
    val, vx = data(12, 2, "val")
    with pytest.raises(ValueError, match="disjoint"):
        fit_heads(train, tx, train, tx)
    with pytest.raises(ValueError, match="feature"):
        fit_heads(train, tx, val, replace(vx, feature_names=np.array(["b", "a"])))
    with pytest.raises(ValueError):
        fit_heads(train, tx, val, vx, nominal_cap=0.99)
