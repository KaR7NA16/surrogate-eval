import numpy as np
import pytest

from surrogate_eval.geometry import decompose, oracle_risk


def test_known_orthogonal_decomposition():
    a = np.array([[1.0, 0, 0], [0, 1, 0]])
    b = np.array([[1.0, 2, 3]])
    m = np.array([[1.0, 1, 0]])
    out = decompose(a, b, m)
    assert out["mean_total"] == pytest.approx(10 / 3)
    assert out["mean_invisible"] == pytest.approx(3)
    assert out["mean_visible"] == pytest.approx(1 / 3)
    assert out["identity_relative_error"] < 1e-14


def test_zero_and_repeated_rows_have_no_spurious_directions():
    a = np.zeros((3, 5))
    b = np.eye(5)[:2]
    risk = oracle_risk(a, b, [0, 0.001, 1])
    np.testing.assert_allclose(risk["risk_by_noise_and_target"], np.full((3, 2), 0.2))
    repeated = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    out = decompose(repeated, np.eye(3))
    assert out["rank"] == 1
    assert out["invisible_by_target"] == pytest.approx([0, 1 / 3, 1 / 3])


def test_batched_oracle_matches_direct_regularized_solution():
    rng = np.random.default_rng(91)
    a = rng.normal(size=(4, 3, 5))
    b = rng.normal(size=(4, 2, 5))
    out = oracle_risk(a, b, [0, 0.01, 0.2])
    expected = []
    for sigma in [0, 0.01, 0.2]:
        if sigma == 0:
            coefficients = b @ np.linalg.pinv(a)
        else:
            gram = a @ a.swapaxes(-2, -1) + 5 * sigma**2 * np.eye(3)
            coefficients = np.linalg.solve(gram, a @ b.swapaxes(-2, -1)).swapaxes(-2, -1)
        expected.append(
            np.sum((coefficients @ a - b) ** 2, axis=-1) / 5
            + sigma**2 * np.sum(coefficients**2, axis=-1)
        )
    np.testing.assert_allclose(out["risk_by_noise_and_target"], expected, atol=1e-12)
    assert np.all(np.diff(out["mean_risk_by_noise"]) >= 0)


def test_cutoff_and_chain_rule_contract():
    a = np.diag([1.0, 1e-11])
    b = np.eye(2)
    assert decompose(a, b)["rank"] == 1
    assert decompose(a, b, cutoff=1e-12)["rank"] == 2
    with pytest.raises(ValueError, match="realizable"):
        decompose(a, b, b)
    with pytest.raises(ValueError):
        oracle_risk(a, b, [-1, 0])
    with pytest.raises(ValueError):
        decompose(a, b, cutoff=float("nan"))


def test_invertible_change_of_input_basis_preserves_noiseless_projection():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(2, 4))
    b = rng.normal(size=(3, 4))
    t = np.array([[2.0, 0.3], [0.0, 0.5]])
    first = decompose(a, b)
    second = decompose(t @ a, b)
    assert first["invisible_by_target"] == pytest.approx(second["invisible_by_target"], abs=1e-12)


def test_large_batch_member_cannot_hide_an_invalid_small_member():
    a = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
    b = np.array([[[1e12, 0.0]], [[1.0, 0.0]]])
    m = b.copy()
    m[1, 0, 1] = 0.1
    with pytest.raises(ValueError, match="realizable"):
        decompose(a, b, m)


def test_nearly_visible_risk_is_not_lost_to_subtraction():
    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 1e-12]])
    risk = oracle_risk(a, b, [0])
    assert risk["mean_risk_by_noise"][0] == pytest.approx(5e-25, rel=1e-12, abs=0)
