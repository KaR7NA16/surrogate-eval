import numpy as np
import pytest

from surrogate_eval import PairedReference, Prediction


@pytest.fixture
def paired():
    n = 6
    eps = np.linspace(0.001, 0.006, n)
    base = np.arange(n, dtype=float)[:, None, None]
    response = np.broadcast_to(np.array([1.0, 2.0])[None, None, :], (n, 1, 2))
    nominal = np.broadcast_to(base, (n, 1, 2))
    truth = np.stack(
        [nominal, nominal + eps[:, None, None] * response, nominal - eps[:, None, None] * response],
        axis=1,
    )
    ref = PairedReference(
        truth,
        eps,
        1.0,
        [0.2, 0.7],
        np.array([f"s{i}" for i in range(n)]),
        np.array(["a", "a", "b", "b", "c", "c"]),
        "position",
    )
    wrong = np.stack(
        [
            nominal + 1,
            nominal + 1 + 0.5 * eps[:, None, None] * response,
            nominal + 1 - 0.5 * eps[:, None, None] * response,
        ],
        axis=1,
    )
    return ref, Prediction(wrong, ref.sample_ids, ref.horizons, ref.units)
