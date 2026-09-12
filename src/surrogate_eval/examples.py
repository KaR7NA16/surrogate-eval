"""Small CPU examples independent of the historical L96 study.

They demonstrate interfaces and numerical checks, not new prediction methods.
"""

from pathlib import Path
import numpy as np

from .metrics import compare
from .repair import PairedFeatures, fit_heads, save_features, save_head
from .schema import (
    PairedReference,
    Prediction,
    save_reference,
    save_prediction,
    write_json,
    write_npz,
)

HORIZONS = np.array([0.05, 0.15, 0.3])
DELAY = 0.1


def linear_flow(x, time):
    """Exact damped oscillator flow, evaluated through a 2x2 eigendecomposition."""
    generator = np.array([[0.0, 1.0], [-1.0, -0.2]])
    eig, vectors = np.linalg.eig(generator)
    flow = (vectors @ np.diag(np.exp(eig * time)) @ np.linalg.inv(vectors)).real
    return x @ flow.T


def pendulum_flow(x, duration, dt=0.001):
    """Vectorized RK4 for q'=v, v'=-sin(q)-.15v, in radians."""
    count = round(duration / dt)
    if count < 0 or abs(count * dt - duration) > 1e-12:
        raise ValueError("duration must be a nonnegative integer number of steps")
    state = np.asarray(x, dtype=float).copy()

    def rhs(z):
        return np.stack([z[..., 1], -np.sin(z[..., 0]) - 0.15 * z[..., 1]], axis=-1)

    for _ in range(count):
        a = rhs(state)
        b = rhs(state + dt * a / 2)
        c = rhs(state + dt * b / 2)
        d = rhs(state + dt * c)
        state += dt * (a + 2 * b + 2 * c + d) / 6
    return state


def _trajectories(system, starts, directions, epsilon):
    states = np.stack(
        [starts, starts + epsilon * directions, starts - epsilon * directions], axis=1
    )
    flow = linear_flow if system == "linear" else pendulum_flow
    current = flow(states, DELAY)
    future = np.stack([flow(states, DELAY + h)[..., 0] for h in HORIZONS], axis=-1)
    previous = states[..., 0]
    q = current[..., 0]
    if system == "linear":
        features = np.stack([previous, q], axis=-1)
        names = np.array(["previous_position", "current_position"])
        spec = "damped linear oscillator positions at -0.1 and 0"
    else:
        features = np.stack([q, (q - previous) / DELAY, np.sin(q)], axis=-1)
        names = np.array(["current_angle", "history_secant_velocity", "sin_current_angle"])
        spec = "damped pendulum q(0), (q(0)-q(-0.1))/0.1, sin(q(0))"
    return future[:, :, None, :], features, names, spec, q


def make_demo(output, system="linear", seed=17):
    """Write train/validation/test files, fitted baselines and a held-out report.

    All choices are fixed before evaluation. Demo clusters group independent
    initial states for a fixed fitted model, not independent retraining runs.
    """
    if system not in ["linear", "pendulum"]:
        raise ValueError("system must be linear or pendulum")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Demo output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = {}
    epsilon = 0.001
    for split, n in [("train", 96), ("validation", 48), ("test", 64)]:
        starts = rng.normal(scale=0.8, size=(n, 2))
        directions = rng.normal(size=(n, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        data[split] = (*_trajectories(system, starts, directions, epsilon), starts, directions)
    scale = float(data["train"][4][:, 0].std())
    records = {}
    for split, (truth, x, names, spec, q, starts, directions) in data.items():
        n = len(truth)
        ids = np.array([f"{system}-{split}-{i}" for i in range(n)])
        clusters = np.array([f"{split}-group-{i % 4}" for i in range(n)])
        unit = "dimensionless position" if system == "linear" else "radian"
        ref = PairedReference(truth, epsilon, scale, HORIZONS, ids, clusters, unit)
        feat = PairedFeatures(x, ids, names, spec)
        save_reference(output / f"{split}_reference.npz", ref)
        save_features(output / f"{split}_features.npz", feat)
        records[split] = (ref, feat)
    models, fit_report = fit_heads(*records["train"], *records["validation"])
    # A saved selection precedes any scoring of the test split.
    write_json(output / "selection.json", fit_report)
    for name, model in models.items():
        save_head(output / f"{name}_head.npz", model)
    ref, feat = records["test"]
    q = data["test"][4]
    baseline = Prediction(
        np.broadcast_to(q[:, :, None, None], ref.reference.shape),
        ref.sample_ids,
        HORIZONS,
        ref.units,
    )
    predictions = {
        "persistence": baseline,
        "nominal_head": models["nominal"].predict(feat),
        "response_head": models["response"].predict(feat),
    }
    for name, prediction in predictions.items():
        save_prediction(output / f"{name}_predictions.npz", prediction)
    result = compare(ref, predictions, "persistence", seed=seed)
    result["example"] = {
        "system": system,
        "seed": seed,
        "scope": "Interface demonstration with fixed train/validation/test split; not a method-performance claim",
        "cluster_meaning": "four groups of independently sampled evaluation initial states, conditional on one training run",
    }
    write_json(output / "comparison.json", result)
    if system == "linear":
        flow = linear_flow(np.eye(2), DELAY).T
        a = np.stack([np.array([1.0, 0.0]), flow[0]])
        b = np.stack([linear_flow(np.eye(2), DELAY + h).T[0] for h in HORIZONS])
        write_npz(output / "geometry.npz", A=a, B=b, M=b)
    else:
        _, _, _, _, _, starts, directions = data["test"]
        base = pendulum_flow(starts[:4], DELAY + HORIZONS[-1])
        fine = pendulum_flow(starts[:4], DELAY + HORIZONS[-1], dt=0.0005)
        write_json(
            output / "numerical_check.json",
            {
                "step_refinement_max_absolute_state_difference": float(np.max(abs(base - fine))),
                "scope": "Example trajectory RK4 step refinement; not a derivative or global-model proof",
            },
        )
    return result
