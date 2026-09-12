# Evaluate an external model

Generate a nominal reference trajectory and physically matched plus/minus perturbations for each evaluation row. Feed the corresponding inputs to your already fixed surrogate. Convert predictions back to physical units before scoring. Keep model development, scale estimation and final evaluation separate.

The runnable example below illustrates the interchange with two targets and two horizons. Its constructed predictions are a schema example, not a scientific benchmark.

```python
from pathlib import Path
import numpy as np
from surrogate_eval import PairedReference, Prediction, evaluate
from surrogate_eval.schema import save_reference, save_prediction, write_json

out = Path("outputs/external-model")  # must be a new output location
n, d, h = 12, 2, 2
rng = np.random.default_rng(20)
nominal = rng.normal(size=(n, d, h))
directional_response = rng.normal(size=(n, d, h))
epsilon = 0.01
truth = np.stack([nominal, nominal + epsilon * directional_response,
                  nominal - epsilon * directional_response], axis=1)
ref = PairedReference(
    reference=truth, epsilon=epsilon, scale=np.array([1.0, 2.0]),
    horizons=np.array([0.1, 0.2]), sample_ids=np.arange(n),
    cluster_ids=np.repeat(np.arange(3), 4), units="example state units",
    target_ids=np.array(["position", "velocity"]),
)
pred = Prediction(0.9 * truth, ref.sample_ids, ref.horizons, ref.units, ref.target_ids)
save_reference(out / "reference.npz", ref)
save_prediction(out / "model.npz", pred)
write_json(out / "score.json", evaluate(ref, pred))
```

For a real integration, replace `truth` with trusted simulator/experimental outputs and `0.9 * truth` with model outputs. Train-derived target scales must be supplied explicitly. Save each candidate prediction file against exactly the same rows; then use:

```sh
surrogate-eval score --reference outputs/external-model/reference.npz --prediction outputs/external-model/model.npz --output outputs/external-model/cli-score.json --markdown outputs/external-model/score.md
```

Use `surrogate-eval compare --help` for multiple candidates. Reports include each target/horizon, equal-weight cluster aggregates and paired cluster comparisons. Never interpret an absent label as a zero response. A time series without matched perturbation reference outputs cannot identify response error through this API.
