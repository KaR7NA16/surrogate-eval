# Optional fixed-feature response baseline

`fit_heads` fits an established response-aware ridge baseline with NumPy least squares. Supply separate `PairedReference` and `PairedFeatures` training and validation records. Sample IDs must be disjoint, output scales/horizons/units/target IDs must match, and features must have the same names and construction specification. This fitter requires a common per-target scale across rows; general scoring supports per-row scales too.

Feature center and spread use nominal training inputs only; spread is floored at 1e-8. An intercept is added and left unpenalized. Physical targets are divided by the supplied scales. The objective combines equal-cluster squared errors across all three value branches, central-response squared error, and ridge penalty, summing output columns. A training-only energy ratio balances response and value terms. Increasing output dimension changes the effective ridge strength; this is recorded rather than hidden.

The zero-response-weight candidate with smallest nominal validation MSE is the nominal baseline. Among all candidates satisfying nominal validation MSE ≤ `nominal_cap` times that baseline, select the smallest response validation MSE. Default cap: 1.05. Grid ties keep first occurrence. Save both selected models and the full candidate report before obtaining final test scores. The validation cap does not guarantee a test cap.

```sh
surrogate-eval fit-head --train-reference train-reference.npz --train-features train-features.npz --validation-reference validation-reference.npz --validation-features validation-features.npz --output response-head.npz --nominal-output nominal-head.npz --report selection.json
surrogate-eval predict-head --model response-head.npz --features test-features.npz --output test-prediction.npz
```

`surrogate-eval demo` runs this complete pattern on generated linear or pendulum data. Caller responsibility includes leakage-free feature construction and preserving an untouched evaluation set. The fitter cannot identify hidden test leakage inside supplied features. It implements no automatic neural-network training or Jacobian estimation, and is not a new optimization algorithm.
