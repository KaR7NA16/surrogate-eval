# Python and command-line API

| Module | Public entry points | Contract |
|---|---|---|
| `response_fidelity` | `PairedReference`, `Prediction`, `evaluate`, `compare`, `load_reference`, `load_prediction` | Versioned paired scoring |
| `schema` | `save_reference`, `save_prediction`, `write_json`, `sha256` | Validated interchange and exclusive writes |
| `geometry` | `decompose(a,b,predicted=None,cutoff=1e-10,chain_tolerance=1e-8)` | Reference Jacobians in shared state basis |
| `geometry` | `oracle_risk(a,b,noise,cutoff=1e-10)` | Increasing nonnegative noise vector |
| `repair` | `PairedFeatures`, `fit_heads`, `HeadModel` | Train/validation fitting and physical predictions |
| `repair` | `save_features`, `load_features`, `save_head`, `load_head` | Optional model interchange |
| `legacy` | `verify_evidence`, `export_l96` | Complete hash-matched optional evidence |

`evaluate(reference, prediction)` returns a JSON-compatible dictionary. `compare(reference, predictions, baseline, bootstrap_samples=2000, seed=0)` accepts a mapping from names to Prediction instances and returns reports plus paired comparisons. Input arrays are copied and made read-only by record constructors. Invalid scientific inputs generally raise `ValueError`; writes to existing paths raise `FileExistsError`.

`rfl --help` lists workflows; `rfl COMMAND --help` gives exact arguments. CLI errors handled by the interface return status 2 with a short message; success returns 0. `python -m response_fidelity` is equivalent to `rfl`. Reports and NPZ schemas are version 1; this alpha release does not promise stable internal underscore-prefixed functions.
