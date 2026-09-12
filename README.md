# SurrogateEval

**Evaluate predictions. Examine responses. Make model comparisons reproducible.**

A research toolkit for evaluating the **perturbation-response fidelity of scientific surrogate models**—from paired-output scoring to local derivative diagnostics and response-aware baseline fitting.

[中文](README.zh-CN.md) · [Quick start](#quick-start) · [Bring your own model](docs/your_model.md) · [Methods](docs/methods.md) · [Research case](docs/evidence.md)

**Python 3.11+ · NumPy core · CLI + Python API · Version 0.1.0 / Alpha**

## Why response fidelity?

A surrogate may predict a trajectory accurately while responding incorrectly to a change in its input. For scientific workflows that use perturbations to investigate sensitivity or compare interventions, prediction error alone leaves an essential part of model behavior unmeasured.

SurrogateEval evaluates the nominal prediction and the response to matched positive/negative perturbations together. It helps researchers investigate:

- **Response accuracy:** Does the model reproduce the reference response, across targets and forecast horizons?
- **Model comparison:** Does an apparent improvement persist across the supplied replication groups?
- **Local information limits:** With reference Jacobians available, how much target sensitivity lies outside the retained input row space?
- **Training trade-offs:** Does response-aware fitting improve held-out response error while preserving nominal accuracy?

The workflow accepts predictions from an existing model. Basic evaluation requires physical reference outputs and matching predictions; derivative diagnostics are optional.

## What you can do

| Capability | Research use | Interface |
|---|---|---|
| **Paired-response evaluation** | Measure nominal and finite-response error by target, horizon and replication group | `surrogate-eval score` |
| **Paired model comparison** | Compare candidates against a baseline with cluster-level results and descriptive bootstrap intervals | `surrogate-eval compare` |
| **Local tangent diagnostics** | Decompose realizable surrogate tangent error into visible and invisible components | `surrogate-eval diagnose` |
| **Noise sensitivity analysis** | Evaluate a pointwise linear oracle under a declared input-noise model | `surrogate-eval noise` |
| **Response-aware baseline fitting** | Fit and select fixed-feature ridge heads using separate training and validation records | `surrogate-eval fit-head`, `surrogate-eval predict-head` |
| **Optional case verification** | Verify and convert the frozen Lorenz–96 evidence into the common evaluation format | `surrogate-eval verify-evidence`, `surrogate-eval l96-export` |

### Designed for inspectable scientific comparisons

**Explicit physical pairing.** Versioned records specify nominal/plus/minus branches, perturbation amplitudes, target scales, sample identities, units and forecast horizons. Mismatched identities or nonfinite values fail validation.

**Replication-aware aggregation.** Supplied clusters receive equal aggregate weight, including when their row counts differ. Network initializations and spatial coordinates need not be treated as additional independent experiments.

**Traceable outputs.** JSON results support further analysis; Markdown reports support review. File-based scoring and fitting record input hashes and software version. Writers refuse to overwrite existing results.

**Lightweight integration.** The core requires NumPy only. Use the CLI for files or the Python API in an existing analysis pipeline. Neither a GPU nor the original study is needed for general evaluation.

## Quick start

From the repository root, install the package and run either self-contained CPU example:

```sh
python -m pip install .
surrogate-eval demo --system linear --output outputs/linear
surrogate-eval demo --system pendulum --output outputs/pendulum
```

Open `outputs/linear/comparison.md` or `outputs/pendulum/comparison.md`. Each example generates separate training, validation and test records, fits baseline heads, freezes validation selection and evaluates on held-out initial states.

The linear example also provides Jacobians for the optional diagnostics:

```sh
surrogate-eval diagnose --input outputs/linear/geometry.npz --output outputs/linear/geometry.json --markdown outputs/linear/geometry.md
surrogate-eval noise --input outputs/linear/geometry.npz --sigma 0 0.001 0.01 --output outputs/linear/noise.json
```

Use a new output directory when rerunning. `python -m surrogate_eval` is equivalent to `surrogate-eval`.

## Bring your own model

Export physically matched reference and predicted outputs in the common shape:

```text
[evaluation row, nominal / plus / minus, target, forecast horizon]
```

Then score one model or compare multiple candidates:

```sh
surrogate-eval score --reference reference.npz --prediction model.npz --output score.json --markdown score.md
surrogate-eval compare --reference reference.npz --model baseline=baseline.npz --model candidate=model.npz --baseline baseline --output comparison.json --markdown comparison.md
```

The Python interface uses the same validated records:

```python
from surrogate_eval import load_reference, load_prediction, evaluate

reference = load_reference("reference.npz")
prediction = load_prediction("model.npz")
report = evaluate(reference, prediction)
print(report["aggregate"]["response_mse"])
```

Start with the [runnable external-model tutorial](docs/your_model.md), then consult the [data format](docs/data_format.md) and [API reference](docs/api.md). Reference outputs for matched perturbations are required; ordinary time-series data alone do not provide response-error labels.

## Research-case verification

The toolkit has been checked against a frozen, partially observed Lorenz–96 research case. Verification follows the data from the original evidence through saved-model prediction reconstruction to the generic comparison report:

| Verification | What is checked |
|---|---|
| Evidence integrity | SHA-256 verification of all 119 manifest-bound files before export |
| Prediction reconstruction | Saved-model predictions reconstructed through the NumPy adapter |
| Metric agreement | The F24 history-input regression test matches the archived median cluster response reduction to an absolute tolerance of 1e-12 |
| Replication structure | The same regression test checks that the exported comparison retains eight dataset clusters |

The [verification record](docs/evidence.md) separates historical-case checks from generic numerical and data-contract tests. The original evidence bundle is external; [case verification and export](reproduction/README.md) require access to it. Both bundled CPU examples run independently of that bundle.

## Methodological scope

SurrogateEval reports finite perturbation errors, not unmeasured full-Jacobian accuracy. Local geometry assumes supplied, compatible reference derivatives; a pointwise linear oracle does not establish global learnability. Bootstrap intervals are descriptive for fixed models and supplied clusters. Physical pairing, training-only scales and cluster independence remain the researcher's responsibility.

Response-aware ridge fitting uses established methods. The project's contribution is a reusable evaluation workflow with explicit data, aggregation and provenance conventions. Read the [method definitions](docs/methods.md), [fitting procedure](docs/repair.md) and [related work](docs/related_work.md) for details.

## Verification and development

Local validation on Windows / Python 3.14 passed **38 tests**, including the optional historical-case checks. The 119 manifest-bound evidence files verified successfully. A clean wheel installation also ran both demos, the external-model tutorial and the geometry/noise workflows with NumPy as its only runtime dependency. These are implementation checks, distinct from scientific validation across new systems.

```sh
python -m pip install ".[dev]"
python scripts/check.py
python scripts/check.py --legacy --evidence-root PATH_TO_EVIDENCE
python -m build
python scripts/check_distribution.py
```

[GitHub Actions](https://github.com/KaR7NA16/surrogate-eval/actions/workflows/tests.yml) runs tests, builds distributions and verifies a clean wheel installation on Linux and Windows with Python 3.11 and 3.14. Each run records the outcome for its commit. See [contribution guidelines](CONTRIBUTING.md) and the [roadmap](docs/roadmap.md).

## Documentation and citation

| Start here | Go deeper |
|---|---|
| [External-model tutorial](docs/your_model.md) | [Metrics and assumptions](docs/methods.md) |
| [Data format](docs/data_format.md) | [Design and scope](docs/design.md) |
| [Python API](docs/api.md) | [Related work](docs/related_work.md) |
| [Case verification and export](reproduction/README.md) | [Development provenance](docs/provenance.md) |

Citation metadata is provided in [CITATION.cff](CITATION.cff); version history is in [CHANGELOG.md](CHANGELOG.md). Version 0.1.0 is an alpha release, with no public package index release or DOI assigned.

New package code, tests and documentation use the [MIT License](LICENSE). Historical research evidence retains its separate [licensing scope](docs/licensing.md).
