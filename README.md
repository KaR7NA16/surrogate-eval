# Response Fidelity Lab

**Evaluate predictions. Examine responses. Make model comparisons reproducible.**

A research toolkit for evaluating the **perturbation-response fidelity of scientific surrogate models**—from paired-output scoring to local derivative diagnostics and response-aware baseline fitting.

[中文](README.zh-CN.md) · [Quick start](#quick-start) · [Bring your own model](docs/your_model.md) · [Methods](docs/methods.md) · [Research case](docs/evidence.md)

**Python 3.11+ · NumPy core · CLI + Python API · Version 0.1.0 / Alpha**

## Why response fidelity?

A surrogate may predict a trajectory accurately while responding incorrectly to a change in its input. For scientific workflows that use perturbations to investigate sensitivity or compare interventions, prediction error alone leaves an essential part of model behavior unmeasured.

Response Fidelity Lab evaluates the nominal prediction and the response to matched positive/negative perturbations together. It helps researchers investigate:

- **Response accuracy:** Does the model reproduce the reference response, across targets and forecast horizons?
- **Model comparison:** Does an apparent improvement persist across the supplied replication groups?
- **Local information limits:** With reference Jacobians available, how much target sensitivity lies outside the retained input row space?
- **Training trade-offs:** Does response-aware fitting improve held-out response error while preserving nominal accuracy?

The workflow accepts predictions from an existing model. Basic evaluation requires physical reference outputs and matching predictions; derivative diagnostics are optional.

## What you can do

| Capability | Research use | Interface |
|---|---|---|
| **Paired-response evaluation** | Measure nominal and finite-response error by target, horizon and replication group | `rfl score` |
| **Paired model comparison** | Compare candidates against a baseline with cluster-level results and descriptive bootstrap intervals | `rfl compare` |
| **Local tangent diagnostics** | Decompose realizable surrogate tangent error into visible and invisible components | `rfl diagnose` |
| **Noise sensitivity analysis** | Evaluate a pointwise linear oracle under a declared input-noise model | `rfl noise` |
| **Response-aware baseline fitting** | Fit and select fixed-feature ridge heads using separate training and validation records | `rfl fit-head`, `rfl predict-head` |
| **Research-case reproduction** | Verify and convert the frozen Lorenz–96 evidence into the common evaluation format | `rfl verify-evidence`, `rfl l96-export` |

### Designed for inspectable scientific comparisons

**Explicit physical pairing.** Versioned records specify nominal/plus/minus branches, perturbation amplitudes, target scales, sample identities, units and forecast horizons. Mismatched identities or nonfinite values fail validation.

**Replication-aware aggregation.** Supplied clusters receive equal aggregate weight, including when their row counts differ. Network initializations and spatial coordinates need not be treated as additional independent experiments.

**Traceable outputs.** JSON results support further analysis; Markdown reports support review. File-based scoring and fitting record input hashes and software version. Writers refuse to overwrite existing results.

**Lightweight integration.** The core requires NumPy only. Use the CLI for files or the Python API in an existing analysis pipeline. Neither a GPU nor the original study is needed for general evaluation.

## Quick start

From the repository root, install the package and run either self-contained CPU example:

```sh
python -m pip install .
rfl demo --system linear --output outputs/linear
rfl demo --system pendulum --output outputs/pendulum
```

Open `outputs/linear/comparison.md` or `outputs/pendulum/comparison.md`. Each example generates separate training, validation and test records, fits baseline heads, freezes validation selection and evaluates on held-out initial states.

The linear example also provides Jacobians for the optional diagnostics:

```sh
rfl diagnose --input outputs/linear/geometry.npz --output outputs/linear/geometry.json --markdown outputs/linear/geometry.md
rfl noise --input outputs/linear/geometry.npz --sigma 0 0.001 0.01 --output outputs/linear/noise.json
```

Use a new output directory when rerunning. `python -m response_fidelity` is equivalent to `rfl`.

## Bring your own model

Export physically matched reference and predicted outputs in the common shape:

```text
[evaluation row, nominal / plus / minus, target, forecast horizon]
```

Then score one model or compare multiple candidates:

```sh
rfl score --reference reference.npz --prediction model.npz --output score.json --markdown score.md
rfl compare --reference reference.npz --model baseline=baseline.npz --model candidate=model.npz --baseline baseline --output comparison.json --markdown comparison.md
```

The Python interface uses the same validated records:

```python
from response_fidelity import load_reference, load_prediction, evaluate

reference = load_reference("reference.npz")
prediction = load_prediction("model.npz")
report = evaluate(reference, prediction)
print(report["aggregate"]["response_mse"])
```

Start with the [runnable external-model tutorial](docs/your_model.md), then consult the [data format](docs/data_format.md) and [API reference](docs/api.md). Reference outputs for matched perturbations are required; ordinary time-series data alone do not provide response-error labels.

## Case introduction: tested in a Lorenz–96 research workflow

Response Fidelity Lab grew from a study of perturbation-response fidelity in partially observed chaotic dynamics. This historical case shows the workflow in use under a defined scientific protocol, with controls and numerical verification.

| Test dimension | Recorded scope |
|---|---|
| Dynamical regimes | Partially observed Lorenz–96 at F16 and F24 |
| Evaluation design | Two intervention studies; selection fixed before generating 1,536 fresh initial states per study |
| Replication | Eight training-dataset clusters per forcing; three network initializations within each cluster and input block |
| Controls | Original same-input MLP, nominal continuation and response-aware interventions; current-only and history inputs |
| Numerical checks | Independent NumPy prediction/metric checks, derivative checks and a separate head-solve reproduction |
| Provenance | 119 manifest-bound files covering frozen protocols, weights, reference data and results |

History-input models yielded these median cluster response-error reductions against their original same-input MLP:

| Intervention | F16 | F24 |
|---|---:|---:|
| Frozen-feature head | 2.13% | 0.33% |
| Full-network intervention | 3.03% | 0.88% |

This case demonstrates use in a controlled scientific evaluation, including faithful reporting of modest effects. No cluster met the declared joint practical criterion (at least 10% response improvement and at most 5% nominal-error deterioration) in any of the 24 study/forcing/input/arm combinations. The studies used different fresh cohorts, so their rows are not a paired comparison of methods.

The [evidence record](docs/evidence.md) explains the controls, verification and limits. This historical case introduction does not constitute third-party certification or validation across unrelated systems. Original evidence is retained externally and is not distributed with the repository; [optional reproduction](reproduction/README.md) requires access to that bundle. Both bundled CPU examples run from a clone without historical data.

## Methodological scope

Response Fidelity Lab reports finite perturbation errors, not unmeasured full-Jacobian accuracy. Local geometry assumes supplied, compatible reference derivatives; a pointwise linear oracle does not establish global learnability. Bootstrap intervals are descriptive for fixed models and supplied clusters. Physical pairing, training-only scales and cluster independence remain the researcher's responsibility.

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

CI is configured for Linux and Windows with Python 3.11 and 3.14; remote execution has not yet been verified. See [contribution guidelines](CONTRIBUTING.md) and the [roadmap](docs/roadmap.md).

## Documentation and citation

| Start here | Go deeper |
|---|---|
| [External-model tutorial](docs/your_model.md) | [Metrics and assumptions](docs/methods.md) |
| [Data format](docs/data_format.md) | [Design and scope](docs/design.md) |
| [Python API](docs/api.md) | [Related work](docs/related_work.md) |
| [Research-case reproduction](reproduction/README.md) | [Development provenance](docs/provenance.md) |

Citation metadata is provided in [CITATION.cff](CITATION.cff); version history is in [CHANGELOG.md](CHANGELOG.md). Version 0.1.0 is an alpha release, with no public package index release or DOI assigned.

New package code, tests and documentation use the [MIT License](LICENSE). Historical research evidence retains its separate [licensing scope](docs/licensing.md).
