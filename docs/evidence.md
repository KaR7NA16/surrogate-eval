# Software verification record

## Historical-case integration

The optional Lorenz–96 adapter connects frozen research records to the generic evaluation interface. The regression test in `tests/test_cli_examples.py` exercises the default F24 history-input export and comparison:

1. Verify the SHA-256 hashes of all 119 files bound by the external evidence manifest before reading the case.
2. Reconstruct saved-model predictions using the NumPy adapter and export reference/prediction records.
3. Run the generic comparison command on those records.
4. Match the median cluster response reduction to the archived summary with an absolute tolerance of 1e-12.
5. Check that the original-model report retains eight dataset clusters.

This regression checks the selected export path and statistic. It does not rerun the original training or test every historical study/configuration. The complete evidence bundle remains external; see [case verification and export](../reproduction/README.md) for access requirements and commands.

## Generic numerical and data-contract tests

| Area | Executable checks |
|---|---|
| Pairing metadata | Reject mismatched sample IDs, horizons, target IDs and units; reject invalid shapes and nonfinite values |
| Aggregation | Check equal cluster weighting with unequal row counts, zero baselines and reproducible bootstrap output |
| Local geometry | Check known decompositions, rank-deficient inputs and agreement with a directly solved regularized oracle |
| Baseline fitting | Check training-only normalization, split consistency and agreement with independent normal equations |
| End-to-end use | Run linear and pendulum examples, external scoring, comparison reports and CLI diagnostics |

Metadata checks cannot establish physical pairing on their own: the caller supplies physically matched reference and prediction branches. See the [data format](data_format.md) for that contract.

## Reproducing the checks

`python scripts/check.py` runs the core suite. At version 0.1.0, it has 37 passing tests and one optional historical test skipped when no evidence bundle is configured. With `--legacy --evidence-root PATH_TO_EVIDENCE`, all 38 tests run. Validation records under `outputs/check-*/validation.json` contain source hashes, interpreter/library versions and command outcomes.

After `python -m build`, `python scripts/check_distribution.py` creates a clean environment, installs the wheel, and runs both demos, the external-model tutorial and geometry/noise commands outside the source working directory.

[GitHub Actions](https://github.com/KaR7NA16/response-fidelity-lab/actions/workflows/tests.yml) runs core checks, packaging and clean-wheel verification on Linux and Windows with Python 3.11 and 3.14. Historical integration is a separate local check requiring the external bundle. These checks establish the tested implementation behavior within the stated scope.
