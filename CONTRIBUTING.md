# Contributing and support

Use Python 3.11+ and install `python -m pip install ".[dev]"`. Run `python scripts/check.py` for lint, formatting and tests with a unique repository-owned temporary directory. `python scripts/check.py --legacy --evidence-root PATH_TO_EVIDENCE` additionally checks an external complete evidence bundle. Run `python -m build` to build the small distribution.

Changes to scientific metrics should include definitions, edge cases and an independent numerical/analytic check. Keep reference labels, train/validation selection and final evaluation separate. A green test does not justify a stronger scientific claim. Never edit frozen evidence to make a test pass; add a new version and an explicit provenance record when a correction is authorized.

Report an issue with the command, version, minimal nonsensitive input example, expected behavior and actual traceback/result. This local repository has no public issue tracker yet. Do not include private training data or assume that an example contribution grants redistribution rights to its source data.
