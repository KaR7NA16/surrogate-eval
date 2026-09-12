# Development and reproducibility

The package separates reusable numerical methods from optional research evidence. Its development included AI-assisted implementation and documentation; numerical behavior is checked through executable tests and independently reconstructed case-study predictions.

The original Lorenz–96 evidence is distributed separately from this repository. The external bundle's scoped manifest, `RESP_response_lab_release_manifest_20260912.json`, binds 119 files. The adapter verifies that manifest before reading model weights and reference data. A matching hash establishes file identity, not scientific correctness or redistribution rights.

Package validation records include source hashes, software versions and command outcomes. They are generated in ignored output directories for local review. Source paths, private project notes and internal editorial discussions are not part of the public software documentation.

The linear and pendulum examples demonstrate the generic workflow independently of the historical case. Software verification and its scope are recorded in [evidence](evidence.md); numerical definitions are in [methods](methods.md). No peer review, external adoption or cross-system performance claim is implied by software tests.
