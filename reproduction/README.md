# Optional Lorenz–96 reproduction case

The generic package and examples run without historical evidence. An external, complete L96 evidence bundle is required only for case-study verification, export and integration tests. This repository does not contain its raw records, internal reports, data or model weights. No public download is currently provided.

If you have authorized access to the complete bundle, extract it outside the repository. Substitute its directory for `PATH_TO_EVIDENCE` below; use a new output directory for each export.

```sh
rfl verify-evidence --root PATH_TO_EVIDENCE
rfl l96-export --root PATH_TO_EVIDENCE --study finetune --forcing 24 --block history --output outputs/l96-f24
rfl compare --reference outputs/l96-f24/reference.npz --model original=outputs/l96-f24/original.npz --model response=outputs/l96-f24/response_trained.npz --baseline original --output outputs/l96-f24/comparison.json --markdown outputs/l96-f24/comparison.md
python scripts/check.py --legacy --evidence-root PATH_TO_EVIDENCE
```

Alternatively, set `RFL_L96_ROOT` to the external evidence directory before `python scripts/check.py --legacy`. With no evidence configured, normal core tests run without this optional case. The legacy flag requires an existing evidence directory and does not silently substitute an unavailable case.

Export preserves eight training-dataset clusters. By default all three network initializations appear as repeated rows inside each dataset cluster, not additional independent clusters. `export_metadata.json` distinguishes physical starts from evaluation rows.

The required scoped manifest is `RESP_response_lab_release_manifest_20260912.json`. Every bound file must match before export. Older manifests inside the external archive do not replace this scoped inventory. Files are read without modifying them; export writes only to the selected output directory.

The external bundle's own reproduction guide covers optional training reruns. The adapter here reconstructs saved-model predictions in NumPy. Core installation does not rerun training, distribute the historical archive or grant rights over it. See [evidence](../docs/evidence.md) and [licensing](../docs/licensing.md).
