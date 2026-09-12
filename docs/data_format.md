# Paired data format, version 1

All NPZ files load with `allow_pickle=False`. Use the Python constructors to validate arrays before saving. Numeric values must be finite; cases are never silently discarded. Identifiers are nonempty strings or integers. Sample and target identifiers are unique and order-sensitive.

| Reference field | Shape | Meaning |
|---|---|---|
| `schema_version` | scalar integer | `1` |
| `reference` | N × 3 × D × H | Physical outputs; branch order nominal, plus, minus |
| `epsilon` | scalar or N | Strictly positive perturbation amplitude |
| `scale` | scalar, D, or N × D | Strictly positive scales fixed without evaluation targets |
| `horizons` | H | Positive, strictly increasing forecast times |
| `sample_ids` | N | Evaluation row identities |
| `cluster_ids` | N | Supplied independent replication groups |
| `target_ids` | D | Physical target identities, in output order |
| `units` | scalar string | Physical unit declaration, checked for exact equality |

Prediction records contain `schema_version`, `predictions` (same N × 3 × D × H), `sample_ids`, `horizons`, `target_ids` and `units`. Amplitudes/scales come from the reference. Unit labels do not perform unit conversion; mixed targets need a consistent declared convention and explicit scales. Array shapes alone cannot verify that plus/minus branches describe physically matched interventions.

Distinct directions at one initial state may be separate rows, but should remain in the appropriate common cluster. Repeated neural-network initializations do not become independent training datasets. The caller specifies this design; software cannot certify independence.

Writers refuse overwrite and require `.npz`. JSON reports use `null` for undefined relative reductions and reject NaN. Unknown NPZ fields are ignored; required version-1 fields must be present. A format version is not a claim of permanent API stability.

Optional feature files contain `schema_version=1`, `features` (N × 3 × K), `sample_ids`, `feature_names` (K distinct names), and scalar string `feature_spec`. Optional geometry files contain `A` (... × input coordinates × state directions), `B` (... × output targets × state directions), and optionally `M` with the shape of B. These are a separate derivative interface, not fields automatically inferred from paired outputs.
