# Design and scope

The user problem is concrete: low ordinary prediction error does not by itself establish fidelity to a physical perturbation. A researcher needs matched response labels, comparable predictions, explicit aggregation, and a record of what was selected before evaluation.

The package separates validated data interchange (`schema`), model-independent scoring (`metrics`), optional reference-derivative diagnostics (`geometry`), an established fixed-feature baseline (`repair`), reports/CLI, generated examples, and the optional original-study adapter (`legacy`). The core needs only NumPy. It does not import the historical experiment scripts. This boundary makes an external simulator or surrogate usable without the original L96 weights.

Writers refuse existing files; saved CLI reports include software version, timestamp and input hashes where inputs are file-based. These records aid tracing but do not cryptographically certify the entire scientific process. Feature descriptions, physical pairing, scale provenance and cluster independence remain caller responsibilities.

Historical evidence is stored externally and excluded from this repository, wheels and source distributions. `reproduction/` contains instructions for the optional case, not historical records or package implementation. Independent installation and generated examples are tested separately from migration equivalence.
