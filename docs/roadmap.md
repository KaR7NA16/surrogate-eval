# Release limits and future acceptance criteria

Version 0.1.0 supplies a working local research toolkit. It has no measured adoption, external benchmark replication, public release or mature maintenance record. It expects dense in-memory arrays and does not support streaming/distributed evaluation. Caller-supplied physical pairing and independence cannot be verified from NPZ metadata alone.

Potential future work is listed with evidence needed before making stronger claims:

| Work | Acceptance evidence |
|---|---|
| External-user integration | A second researcher supplies real model outputs without editing package internals |
| Broader benchmark validity | Frozen protocol on a genuinely different system with independently defined replication units |
| Additional model adapters | Round-trip identity and physical-unit tests against upstream predictions |
| Derivative-learning effectiveness | Equal-budget controls and a fresh protocol showing a prespecified practical gain |
| Public software release | Responsible authorship, historical-data rights, remote CI results and a tagged release |

These are unimplemented future milestones. The current software should be assessed on its implemented interfaces and local verification, not on this roadmap.
