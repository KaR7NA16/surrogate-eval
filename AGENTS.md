# Response Fidelity Lab development

Keep reusable software and publishable documentation in this repository.

- Historical evidence is external. Do not copy internal reports, local source
  paths, private records or credentials into the repository. Preserve external
  historical protocols, results, models and manifests without rewriting them.
- Develop reusable functionality in `src/response_fidelity/`; use tests and
  examples that do not depend on the historical experiment.
- Basic installation and evaluation require NumPy only. Import heavyweight
  training libraries only in optional reproduction paths.
- Respect sample identity, train-only normalization, explicit perturbation
  amplitude, and dataset-level replication. Never silently discard nonfinite
  cases or turn a pointwise oracle into a global learnability claim.
- Keep scientific results distinct from software QA. New benchmarks must state
  which choices preceded evaluation and which analyses are exploratory.
- Do not publish, commit, push, or change historical evidence without a user
  request. Test outputs belong in ignored `outputs/` or temporary directories.
