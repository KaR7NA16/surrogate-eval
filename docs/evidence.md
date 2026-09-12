# Evidence record

## Original L96 case

Two intervention studies each froze selection before generating 1,536 fresh initial states, across F16/F24 and eight training-dataset clusters per forcing. Each study retained three network initializations per input block inside the same dataset cluster. The second study trained 384 full-network trajectories; this is computational search effort, not 384 independent datasets. It used finite paired increments, not exact full-Jacobian labels.

History-input response-trained models versus their original same-input MLP baseline yielded:

| Study | F16 median cluster response-error reduction | F24 | F24 longest-horizon reduction |
|---|---:|---:|---:|
| Frozen-feature head intervention | 2.13% | 0.33% | 0.07% |
| Full-network continuation/intervention | 3.03% | 0.88% | 0.54% |

The studies used different fresh cohorts; rows are not a direct paired comparison between intervention methods. All 24 study/forcing/input/arm combinations had 0/8 clusters meeting the declared practical criterion of at least 10% response improvement with at most 5% nominal-error deterioration. The response pool searched more trajectories than the nominal continuation control. These facts do not demonstrate a strong new method or resolve the F24 difficulty.

New starts came from the known numerical ensemble; they are not newly trained datasets, unseen systems or an untouched external benchmark. Exploratory scalar sign decisions likewise showed modest benefit and do not establish control utility. Original timing numbers describe one local hardware run, not portable speed benchmarks. Full training was executed once; selected predictions, metrics, derivatives and the head solve received separate checks. A complete second full-training reproduction was not performed.

## Software evidence

Generic tests check identity/shape failures, nonfinite inputs, unequal clusters, zero baselines, analytic rank-deficient geometry, independent oracle solves, leakage-sensitive fitting, CLI behavior and generated linear/pendulum workflows. Optional L96 integration tests verify the 119-file evidence manifest and migration equivalence. These tests validate implementation behavior; they do not add scientific replications or prove global observability.

Local test records are generated under `outputs/check-*/validation.json`, with source hashes, interpreter/library versions and command outcomes. See [reproduction](../reproduction/README.md) for the optional case. Remote CI is configured separately; no remote success is implied by local checks.
