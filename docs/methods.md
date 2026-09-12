# Metrics and assumptions

For row i, target j and horizon t, let y0, y+ and y− be physical reference outputs, with predicted counterparts, positive amplitude εᵢ and supplied scale sᵢⱼ. The finite central response is r = (y+ − y−)/(2εᵢ sᵢⱼ). Nominal error is ((ŷ0 − y0)/sᵢⱼ)²; response error is (r̂ − r)². This measures the supplied finite perturbations. Convergence to a derivative requires smoothness and an amplitude study that this package does not assume.

Rows and targets receive equal weight within each cluster, then clusters receive equal weight. Horizons also receive equal weight in the scalar aggregate. Comparisons report both `1 − mean(candidate)/mean(baseline)` and the median of cluster-specific relative reductions: these are different statistics. Positive means improvement. Zero baseline error gives an undefined relative reduction (`null`), while absolute errors remain meaningful.

Paired percentile bootstrap intervals resample whole supplied clusters, using the same sampled cluster indices for both models. Default: 2,000 resamples, seed 0, 95% percentile interval. One cluster yields no interval. These intervals describe the fixed comparison conditional on the protocol; they do not correct adaptive test reuse, establish replication across systems or certify that supplied clusters are independent.

## Optional local geometry

Let A map a full-state perturbation to model inputs, B map it to reference targets, and M be the surrogate's composite state Jacobian. SVD retains singular values strictly greater than `cutoff * largest_singular_value`, default cutoff 1e-10. P projects onto that retained row space. With isotropic perturbation covariance I/d, invisible risk is ||B(I−P)||²/d, per target. If M = MP, total error splits into visible ||M−BP||²/d plus invisible risk. The implementation checks relative chain leakage per batch member (default tolerance 1e-8) and reports the identity residual. Truncation is part of this numerical definition.

Local linear access is not proof of a globally well-defined or learnable predictor. Visible error can include model capacity, optimization, estimation and globally ambiguous inputs. Derivatives must be provided in consistent physical coordinates; no observability claim is inferred from ordinary samples.

For z = Av + η, Cov(v)=I/d and Cov(η)=σ²I, `noise` evaluates the unrestricted pointwise linear oracle. A retained singular mode s contributes its target energy times dσ²/(s²+dσ²), divided by d; invisible modes retain their full energy, including at zero noise. Numerically discarded directions are treated as unavailable for every noise amplitude. Independent noise is assumed in the supplied input basis: lag differences sharing a raw measurement generally violate that assumption.

## Exploratory scalar decision score

For each target/horizon separately, choose plus if predicted response ≤ 0, otherwise minus, to minimize that scalar output. Ties choose plus. Normalized regret is chosen true signed response + |true response|; a random sign has expected regret |true response|. Wrong-direction rate counts strictly positive chosen responses. This is an exploratory one-step scalar action, not simultaneous multi-target control or closed-loop utility.
