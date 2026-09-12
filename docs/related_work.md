# Related work and differentiation

This project packages a response-evaluation workflow; derivative supervision, ridge fitting and delay coordinates are established methods. The following primary sources delimit its claims.

| Source | Relevant contribution | Relationship to this software |
|---|---|---|
| [Sobolev Training for Neural Networks, NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html) | Training with derivative information | Response-aware fitting here is not a new derivative-training principle |
| [Tian, JENN, 2024 preprint](https://arxiv.org/html/2412.01013v1) | Jacobian enforcement and dynamical-model data assimilation, including L96 | Direct antecedent; partial observation and reused paired labels define a different setting but do not alone establish novelty |
| [Gonzalez-Sieiro et al., Engineering with Computers, 2026](https://doi.org/10.1007/s00366-026-02327-z) | Joint nonlinear optimization and linear least-squares approaches | Output-head refitting is an established baseline, not this project's algorithmic invention |
| [PySINDy documentation](https://pysindy.readthedocs.io/en/stable/) | Sparse identification of dynamical systems | Identified models may supply predictions for this evaluation interface |
| [PyKoopman project](https://github.com/dynamicslab/pykoopman) | Data-driven Koopman modeling | Another possible upstream source of model predictions |
| [TorchMetrics project](https://github.com/Lightning-AI/torchmetrics) | General model metrics infrastructure | This package makes the physical pairing, normalization and cluster contract explicit for a narrower research workflow |

These are scope comparisons, not measured performance comparisons or assertions that other libraries cannot implement the same calculations. The proposed software contribution is the combination of a portable paired-data contract, response/nominal comparison, optional local diagnostics, explicit limitations and a frozen case study. Adoption, independent reuse and comparative usability have not yet been measured.
