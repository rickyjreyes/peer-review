# Preregistered prediction: multifield stochastic stress test

Date: 2026-08-07

This prediction is committed before the multifield runner or its results are added.

## Question

Does the relative performance of the modeled publication systems change when scientific understanding is represented as a set of heterogeneous fields subject to correlated stochastic evidence shocks and cross-field spillover, rather than as independent papers with only paper-level evidence noise?

## Design

The stress test will preserve the existing calibrated reviewer mechanism and equal downstream evaluation budgets. The same underlying paper populations will be evaluated under the same four institutional systems: peer review, open publication, open triage, and hybrid.

The extension will add explicit field labels and three epistemic regimes:

1. **Independent**: the existing calibrated ecosystem behavior with no field-level coupling.
2. **Multifield, moderate coupling**: field-specific evidence noise, learning and replication multipliers, persistent stochastic field shocks, within-field belief spillover, and weaker cross-field spillover.
3. **Multifield, strong coupling**: the same mechanisms with larger shock/spillover strength.

Field shocks and spillovers will be institution-neutral. They act on the evidence and belief process, not directly on publication-system labels. The systems differ only through their existing review and attention mechanisms.

Primary outcome: true scientific value recovered. Secondary outcomes: false recognition, calibration MSE, time to recognition, attention coverage/Gini, and system win share.

The primary comparison is the peer-review versus open-triage true-value gap in each epistemic regime and the change in that gap from Independent to Multifield.

## Prediction

Because concentrated attention makes early noisy judgments more persistent, adding correlated stochastic shifts and field-level spillover is predicted to **not improve peer review relative to open triage**. The preregistered directional prediction is that the peer-review minus open-triage true-value difference will remain negative and will become at least as negative under one or both multifield regimes as in the independent regime.

A stronger result would be a monotonic widening of the gap with coupling strength, but monotonicity is not required by the prediction.

## Falsification conditions

The prediction is not supported if peer review overtakes open triage in mean true-value recovery under the multifield regimes, or if both multifield regimes materially narrow the peer-review deficit relative to the independent regime.

## Interpretive boundary

This is a structural stress test, not a literal reconstruction of named scientific disciplines. Field-specific parameters and knowledge spillovers are deliberately simulated over broad ranges. A result that survives this extension shows robustness to correlated, path-dependent learning; it does not by itself establish an empirical effect size for the real scientific literature.
