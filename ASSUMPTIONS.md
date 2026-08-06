# Assumption and Calibration Status

The repository separates observable empirical targets from latent model coefficients. A controlled-study percentage is never inserted directly as a coefficient unless the model defines that coefficient as the same observable.

## Empirically estimated reviewer observables

The machine-readable targets are in `data/empirical_targets.json`.

| Observable | Target | Source role |
|---|---:|---|
| Major-error detection | 2.58 / 9 = 0.2867 | Primary calibration target |
| Inter-reviewer reliability | 0.34 | Primary calibration target |
| Cohen's kappa | 0.17 | Primary calibration target |
| Positive-result recommendation | 0.973 | Primary calibration target |
| Null-result recommendation | 0.800 | Primary calibration target |
| Detection among reviewers recommending rejection | 0.391 | Secondary calibration target |
| Unsupported-conclusion miss rate | 0.68 | Validation-only target |

The calibration routine uses simulation-based parameter search so the generated reviewer behavior reproduces these observables. The retained internal parameters are not themselves direct empirical measurements.

## Empirically bounded

- Blinding or requiring signatures is represented with a zero baseline effect on planted-error detection because the cited randomized trial found no statistically significant improvement. This is a baseline assumption, not proof of exact zero effect.
- Reviewer-training effects are labeled small but are not assigned a numerical coefficient until the full trial estimates are extracted and modeled.

## Partially constrained

- Resubmission probability varies strongly by discipline, journal tier, time period, and whether transfer systems are counted.
- Review-induced manuscript improvement has empirical evidence, but there is no universal cross-disciplinary rate that maps cleanly onto the model's quality vector.
- Prestige and affiliation effects are documented, but their magnitude depends on review design and field.
- Review and publication delay can be measured, but its scientific cost is context-dependent.

## Weakly informed or unidentified

- Post-rejection attention loss.
- Abandonment after rejection.
- Long-term discoverability of rejected but valid work.
- Comparative effectiveness of open triage at scale.
- Downstream correction and replication rates under alternative institutions.
- The fraction of submissions that are substantially true, partially true, or substantially false.

These quantities must be swept across broad ranges or estimated in discipline-specific datasets. The repository does not label one selected value as empirical.

## Illustrative

- Heavy-tailed scientific-value distributions.
- Domain composition in the high-consequence experiment.
- Fraud prevalence where no unified target is supplied.
- The exact functional form of attention concentration.

## Normative

- Relative utility weights for false recognition, missed discoveries, delay, labor, equity, and social harm.
- Random exploration share in open triage.
- Recognition and withdrawal thresholds when interpreted as policy choices rather than measurements.

## Interpretive boundary

The calibrated reviewer submodel supports statements about whether the simulator can reproduce controlled observations of reviewer behavior. It does not by itself calibrate the complete scientific ecosystem. The headline institutional simulations remain structural and partially identified until rejection consequences, resubmission, discoverability, and long-run correction are empirically constrained.
