# Multifield stochastic stress test

This is the preregistered robustness test defined in `PREDICTION_MULTIFIELD_STRESS.md`. It uses 300 paired worlds, 250 papers per world, 25 periods, six explicit fields, the committed reviewer calibration, and equal evaluation budgets across systems.

The multifield regimes add field-specific evidence noise, learning and replication multipliers, persistent stochastic field shocks, within-field belief spillover, and weaker cross-field spillover. These mechanisms are institution-neutral: all publication systems experience the same epistemic environment.

## Primary result

The preregistered directional prediction was **not supported** for true scientific value recovery. Correlated field learning narrowed, rather than widened, the peer-review deficit relative to open triage.

| Regime | Peer review | Open triage | Peer minus triage | 95% CI |
|---|---:|---:|---:|---:|
| Independent | 74.91% | 83.34% | -8.43 pp | [-9.59, -7.27] pp |
| Multifield, moderate | 79.14% | 83.63% | -4.49 pp | [-5.22, -3.76] pp |
| Multifield, strong | 83.37% | 84.25% | -0.88 pp | [-1.45, -0.31] pp |

The change in the peer-review deficit relative to the independent regime was +3.94 pp under moderate coupling and +7.55 pp under strong coupling. In other words, field-level spillover allowed concentrated systems to recover substantially more true value because evidence gathered on some claims shifted beliefs about related claims.

## But correlated learning amplified false recognition

The same mechanism produced a large downside for peer review. Its false-recognition rate rose from 0.41% in the independent regime to 2.06% under moderate coupling and 6.04% under strong coupling. Open triage rose from 0.06% to 0.17% and 0.28%, respectively. Under strong coupling, peer review therefore produced about 5.76 percentage points more false recognition than open triage.

Calibration error also worsened more strongly for peer review. Under strong coupling its calibration MSE was 0.0808 versus 0.0501 for open triage.

## Net modeled score

Peer review remained the weakest system on the model's composite score in every regime:

- Independent: peer review 0.7822, open 0.8330, hybrid 0.8918, open triage 0.9063.
- Multifield moderate: peer review 0.7998, open 0.8513, hybrid 0.8934, open triage 0.9032.
- Multifield strong: peer review 0.7748, open 0.8411, hybrid 0.8893, open triage 0.9013.

Peer review's composite-score win share was 4.67%, 9.0%, and 6.33% across the three regimes, the lowest of the four systems each time. These scores include structurally sampled weights for true-value recovery, false recognition, calibration, delay, and labor, so they are model outcomes rather than empirical estimates of real-world welfare.

## Interpretation

The multifield extension changes the mechanism story. Richer correlated scientific understanding does **not** make peer review worse at recovering true value in this implementation. It actually helps concentrated systems catch up because discoveries propagate through field-level belief spillover. But that same concentration becomes much more vulnerable to correlated error: false recognition and miscalibration grow sharply, and peer review remains weakest on the composite outcome.

The result therefore strengthens a different game-theoretic concern: when scientific beliefs are correlated and path-dependent, concentrated attention can spread both valid discoveries and mistakes. The question is not only how much true work is recovered, but how costly institutional amplification of wrong beliefs becomes.

## Boundary

This is a structural stress test, not a reconstruction of named disciplines or an empirical estimate of peer review's real-world effect. Field parameters and coupling strengths are simulated over specified ranges. The test is useful because the preregistered prediction was allowed to fail rather than being tuned after observing the result.
