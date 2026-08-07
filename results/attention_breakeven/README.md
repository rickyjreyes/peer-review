# Rejected-attention breakeven sweep

This directory records the calibrated sweep over rejected-paper attention retention. The sweep holds the reviewer calibration fixed and varies the fraction of normal downstream attention retained by rejected work from 0 to 1.

The run used 300 worlds, 250 papers per world, 25 periods, and attention-retention values from 0% to 100% in 5-percentage-point increments.

## Result

No mean true-value-recovery breakeven occurred anywhere on the admissible interval.

| Rejected-work attention retained | Peer review true value | Open triage true value | Peer minus triage |
|---:|---:|---:|---:|
| 0% | 76.74% | 84.25% | -7.51 pp |
| 55% | 77.55% | 84.25% | -6.70 pp |
| 100% | 76.78% | 84.25% | -7.48 pp |

At 100% retention, the 95% interval for the mean peer-minus-triage difference was -8.74 to -6.22 percentage points. Across the full sweep, changing rejected-paper attention retention had little systematic effect on peer-review truth recovery.

## Interpretation

This falsifies a simple reading of the current calibrated model in which post-rejection attention loss alone drives the peer-review/open-triage gap. In the calibrated run, only about 4.1% of peer-reviewed papers remain rejected at the end, while the two institutions also use different downstream attention-allocation rules. Open triage deliberately mixes merit-based allocation with randomized exploration; the peer-review arm allocates attention using confidence, prestige, clarity, and popularity.

Therefore the absence of a breakeven is not a dominance proof for veto itself. It shows that, under the present calibrated architecture, restoring rejected papers to full attention is insufficient to eliminate the gap. The next isolating test should give peer review and the no-veto comparator the same downstream attention-allocation rule and vary only the gate.

The generated `summary.csv` reports the complete sweep, confidence intervals across simulated worlds, false-recognition rates, and calibration error. Reviewer behavior is empirically calibrated; remaining ecosystem quantities are structural draws.
