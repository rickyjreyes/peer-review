# Scale and Incentive Stress Test

This experiment extends the calibrated publication ecosystem into a repeated-game setting. It asks whether mechanisms omitted from the main model change the comparison between mandatory prepublication peer review and open publication with transparent triage.

## Design

Each simulated world contains 10 generations. Every generation begins with the same latent pool of 300 research opportunities for both institutions, of which 120 projects are pursued. The two institutions receive the same total expert-action budget. Formal prepublication review consumes part of the peer-review budget; open triage uses the corresponding actions for continuing evaluation.

Reviewer error detection, disagreement, recommendation behavior, and positive-outcome bias use the committed calibrated reviewer parameters. The added mechanism strengths are structural sweeps, not empirical estimates.

The ablations are:

- `baseline`: no added scale-incentive mechanism;
- `finance`: commercial alignment can influence project choice and rewards;
- `government`: government-priority alignment can influence project choice and rewards;
- `fame`: prestige, popularity, certification, and conformity can influence career rewards;
- `entropy`: neglected ideas lose discoverability over time, with rejected work exposed to additional decay;
- `attribution`: an unrecognized independent idea can later be rediscovered under a higher-prestige lineage, with possible loss of original credit and feedback on independent participation;
- `all`: all five mechanisms together.

`attribution` is deliberately modeled as priority/credit capture or unattributed rediscovery. It is not an empirical claim that deliberate theft occurs at the sampled rate.

Common random numbers are used across systems and ablations within each world and generation: identical latent opportunity pools, project-choice shocks, and evaluation RNG streams. This makes the comparisons paired.

## 500-world result

The primary metric is the paired difference in recovered true scientific value: `open_triage - peer_review`. Positive values favor open triage.

| Scenario | Mean recovery gap | 95% bootstrap CI | Open triage higher |
|---|---:|---:|---:|
| Baseline | 1.83 pp | 1.67 to 2.00 pp | 84.8% |
| Finance | 2.01 pp | 1.85 to 2.16 pp | 89.0% |
| Government | 1.79 pp | 1.63 to 1.95 pp | 87.0% |
| Fame | 1.97 pp | 1.80 to 2.14 pp | 86.2% |
| Entropy | 2.92 pp | 2.71 to 3.12 pp | 91.8% |
| Attribution | 1.88 pp | 1.73 to 2.04 pp | 87.4% |
| All mechanisms | 3.07 pp | 2.87 to 3.28 pp | 93.2% |

The combined incentive layer increases the open-triage recovery advantage by **1.24 percentage points** relative to baseline (95% bootstrap CI **1.04 to 1.44 pp**). The amplification is positive in **72.6%** of paired worlds.

### Mean system outcomes

Under the baseline repeated-game model:

- open triage recovered 95.43% of true value;
- peer review recovered 93.60%;
- open triage false recognition was 0.36% versus 0.70% for peer review;
- attention Gini was 0.186 for open triage versus 0.404 for peer review.

With all added mechanisms:

- open triage recovered 94.54% of true value;
- peer review recovered 91.47%;
- open triage false recognition was 0.35% versus 0.67% for peer review;
- mean attribution integrity was 0.675 versus 0.563;
- modeled captured-value share was 0.55% versus 0.79%;
- attention Gini remained 0.186 versus 0.405.

## Scale curve

A separate 120-world run varied the number of generations while holding the opportunity pool and expert-budget rules fixed.

The recovery-rate difference shrinks with time because both institutions eventually recover more of the finite opportunity stream. The cumulative missed-value difference moves in the opposite direction.

| Generations | Baseline extra missed value under peer review | All-mechanism extra missed value under peer review |
|---:|---:|---:|
| 1 | 12.18 | 12.12 |
| 3 | 26.20 | 28.60 |
| 5 | 27.78 | 36.11 |
| 10 | 30.80 | 46.65 |
| 15 | 31.93 | 57.72 |

At 15 generations, the added incentive layer increases peer review's cumulative missed-value disadvantage by **25.80 model-value units** (95% bootstrap CI **18.84 to 32.92**). The recovery-rate gap at generation 15 is 1.25 pp in baseline and 2.38 pp with all mechanisms.

## Interpretation

The experiment does not show that every real financial, government, prestige, or attribution incentive harms peer review. Finance, government, fame, and attribution individually produced modest and sometimes mixed changes. The strongest individual amplifier was **attention entropy**.

The robust result is narrower: when a noisy publication veto already changes attention, adding repeated incentives and decay can compound the consequences of early false negatives. The cumulative scale effect is more visible in missed scientific value and attribution integrity than in the long-run recovery percentage, which tends to saturate as both systems accumulate evidence.

## Reproduction

```bash
python src/run_scale_incentive_experiment.py \
  --worlds 100 \
  --generations 10 \
  --pool 300 \
  --pursued 120 \
  --output-dir results/scale_incentives
```

The committed 500-world summary pools five independent 100-world batches with seeds 20260806 through 20260810.
