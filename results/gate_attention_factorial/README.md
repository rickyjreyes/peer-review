# Gate x attention factorial

This is the clean rerun of the preregistered 2x2 mechanism-isolation experiment. The directional prediction was committed first in `PREDICTION_ATTENTION_FACTORIAL.md` and was not changed after seeing results.

The run used 300 paired worlds, 250 papers per world, and 25 periods. Reviewer behavior was fixed to the committed empirical calibration. Every arm retained the same formal review and review-triggered revision. The factors were:

- **Gate vs no gate:** a negative review either down-weighted downstream attention and incurred publication delay, or remained a non-vetoing review label.
- **Concentrated vs triage allocation:** the repository's peer-style confidence/prestige-weighted allocation rule versus its transparent triage rule. A separate 150-world sweep varied only the randomized-exploration share inside the triage rule from 0 to 100%.

## Primary result: true scientific value recovered

| Arm | Mean true value recovered |
|---|---:|
| Gate + concentrated allocation | 74.76% |
| No gate + concentrated allocation | 74.86% |
| Gate + triage allocation | 83.63% |
| No gate + triage allocation | 83.52% |

Paired factorial effects:

| Effect | Mean | 95% interval |
|---|---:|---:|
| Gate main effect | +0.006 percentage points | -0.365 to +0.378 pp |
| Triage-allocation main effect | **+8.760 pp** | **+7.701 to +9.820 pp** |
| Gate x triage interaction | +0.211 pp | -0.544 to +0.966 pp |

The preregistered prediction was therefore supported in the current architecture: the isolated gate effect was effectively zero, while changing the downstream attention-allocation rule produced the large recovery gain. Keeping the gate while switching to triage allocation recovered essentially the same true value as removing the gate and using triage allocation.

## Rejection-path diagnostics

- Initial rejection share: **5.96%**.
- Final rejection share: **4.35%**.
- Fraction of initially rejected papers that resubmitted at least once: **37.14%**.
- Mean modeled publication delay among initially rejected papers in gated arms: **1.40 periods**.
- Mean review labor among initially rejected papers: **3.48 reviewer assignments**.

These values reinforce the earlier warning that the gate is weak in the present calibrated regime. This experiment therefore does not establish that real publication gates are harmless; it establishes that the current model's previous peer-review/open-triage gap is not primarily a veto effect.

## Exploration-share sweep

Varying the randomized-exploration share from 0% to 100% inside the triage allocation rule did **not** produce a monotonic increase in truth recovery. Recovery stayed in a relatively narrow band, roughly 83.5% to 84.6%, across the sweep. In particular, the triage rule already recovered about 83.8% with zero randomized exploration.

Therefore the +8.76 percentage-point factorial effect should not be described as an effect of random exploration alone. It is an effect of the **full downstream allocation rule**: the triage arm changes the scoring basis and concentration structure as well as adding an optional exploration component.

## Interpretation

The clean mechanism result is narrower, but more informative, than the original headline comparison:

> In the present calibrated simulation, downstream attention allocation is the dominant source of the peer-review/open-triage truth-recovery difference. The modeled publication veto has approximately zero isolated effect under the current low-rejection, low-delay regime.

The next decomposition should isolate which components of the triage allocation rule matter most, especially prestige weighting, novelty weighting, concentration, and randomized exploration. Real-world gate effects also require stronger empirical constraints on first-round rejection, sequential resubmission delay, abandonment, labor, and endogenous author behavior.

Files:

- `arm_means.csv`: compact arm-level means.
- `factorial_effects.csv`: paired main effects, interaction, conditional effects, and confidence intervals.
- `exploration_sweep.csv`: secondary sensitivity sweep of randomized exploration share.
- `metadata.json`: fixed run settings and interpretive boundary.
