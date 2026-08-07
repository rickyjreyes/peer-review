# Preregistered prediction: gate × attention factorial

Date committed: 2026-08-07

This prediction is recorded before running the gate × downstream-attention factorial described below. It is motivated by the completed rejected-attention sweep, which found essentially no change in the peer-review/open-triage truth-recovery gap as rejected-paper attention retention varied from 0% to 100%, together with the calibrated run's low final rejection share.

## Factorial design

The next experiment will separate two mechanisms:

1. **Gate**: formal review has publication/attention consequences for papers that are not accepted versus the same review information being non-vetoing.
2. **Downstream attention**: concentrated allocation versus transparent triage with randomized exploration.

The four arms are:

- gate + concentrated attention;
- gate + exploration;
- no gate + concentrated attention;
- no gate + exploration.

Reviewer behavior will remain fixed to the committed empirical calibration. The same underlying simulated worlds and structural parameter draws will be shared across the four arms as far as practical. The primary outcome is true scientific value recovered. Secondary outcomes include false recognition, calibration error, time to recognition, review labor, publication delay, and rejection-path diagnostics.

## Prediction recorded before the run

- **Gate main effect:** small, predicted absolute effect on mean true-value recovery **below 2 percentage points**, and plausibly statistically indistinguishable from zero under the current calibrated architecture.
- **Attention-allocation main effect:** substantially larger than the gate main effect. Exploration is predicted to account for most of the previously observed open-triage advantage.
- **Interaction:** predicted to be small to moderate relative to the attention main effect.
- **Policy prediction:** `gate + exploration` is predicted to perform much closer to `no gate + exploration` than to `gate + concentrated attention`. If so, changing downstream attention allocation would be the primary reform target in the current model, not removing formal review or veto by itself.

These are directional predictions, not acceptance criteria. They will not be edited after the experiment to match the result.

## Additional diagnostic prediction

The exploration share is a normative policy parameter rather than an empirically calibrated quantity. A separate sweep will vary the exploration share across its admissible range and beyond it for sensitivity analysis. The prediction is that some positive exploration improves recovery relative to fully concentrated allocation, but that performance need not increase monotonically all the way to 100% random exploration.

## Interpretive boundary

A small gate effect in this experiment would not establish that real-world publication gates are harmless. The current model may understate rejection-path delay, labor, abandonment, and endogenous author responses. Conversely, a large attention-allocation effect would be a mechanism result about the present model, not a direct empirical estimate of real publication systems.
