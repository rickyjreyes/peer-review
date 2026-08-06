# Peer Review Ecosystem Simulation

This repository contains an agent-based simulation study comparing four scientific publication systems:

1. traditional prepublication peer review;
2. immediate open review;
3. open publication with transparent triage and randomized exploration; and
4. a hybrid model combining immediate public release with optional formal review.

The simulator creates paper populations with hidden false, partially true, and substantially true claims. Simulated agents do not know the hidden state. They observe noisy evidence over time, allocate finite attention, attempt replications, revise manuscripts, detect or miss defects, and update confidence.

## Main paper

- [`paper/peer_review_ecosystem_simulation.tex`](paper/peer_review_ecosystem_simulation.tex)

The LaTeX source is included directly so the manuscript can be compiled and audited alongside the code.

## Source code

- `src/scientific_publication_simulator.py`: canonical dynamic publication-ecosystem model.
- `src/symmetric_scientific_review_simulator.py`: symmetric model in which every system can detect defects, repair methods and reporting, miss errors, introduce harmful revisions, and consume expert labor.
- `src/reproduce_robustness_experiments.py`: exact runner for the symmetric-correction and equal-total-labor headline tests.
- `src/legacy_baseline_simulation.py`: earlier known-truth baseline retained for provenance.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the canonical simulator

```bash
python src/scientific_publication_simulator.py \
  --worlds 5000 \
  --papers 300 \
  --periods 30 \
  --output-dir simulation_output
```

## Run the symmetric simulator

```bash
python src/symmetric_scientific_review_simulator.py \
  --worlds 2000 \
  --papers 300 \
  --periods 30 \
  --output-dir symmetric_output
```

## Reproduce the final robustness results

```bash
python src/reproduce_robustness_experiments.py \
  --output-root reproduced_results
```

This command regenerates the committed symmetric and equal-total-labor summary tables with the reported fixed seed. Use `--world-level` to retain the larger per-world tables.

## Included result summaries

The `results/` directory contains compact CSV summaries from dynamic hidden-truth, realistic-versus-idealized, high-consequence and equity-sensitive, peer-review-favorable, symmetric, and equal-total-labor experiments.

Large world-level outputs are intentionally omitted. They can be regenerated with the fixed seeds and commands documented in the source and manuscript.

## Interpretation

These simulations are structural counterfactual models, not proof that any historical publication system is net beneficial or harmful. Their purpose is to identify the conditions under which each institution performs better and to expose the empirical quantities that must be measured.

The central conditional result is that when uncertain prepublication decisions reduce later attention, mandatory gating can recover less latent truth than continuous open evaluation, even when expert reviewers can materially improve manuscripts. Peer review becomes competitive when reviewers are unusually accurate and unbiased, rejected work remains highly discoverable, resubmission is easy, and delay and labor costs are small.

## Reproducibility

- Python 3
- NumPy
- pandas
- fixed random seeds in the scripts

Contributions, audits, alternative parameterizations, and adversarial tests are welcome.
