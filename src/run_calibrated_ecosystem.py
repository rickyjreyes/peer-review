#!/usr/bin/env python3
"""Run the publication ecosystem with the calibrated reviewer submodel.

The reviewer decision mechanism is taken from ``calibrate_empirical_model.py``
and the committed best-fitting parameter set. Ecosystem-level quantities that
were not identified by the calibration, including rejected-paper attention,
resubmission, evidence accumulation, and social-value weights, continue to be
sampled from the canonical structural ranges.

This is therefore a calibration-propagated ecosystem experiment, not a claim
that every parameter is an empirical estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import scientific_publication_simulator as ecosystem


DEFAULT_CALIBRATION = Path("results/empirical_calibration/best_fit_parameters.json")
DEFAULT_OUTPUT = Path("results/calibrated_ecosystem")
ERRORS_PER_PAPER = 9

_ORIGINAL_INITIALIZE = ecosystem.initialize_papers
_CALIBRATED_THETA: dict[str, float] = {}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def load_theta(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("Expected one best-fitting parameter record")
    return {key: float(value) for key, value in payload[0].items()}


def calibrated_initialize_papers(
    rng: np.random.Generator,
    parameters: dict,
    count: int,
) -> dict:
    papers = _ORIGINAL_INITIALIZE(rng, parameters, count)
    truth = papers["truth"]

    true_rate = float(parameters["true_major_error_probability"])
    partial_rate = float(parameters["partial_major_error_probability"])
    false_rate = float(parameters["false_major_error_probability"])
    error_probability = np.where(
        truth >= 0.999,
        true_rate,
        np.where(truth <= 0.001, false_rate, partial_rate),
    )
    error_probability = np.clip(
        error_probability + 0.08 * papers["fraud"].astype(float),
        0.01,
        0.98,
    )

    papers["major_errors"] = (
        rng.random((count, ERRORS_PER_PAPER))
        < error_probability[:, None]
    )
    papers["error_difficulty"] = rng.normal(
        size=(count, ERRORS_PER_PAPER)
    )

    # Outcome sign is separated from latent truth so that the calibrated
    # positive-result preference is represented as a bias rather than evidence.
    papers["positive_result"] = (
        rng.random(count) < float(parameters["positive_result_share"])
    )
    return papers


def calibrated_formal_review(
    rng: np.random.Generator,
    papers: dict,
    parameters: dict,
    clarity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    theta = _CALIBRATED_THETA
    count = len(papers["truth"])
    scores: list[np.ndarray] = []

    truth_signal = (papers["truth"] - 0.5) * 2.0
    positive_signal = papers["positive_result"].astype(float)

    for _ in range(int(parameters["reviewers_per_paper"])):
        ability = rng.normal(
            0.0,
            theta["reviewer_ability_sd"],
            count,
        )
        detection_logits = (
            theta["detection_intercept"]
            + ability[:, None]
            - theta["error_difficulty_sd"]
            * papers["error_difficulty"]
        )
        detected = papers["major_errors"] & (
            rng.random((count, ERRORS_PER_PAPER))
            < sigmoid(detection_logits)
        )
        detected_fraction = detected.mean(axis=1)

        institutional_terms = (
            parameters["clarity_bias"] * (clarity - 0.5)
            + parameters["prestige_bias"]
            * (papers["prestige"] - 0.5)
            - parameters["novelty_penalty"] * papers["novelty"]
            + parameters["career_conformity"]
            * (papers["prestige"] - papers["novelty"])
        )

        score = (
            theta["trial_base_merit"]
            + theta["merit_signal"] * truth_signal
            + theta["positive_outcome_bias"] * positive_signal
            - theta["detected_error_penalty"] * detected_fraction
            + institutional_terms
            + rng.normal(0.0, theta["rating_noise_sd"], count)
        )
        scores.append(score)

    mean_score = np.mean(scores, axis=0)
    accepted = mean_score > theta["recommendation_threshold"]
    return accepted, mean_score


def sample_calibration_propagated_parameters(
    rng: np.random.Generator,
) -> dict:
    parameters = ecosystem.sample_parameters(rng)

    # The calibration identifies reviewer behavior, not manuscript defect
    # prevalence. These ranges are therefore swept explicitly rather than
    # presented as empirical estimates.
    parameters.update(
        {
            "true_major_error_probability": rng.uniform(0.05, 0.25),
            "partial_major_error_probability": rng.uniform(0.25, 0.55),
            "false_major_error_probability": rng.uniform(0.50, 0.85),
            "positive_result_share": rng.uniform(0.35, 0.65),
            # Metadata mirrors the calibrated latent scales. The replacement
            # formal-review function above uses the full calibrated mechanism.
            "reviewer_skill": _CALIBRATED_THETA["merit_signal"],
            "review_noise": _CALIBRATED_THETA["rating_noise_sd"],
            "accept_threshold": _CALIBRATED_THETA[
                "recommendation_threshold"
            ],
        }
    )
    return parameters


def run(
    seed: int,
    worlds: int,
    papers_per_world: int,
    periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for world in range(worlds):
        parameters = sample_calibration_propagated_parameters(rng)
        outcomes = ecosystem.simulate_world(
            rng,
            parameters,
            papers_per_world,
            periods,
        )
        for outcome in outcomes:
            rows.append({"world": world, **outcome})

    results = pd.DataFrame(rows)
    score_table = results.pivot(
        index="world",
        columns="system",
        values="score",
    )
    winners = score_table.idxmax(axis=1)

    metric_columns = [
        "score",
        "true_value_recovered",
        "true_recognition_rate",
        "mixed_recognition_rate",
        "false_recognition_rate",
        "calibration_mse",
        "attention_coverage",
        "attention_gini",
        "mean_time_to_recognition",
        "review_labor_per_paper",
        "mean_publication_delay",
        "rejected_final_share",
    ]
    summary = results.groupby("system")[metric_columns].agg(
        ["mean", "std", "median"]
    )
    win_rates = pd.DataFrame(
        {
            "system": ecosystem.SYSTEMS,
            "win_share": [
                float((winners == system).mean())
                for system in ecosystem.SYSTEMS
            ],
        }
    ).set_index("system")
    return results, summary, win_rates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--worlds", type=int, default=500)
    parser.add_argument("--papers", type=int, default=250)
    parser.add_argument("--periods", type=int, default=25)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--save-world-level", action="store_true")
    args = parser.parse_args()

    global _CALIBRATED_THETA
    _CALIBRATED_THETA = load_theta(args.calibration)

    # Replace only the paper-defect initialization and formal-review mechanism.
    # All downstream ecosystem dynamics remain those of the canonical model.
    ecosystem.initialize_papers = calibrated_initialize_papers
    ecosystem.formal_review = calibrated_formal_review

    results, summary, win_rates = run(
        seed=args.seed,
        worlds=args.worlds,
        papers_per_world=args.papers,
        periods=args.periods,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv")
    win_rates.to_csv(args.output_dir / "win_rates.csv")
    if args.save_world_level:
        results.to_csv(
            args.output_dir / "world_level_results.csv",
            index=False,
        )

    metadata = {
        "seed": args.seed,
        "worlds": args.worlds,
        "papers_per_world": args.papers,
        "periods": args.periods,
        "unique_papers": args.worlds * args.papers,
        "calibration_file": str(args.calibration),
        "calibrated_reviewer_parameters": _CALIBRATED_THETA,
        "interpretive_boundary": (
            "Reviewer detection, disagreement, recommendation behavior, and "
            "positive-outcome bias are propagated from the empirical "
            "calibration. Rejection attention, resubmission, defect prevalence, "
            "evidence dynamics, and utility weights remain structural sweeps."
        ),
    }
    with (args.output_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("Calibration-propagated ecosystem simulation")
    print(f"Worlds: {args.worlds:,}")
    print(f"Unique papers: {args.worlds * args.papers:,}")
    print("\nWin shares")
    print(win_rates.round(4).to_string())
    print("\nMean outcomes")
    means = summary.xs("mean", axis=1, level=1)
    print(
        means[
            [
                "score",
                "true_value_recovered",
                "false_recognition_rate",
                "calibration_mse",
                "rejected_final_share",
            ]
        ]
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
