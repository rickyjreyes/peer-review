#!/usr/bin/env python3
"""Reproduce the two final robustness experiments reported in the manuscript.

Experiments
-----------
1. symmetric
   Gives every publication system the same defect-detection, manuscript-repair,
   harmful-revision, evidence, and expert-labor mechanisms.

2. equal-budget
   Deliberately favors peer review while charging initial review, resubmission,
   and downstream evaluation against exactly the same lifetime expert-action
   budget for every system.

The defaults reproduce the committed summary tables. World-level output is
optional because it is large and is not required to verify the reported means.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import symmetric_scientific_review_simulator as symmetric


DEFAULT_SEED = 20260806


def peer_favorable_equal_budget_parameters(
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Sample the locked peer-review-favorable parameter profile."""
    false_share = rng.uniform(0.35, 0.55)
    mixed_share = min(rng.uniform(0.10, 0.22), 0.82 - false_share)

    return {
        "false_share": false_share,
        "mixed_share": mixed_share,
        "fraud_share": rng.uniform(0.02, 0.06),
        "novelty_value_multiplier": rng.uniform(1.0, 3.0),
        "method_quality_mean": rng.uniform(0.45, 0.65),
        "statistics_quality_mean": rng.uniform(0.45, 0.65),
        "reporting_quality_mean": rng.uniform(0.50, 0.70),
        "reproducibility_mean": rng.uniform(0.45, 0.65),
        # Strong, but not perfect, reviewers.
        "expert_skill": rng.uniform(0.85, 1.15),
        "expert_noise": rng.uniform(0.85, 1.15),
        "defect_detection_rate": rng.uniform(0.35, 0.60),
        "repair_success_rate": rng.uniform(0.45, 0.70),
        "harmful_revision_rate": rng.uniform(0.01, 0.07),
        "repair_magnitude": rng.uniform(0.10, 0.24),
        "harm_magnitude": rng.uniform(0.02, 0.08),
        # Small but nonzero institutional biases.
        "prestige_bias": rng.uniform(0.05, 0.20),
        "novelty_penalty": rng.uniform(0.00, 0.15),
        "clarity_bias": rng.uniform(0.20, 0.45),
        "conformity_bias": rng.uniform(0.00, 0.12),
        "accept_threshold": rng.uniform(-0.10, 0.15),
        "reviewers_per_paper": 3,
        "rejected_attention_fraction": rng.uniform(0.50, 0.80),
        "resubmission_probability": rng.uniform(0.65, 0.90),
        # The lifetime total is enforced explicitly below.
        "expert_actions_per_paper": rng.uniform(0.75, 1.15),
        "peer_concentration": rng.uniform(0.7, 1.8),
        "open_concentration": rng.uniform(2.0, 4.5),
        "triage_concentration": rng.uniform(1.5, 3.8),
        "hybrid_concentration": rng.uniform(1.5, 3.8),
        "exploration_share": rng.uniform(0.08, 0.22),
        "popularity_bias": rng.uniform(0.45, 0.90),
        "evidence_noise": rng.uniform(0.95, 1.40),
        "learning_rate": rng.uniform(0.45, 0.80),
        "replication_probability": rng.uniform(0.10, 0.30),
        "fraud_detection_rate": rng.uniform(0.08, 0.25),
        "recognition_threshold": rng.uniform(0.72, 0.84),
        "withdraw_threshold": rng.uniform(0.05, 0.14),
        "min_withdraw_period": int(rng.integers(5, 10)),
        # Strong preference for avoiding false positives.
        "recovery_weight": rng.uniform(0.85, 1.05),
        "false_weight": rng.uniform(1.8, 2.6),
        "calibration_weight": rng.uniform(0.55, 0.95),
        "delay_weight": rng.uniform(0.00, 0.015),
        "labor_weight": 0.0,
    }


def simulate_equal_budget_world(
    rng: np.random.Generator,
    parameters: dict[str, float | int],
    papers_per_world: int,
    periods: int,
) -> list[dict[str, float | str]]:
    """Simulate one world with an exactly equal lifetime expert-action budget."""
    p = parameters
    papers = symmetric.initialize_papers(rng, p, papers_per_world)
    total_budget = int(
        round(float(p["expert_actions_per_paper"]) * papers_per_world * periods)
    )

    states: dict[str, dict] = {}
    spent: dict[str, int] = {}

    for system in symmetric.SYSTEMS:
        state = {
            "methods": papers["methods"].copy(),
            "statistics": papers["statistics"].copy(),
            "reporting": papers["reporting"].copy(),
            "reproducibility": papers["reproducibility"].copy(),
            "confidence": np.full(papers_per_world, 0.5),
            "active": np.ones(papers_per_world, dtype=bool),
            "accepted": np.ones(papers_per_world, dtype=bool),
            "labor": 0,
            "delay": np.zeros(papers_per_world),
            "cumulative_actions": np.zeros(papers_per_world, dtype=int),
            "first_recognition": np.full(papers_per_world, np.nan),
        }

        if system in ("peer_review", "hybrid"):
            initial = np.full(
                papers_per_world, int(p["reviewers_per_paper"]), dtype=int
            )
            symmetric.apply_expert_actions(rng, papers, state, p, initial)
            scores = symmetric.review_scores(rng, papers, state, p)
            state["accepted"] = scores > float(p["accept_threshold"])
            if system == "peer_review":
                state["delay"] += 1

        states[system] = state
        spent[system] = int(state["labor"])

    for period in range(periods):
        for system in symmetric.SYSTEMS:
            state = states[system]

            if system == "peer_review" and period > 0 and spent[system] < total_budget:
                rejected = ~state["accepted"]
                attempts = rejected & (
                    rng.random(papers_per_world)
                    < float(p["resubmission_probability"]) / periods
                )
                if np.any(attempts):
                    counts = np.zeros(papers_per_world, dtype=int)
                    max_attempts = max(
                        (total_budget - spent[system])
                        // int(p["reviewers_per_paper"]),
                        0,
                    )
                    attempt_indices = np.flatnonzero(attempts)[:max_attempts]
                    counts[attempt_indices] = int(p["reviewers_per_paper"])
                    if counts.sum():
                        symmetric.apply_expert_actions(rng, papers, state, p, counts)
                        spent[system] = int(state["labor"])
                        scores = symmetric.review_scores(rng, papers, state, p)
                        state["accepted"][attempt_indices] = (
                            scores[attempt_indices] > float(p["accept_threshold"])
                        )
                        state["delay"][attempt_indices] += 1

            remaining = total_budget - spent[system]
            periods_left = periods - period
            if remaining <= 0:
                actions = np.zeros(papers_per_world, dtype=int)
            else:
                period_budget = min(remaining, max(1, remaining // periods_left))
                actions = symmetric.allocate_actions(
                    rng, system, papers, state, p, period_budget
                )

            state["cumulative_actions"] += actions
            symmetric.apply_expert_actions(rng, papers, state, p, actions)
            spent[system] = int(state["labor"])
            symmetric.update_confidence(rng, papers, state, p, actions)

            recognized = (
                state["confidence"] >= float(p["recognition_threshold"])
            ) & state["active"]
            newly_recognized = recognized & np.isnan(state["first_recognition"])
            state["first_recognition"][newly_recognized] = period

            if period >= int(p["min_withdraw_period"]):
                enough_evidence = state["cumulative_actions"] >= 2
                withdraw = (
                    state["confidence"] < float(p["withdraw_threshold"])
                ) & enough_evidence
                state["active"][withdraw] = False

    total_true_value = papers["scientific_value"].sum() + 1e-12
    outcomes: list[dict[str, float | str]] = []

    for system in symmetric.SYSTEMS:
        state = states[system]
        confidence = state["confidence"]
        recognized = (
            confidence >= float(p["recognition_threshold"])
        ) & state["active"]

        true_value_recovered = (
            papers["scientific_value"][recognized].sum() / total_true_value
        )
        false_mask = papers["truth"] == 0
        false_recognition_rate = (
            (recognized & false_mask).sum() / max(false_mask.sum(), 1)
        )
        calibration_mse = np.average(
            (confidence - papers["truth"]) ** 2,
            weights=papers["intrinsic_value"],
        )

        score = (
            float(p["recovery_weight"]) * true_value_recovered
            - float(p["false_weight"]) * false_recognition_rate
            - float(p["calibration_weight"]) * calibration_mse
            - float(p["delay_weight"]) * np.mean(state["delay"]) / periods
        )

        outcomes.append(
            {
                "system": system,
                "score": float(score),
                "true_value_recovered": float(true_value_recovered),
                "false_recognition_rate": float(false_recognition_rate),
                "calibration_mse": float(calibration_mse),
                "lifetime_actions": float(spent[system]),
                "rejected_share": (
                    float((~state["accepted"]).mean())
                    if system == "peer_review"
                    else 0.0
                ),
            }
        )

    return outcomes


def run_symmetric(
    output_root: Path,
    seed: int = DEFAULT_SEED,
    worlds: int = 500,
    papers: int = 250,
    periods: int = 25,
    save_world_level: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the 125,000-paper symmetric robustness experiment."""
    config = symmetric.Config(
        seed=seed,
        worlds=worlds,
        papers=papers,
        periods=periods,
        output_dir=str(output_root / "symmetric"),
    )
    results, summary, win_rates = symmetric.run(config)
    output = output_root / "symmetric"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv")
    win_rates.to_csv(output / "win_rates.csv")
    if save_world_level:
        results.to_csv(output / "world_level_results.csv", index=False)
    return summary, win_rates


def run_equal_budget(
    output_root: Path,
    seed: int = DEFAULT_SEED,
    worlds: int = 800,
    papers: int = 220,
    periods: int = 24,
    save_world_level: bool = False,
) -> pd.DataFrame:
    """Reproduce the 176,000-paper equal-total-labor stress test."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for world in range(worlds):
        parameters = peer_favorable_equal_budget_parameters(rng)
        for outcome in simulate_equal_budget_world(
            rng, parameters, papers_per_world=papers, periods=periods
        ):
            rows.append({"world": world, **outcome})

    results = pd.DataFrame(rows)
    scores = results.pivot(index="world", columns="system", values="score")
    winners = scores.idxmax(axis=1)

    summary = results.groupby("system")[[
        "score",
        "true_value_recovered",
        "false_recognition_rate",
        "calibration_mse",
        "lifetime_actions",
        "rejected_share",
    ]].mean()
    summary["win_share"] = [(winners == system).mean() for system in summary.index]

    output = output_root / "equal_budget"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv")
    if save_world_level:
        results.to_csv(output / "world_level_results.csv", index=False)
    return summary


def parse_experiments(values: Iterable[str]) -> set[str]:
    requested = set(values)
    if "all" in requested:
        return {"symmetric", "equal-budget"}
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        action="append",
        choices=("all", "symmetric", "equal-budget"),
        default=None,
        help="Experiment to run. Repeat for more than one.",
    )
    parser.add_argument("--output-root", default="reproduced_results")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--world-level",
        action="store_true",
        help="Also save the large per-world output tables.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small validation job instead of the reported sample sizes.",
    )
    args = parser.parse_args()

    experiments = parse_experiments(args.experiment or ["all"])
    output_root = Path(args.output_root)

    if args.quick:
        symmetric_size = (8, 60, 6)
        equal_size = (8, 60, 6)
    else:
        symmetric_size = (500, 250, 25)
        equal_size = (800, 220, 24)

    if "symmetric" in experiments:
        summary, win_rates = run_symmetric(
            output_root,
            seed=args.seed,
            worlds=symmetric_size[0],
            papers=symmetric_size[1],
            periods=symmetric_size[2],
            save_world_level=args.world_level,
        )
        print("\nSymmetric experiment")
        print(summary.round(4).to_string())
        print("\nWin rates")
        print(win_rates.round(4).to_string())

    if "equal-budget" in experiments:
        summary = run_equal_budget(
            output_root,
            seed=args.seed,
            worlds=equal_size[0],
            papers=equal_size[1],
            periods=equal_size[2],
            save_world_level=args.world_level,
        )
        print("\nEqual-total-labor experiment")
        print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
