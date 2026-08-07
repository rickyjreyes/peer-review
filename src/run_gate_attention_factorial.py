#!/usr/bin/env python3
"""Run the preregistered gate × downstream-attention factorial.

The experiment keeps formal reviewer behavior fixed to the committed empirical
calibration and separates two mechanisms:

1. Gate: a negative formal-review decision reduces downstream attention and adds
   publication delay, versus the same review process being non-vetoing.
2. Attention allocation: concentrated allocation versus transparent triage with
   randomized exploration.

All four arms retain formal review and review-triggered revision so the gate
contrast isolates veto consequences rather than expert criticism itself.

A secondary sweep varies the exploration share from 0 to 1 under both gate
conditions. The prediction was committed before this runner was added in
PREDICTION_ATTENTION_FACTORIAL.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import scientific_publication_simulator as ecosystem
import run_calibrated_ecosystem as calibrated


ARMS = (
    (True, "concentrated"),
    (True, "exploration"),
    (False, "concentrated"),
    (False, "exploration"),
)


def arm_name(gate: bool, attention: str) -> str:
    return f"{'gate' if gate else 'no_gate'}__{attention}"


def allocate_factorial_attention(
    rng: np.random.Generator,
    gate: bool,
    attention: str,
    papers: dict,
    p: dict,
    confidence: np.ndarray,
    available: np.ndarray,
    formally_accepted: np.ndarray,
    total_budget: int,
) -> np.ndarray:
    prestige = papers["prestige"]
    clarity = papers["clarity"]
    novelty = papers["novelty"]
    popularity = papers["popularity"]

    if attention == "concentrated":
        appeal = (
            1.10 * confidence
            + p["peer_attention_prestige"] * prestige
            + 0.20 * clarity
            + 0.15 * popularity
        )
        weights = ecosystem.normalized_softmax(
            appeal, available, p["peer_concentration"]
        )
    elif attention == "exploration":
        merit = (
            0.30 * clarity
            + 0.25 * novelty
            + 0.25 * confidence
            + 0.15 * (1.0 - prestige)
            + 0.05 * popularity
        )
        weights = ecosystem.normalized_softmax(
            merit, available, p["triage_concentration"]
        )
        exploration = np.zeros_like(weights)
        if np.any(available):
            exploration[available] = 1.0 / available.sum()
        weights = (
            (1.0 - p["exploration_share"]) * weights
            + p["exploration_share"] * exploration
        )
    else:
        raise ValueError(attention)

    # The current repository models a soft publication gate: rejected work remains
    # technically available but is down-weighted until a positive formal decision.
    # The no-gate arm keeps the identical review label but does not let it suppress
    # downstream attention.
    if gate:
        weights[available & (~formally_accepted)] *= p[
            "rejected_attention_fraction"
        ]
        if weights.sum() > 0:
            weights /= weights.sum()

    if weights.sum() <= 0:
        return np.zeros_like(weights, dtype=int)
    weights /= weights.sum()
    return rng.multinomial(total_budget, weights)


def simulate_arm(
    papers: dict,
    parameters: dict,
    initial_accepted: np.ndarray,
    initial_review_score: np.ndarray,
    gate: bool,
    attention: str,
    papers_per_world: int,
    periods: int,
    revision_seed: int,
    dynamics_seed: int,
) -> dict:
    n = papers_per_world
    p = parameters
    revision_rng = np.random.default_rng(revision_seed)
    rng = np.random.default_rng(dynamics_seed)

    confidence = np.full(n, 0.5)
    clarity = papers["clarity"].copy()
    accepted = initial_accepted.copy()
    review_score = initial_review_score.copy()
    active = np.ones(n, dtype=bool)
    first_recognition_period = np.full(n, np.nan)
    cumulative_evaluations = np.zeros(n, dtype=int)
    reviewers = int(p["reviewers_per_paper"])
    review_labor = n * reviewers
    publication_delay = np.ones(n) if gate else np.zeros(n)
    initial_rejected = ~accepted.copy()
    attempts_per_paper = np.zeros(n, dtype=int)

    budget = max(1, int(round(p["evals_per_paper_per_period"] * n)))

    for t in range(periods):
        # Formal-review revision is retained in every arm. Thus no-gate means
        # reviewed-but-non-vetoing, not absence of expert criticism.
        if t > 0:
            rejected = ~accepted
            attempts = rejected & (
                revision_rng.random(n)
                < p["resubmission_probability"] / periods
            )
            if np.any(attempts):
                clarity[attempts] = np.clip(
                    clarity[attempts]
                    + p["revision_effectiveness"]
                    * revision_rng.beta(2, 5, attempts.sum()),
                    0,
                    1,
                )
                accepted_new, score_new = calibrated.calibrated_formal_review(
                    revision_rng, papers, p, clarity
                )
                accepted[attempts] = accepted_new[attempts]
                review_score[attempts] = score_new[attempts]
                attempts_per_paper[attempts] += 1
                review_labor += attempts.sum() * reviewers
                if gate:
                    publication_delay[attempts] += 1

        counts = allocate_factorial_attention(
            rng=rng,
            gate=gate,
            attention=attention,
            papers=papers,
            p=p,
            confidence=confidence,
            available=active,
            formally_accepted=accepted,
            total_budget=budget,
        )
        cumulative_evaluations += counts
        confidence = ecosystem.update_confidence(
            rng, papers, p, confidence, counts
        )

        recognized_now = (
            (confidence >= p["recognition_threshold"])
            & active
        )
        newly = recognized_now & np.isnan(first_recognition_period)
        first_recognition_period[newly] = t

        if t >= p["min_withdraw_period"]:
            enough_evidence = cumulative_evaluations >= 2
            withdraw = (
                (confidence < p["withdraw_threshold"])
                & enough_evidence
            )
            active[withdraw] = False

    total_true_value = papers["scientific_value"].sum() + 1e-12
    recognized = (
        (confidence >= p["recognition_threshold"])
        & active
    )
    weighted_calibration = np.average(
        (confidence - papers["truth"]) ** 2,
        weights=papers["intrinsic_value"],
    )
    true_value_recovered = (
        papers["scientific_value"][recognized].sum()
        / total_true_value
    )
    false_mask = papers["truth"] == 0.0
    mixed_mask = papers["truth"] == 0.5
    true_mask = papers["truth"] == 1.0
    false_recognition = (
        (recognized & false_mask).sum()
        / max(false_mask.sum(), 1)
    )
    mixed_recognition = (
        (recognized & mixed_mask).sum()
        / max(mixed_mask.sum(), 1)
    )
    true_recognition = (
        (recognized & true_mask).sum()
        / max(true_mask.sum(), 1)
    )

    recognized_true = recognized & (papers["truth"] > 0)
    mean_time_to_recognition = math.nan
    if np.any(recognized_true):
        mean_time_to_recognition = float(
            np.nanmean(first_recognition_period[recognized_true])
        )

    attention_coverage = float((cumulative_evaluations > 0).mean())
    attention_gini = ecosystem.gini(cumulative_evaluations.astype(float))
    delay_cost = (
        p["delay_weight"] * np.mean(publication_delay) / max(periods, 1)
    )
    labor_cost = (
        p["labor_weight"] * review_labor / max(n * periods, 1)
    )
    score = (
        p["recovery_weight"] * true_value_recovered
        - p["false_weight"] * false_recognition
        - p["calibration_weight"] * weighted_calibration
        - delay_cost
        - labor_cost
    )

    if np.any(initial_rejected):
        initial_reject_resubmit_fraction = float(
            (attempts_per_paper[initial_rejected] > 0).mean()
        )
        mean_attempts_initial_rejected = float(
            attempts_per_paper[initial_rejected].mean()
        )
        mean_delay_initial_rejected = float(
            publication_delay[initial_rejected].mean()
        )
        mean_review_labor_initial_rejected = float(
            (
                reviewers
                + attempts_per_paper[initial_rejected] * reviewers
            ).mean()
        )
    else:
        initial_reject_resubmit_fraction = math.nan
        mean_attempts_initial_rejected = math.nan
        mean_delay_initial_rejected = math.nan
        mean_review_labor_initial_rejected = math.nan

    return {
        "arm": arm_name(gate, attention),
        "gate": int(gate),
        "attention": attention,
        "score": float(score),
        "true_value_recovered": float(true_value_recovered),
        "true_recognition_rate": float(true_recognition),
        "mixed_recognition_rate": float(mixed_recognition),
        "false_recognition_rate": float(false_recognition),
        "calibration_mse": float(weighted_calibration),
        "attention_coverage": attention_coverage,
        "attention_gini": float(attention_gini),
        "mean_time_to_recognition": mean_time_to_recognition,
        "review_labor_per_paper": float(review_labor / n),
        "mean_publication_delay": float(np.mean(publication_delay)),
        "initial_rejection_share": float(initial_rejected.mean()),
        "final_rejection_share": float((~accepted).mean()),
        "initial_reject_resubmit_fraction": initial_reject_resubmit_fraction,
        "mean_attempts_initial_rejected": mean_attempts_initial_rejected,
        "mean_delay_initial_rejected": mean_delay_initial_rejected,
        "mean_review_labor_initial_rejected": mean_review_labor_initial_rejected,
    }


def paired_effect_summary(values: np.ndarray, name: str) -> dict:
    values = np.asarray(values, dtype=float)
    n = int(np.isfinite(values).sum())
    mean = float(np.nanmean(values))
    sd = float(np.nanstd(values, ddof=1))
    se = sd / math.sqrt(n)
    return {
        "effect": name,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "worlds": n,
    }


def generate_world(
    seed: int,
    world: int,
    papers_per_world: int,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    param_rng = np.random.default_rng(seed + 1000003 * world)
    parameters = calibrated.sample_calibration_propagated_parameters(param_rng)
    papers = calibrated.calibrated_initialize_papers(
        param_rng, parameters, papers_per_world
    )
    review_rng = np.random.default_rng(seed + 1300033 * world + 7)
    accepted, score = calibrated.calibrated_formal_review(
        review_rng, papers, parameters, papers["clarity"].copy()
    )
    return papers, parameters, accepted, score


def run_factorial(
    seed: int,
    worlds: int,
    papers_per_world: int,
    periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for world in range(worlds):
        papers, parameters, accepted, score = generate_world(
            seed, world, papers_per_world
        )
        # The same revision and dynamics seeds are reused across arms within a
        # world to provide paired common-random-number comparisons.
        revision_seed = seed + 1700021 * world + 11
        dynamics_seed = seed + 1900037 * world + 13
        for gate, attention in ARMS:
            outcome = simulate_arm(
                papers=copy.deepcopy(papers),
                parameters=copy.deepcopy(parameters),
                initial_accepted=accepted,
                initial_review_score=score,
                gate=gate,
                attention=attention,
                papers_per_world=papers_per_world,
                periods=periods,
                revision_seed=revision_seed,
                dynamics_seed=dynamics_seed,
            )
            rows.append({"world": world, **outcome})

    world_results = pd.DataFrame(rows)
    metrics = [
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
        "initial_rejection_share",
        "final_rejection_share",
        "initial_reject_resubmit_fraction",
        "mean_attempts_initial_rejected",
        "mean_delay_initial_rejected",
        "mean_review_labor_initial_rejected",
    ]
    arm_summary = world_results.groupby("arm")[metrics].agg(
        ["mean", "std", "median"]
    )

    pivot = world_results.pivot(
        index="world", columns="arm", values="true_value_recovered"
    )
    gc = pivot["gate__concentrated"].to_numpy()
    ge = pivot["gate__exploration"].to_numpy()
    nc = pivot["no_gate__concentrated"].to_numpy()
    ne = pivot["no_gate__exploration"].to_numpy()

    gate_main = 0.5 * ((gc - nc) + (ge - ne))
    attention_main = 0.5 * ((ge - gc) + (ne - nc))
    interaction = (ge - gc) - (ne - nc)
    gate_under_concentrated = gc - nc
    gate_under_exploration = ge - ne
    attention_under_gate = ge - gc
    attention_under_no_gate = ne - nc

    effects = pd.DataFrame(
        [
            paired_effect_summary(gate_main, "gate_main_effect"),
            paired_effect_summary(attention_main, "exploration_main_effect"),
            paired_effect_summary(interaction, "gate_x_exploration_interaction"),
            paired_effect_summary(
                gate_under_concentrated, "gate_effect_under_concentrated"
            ),
            paired_effect_summary(
                gate_under_exploration, "gate_effect_under_exploration"
            ),
            paired_effect_summary(
                attention_under_gate, "exploration_effect_under_gate"
            ),
            paired_effect_summary(
                attention_under_no_gate, "exploration_effect_under_no_gate"
            ),
        ]
    )
    return world_results, arm_summary, effects


def run_exploration_sweep(
    seed: int,
    worlds: int,
    papers_per_world: int,
    periods: int,
    step: float,
) -> pd.DataFrame:
    shares = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)
    rows: list[dict] = []
    for world in range(worlds):
        papers, parameters, accepted, score = generate_world(
            seed + 70000001, world, papers_per_world
        )
        revision_seed = seed + 71000003 * world + 19
        dynamics_seed = seed + 73000007 * world + 23
        for share in shares:
            for gate in (False, True):
                p = copy.deepcopy(parameters)
                p["exploration_share"] = float(share)
                outcome = simulate_arm(
                    papers=copy.deepcopy(papers),
                    parameters=p,
                    initial_accepted=accepted,
                    initial_review_score=score,
                    gate=gate,
                    attention="exploration",
                    papers_per_world=papers_per_world,
                    periods=periods,
                    revision_seed=revision_seed,
                    dynamics_seed=dynamics_seed,
                )
                rows.append(
                    {
                        "world": world,
                        "gate": int(gate),
                        "exploration_share": float(share),
                        "true_value_recovered": outcome[
                            "true_value_recovered"
                        ],
                        "false_recognition_rate": outcome[
                            "false_recognition_rate"
                        ],
                        "calibration_mse": outcome["calibration_mse"],
                        "attention_gini": outcome["attention_gini"],
                    }
                )
    df = pd.DataFrame(rows)
    summary = df.groupby(["gate", "exploration_share"]).agg(
        true_value_recovered=("true_value_recovered", "mean"),
        true_value_sd=("true_value_recovered", "std"),
        false_recognition_rate=("false_recognition_rate", "mean"),
        calibration_mse=("calibration_mse", "mean"),
        attention_gini=("attention_gini", "mean"),
    ).reset_index()
    summary["true_value_se"] = summary["true_value_sd"] / math.sqrt(worlds)
    summary["ci_low"] = (
        summary["true_value_recovered"] - 1.96 * summary["true_value_se"]
    )
    summary["ci_high"] = (
        summary["true_value_recovered"] + 1.96 * summary["true_value_se"]
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--worlds", type=int, default=300)
    parser.add_argument("--sweep-worlds", type=int, default=150)
    parser.add_argument("--papers", type=int, default=250)
    parser.add_argument("--periods", type=int, default=25)
    parser.add_argument("--exploration-step", type=float, default=0.10)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=calibrated.DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gate_attention_factorial"),
    )
    args = parser.parse_args()

    calibrated._CALIBRATED_THETA = calibrated.load_theta(args.calibration)

    world_results, arm_summary, effects = run_factorial(
        seed=args.seed,
        worlds=args.worlds,
        papers_per_world=args.papers,
        periods=args.periods,
    )
    exploration_summary = run_exploration_sweep(
        seed=args.seed,
        worlds=args.sweep_worlds,
        papers_per_world=args.papers,
        periods=args.periods,
        step=args.exploration_step,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    world_results.to_csv(
        args.output_dir / "factorial_world_results.csv", index=False
    )
    arm_summary.to_csv(args.output_dir / "arm_summary.csv")
    effects.to_csv(args.output_dir / "factorial_effects.csv", index=False)
    exploration_summary.to_csv(
        args.output_dir / "exploration_sweep.csv", index=False
    )

    metadata = {
        "seed": args.seed,
        "factorial_worlds": args.worlds,
        "sweep_worlds": args.sweep_worlds,
        "papers_per_world": args.papers,
        "periods": args.periods,
        "exploration_step": args.exploration_step,
        "prediction_file": "PREDICTION_ATTENTION_FACTORIAL.md",
        "gate_definition": (
            "Same calibrated formal review and revision in all arms; gated arms "
            "down-weight unaccepted work by rejected_attention_fraction and incur "
            "publication delay, while no-gate arms keep review non-vetoing."
        ),
        "interpretive_boundary": (
            "This is a mechanism-isolation experiment in the current calibrated "
            "architecture. It is not an empirical estimate of real-world gate or "
            "exploration effects."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    means = arm_summary.xs("mean", axis=1, level=1)
    print("Gate × attention factorial: mean arm outcomes")
    print(
        means[
            [
                "true_value_recovered",
                "false_recognition_rate",
                "calibration_mse",
                "initial_rejection_share",
                "final_rejection_share",
                "mean_publication_delay",
                "initial_reject_resubmit_fraction",
                "mean_delay_initial_rejected",
            ]
        ].round(5).to_string()
    )
    print("\nPaired factorial effects on true-value recovery")
    print(effects.round(5).to_string(index=False))
    print("\nExploration-share sweep")
    print(exploration_summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
