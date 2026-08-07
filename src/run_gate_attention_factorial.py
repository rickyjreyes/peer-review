#!/usr/bin/env python3
"""Run the preregistered 2x2 gate x downstream-attention mechanism test.

The prediction was committed beforehand in PREDICTION_ATTENTION_FACTORIAL.md.
Reviewer behavior is fixed to the committed empirical calibration. All four arms
receive the same formal review and review-triggered revision; the gate factor only
changes whether a negative review suppresses downstream attention and incurs
publication delay. The attention factor compares the current peer-style
concentrated allocation rule with the current open-triage allocation rule.

A secondary sweep varies only the randomized-exploration share inside the triage
allocation rule from 0 to 1. This distinguishes the triage scoring rule itself
from the additional randomized exploration component.
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
    (True, "triage"),
    (False, "concentrated"),
    (False, "triage"),
)


def arm_name(gate: bool, attention: str) -> str:
    return f"{'gate' if gate else 'no_gate'}__{attention}"


def allocate_attention(
    rng: np.random.Generator,
    gate: bool,
    attention: str,
    papers: dict,
    p: dict,
    confidence: np.ndarray,
    available: np.ndarray,
    accepted: np.ndarray,
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
    elif attention == "triage":
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
        explore = np.zeros_like(weights)
        if np.any(available):
            explore[available] = 1.0 / available.sum()
        weights = (
            (1.0 - p["exploration_share"]) * weights
            + p["exploration_share"] * explore
        )
    else:
        raise ValueError(attention)

    if gate:
        weights[available & (~accepted)] *= p["rejected_attention_fraction"]
        if weights.sum() > 0:
            weights /= weights.sum()

    if weights.sum() <= 0:
        return np.zeros_like(weights, dtype=int)
    weights /= weights.sum()
    return rng.multinomial(total_budget, weights)


def simulate_arm(
    papers: dict,
    p: dict,
    initial_accepted: np.ndarray,
    initial_review_score: np.ndarray,
    gate: bool,
    attention: str,
    n: int,
    periods: int,
    revision_seed: int,
    dynamics_seed: int,
) -> dict:
    revision_rng = np.random.default_rng(revision_seed)
    rng = np.random.default_rng(dynamics_seed)

    confidence = np.full(n, 0.5)
    clarity = papers["clarity"].copy()
    accepted = initial_accepted.copy()
    review_score = initial_review_score.copy()
    active = np.ones(n, dtype=bool)
    first_recognition = np.full(n, np.nan)
    cumulative_evals = np.zeros(n, dtype=int)
    reviewers = int(p["reviewers_per_paper"])
    review_labor = n * reviewers
    publication_delay = np.ones(n) if gate else np.zeros(n)
    initial_rejected = ~accepted.copy()
    attempts_per_paper = np.zeros(n, dtype=int)

    budget = max(1, int(round(p["evals_per_paper_per_period"] * n)))

    for t in range(periods):
        # Expert review/revision is identical in all arms; only veto consequences differ.
        if t > 0:
            rejected = ~accepted
            attempts = rejected & (
                revision_rng.random(n) < p["resubmission_probability"] / periods
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
                review_labor += int(attempts.sum()) * reviewers
                if gate:
                    publication_delay[attempts] += 1

        counts = allocate_attention(
            rng, gate, attention, papers, p, confidence, active, accepted, budget
        )
        cumulative_evals += counts
        confidence = ecosystem.update_confidence(
            rng, papers, p, confidence, counts
        )

        recognized_now = (confidence >= p["recognition_threshold"]) & active
        newly = recognized_now & np.isnan(first_recognition)
        first_recognition[newly] = t

        if t >= p["min_withdraw_period"]:
            enough = cumulative_evals >= 2
            withdraw = (confidence < p["withdraw_threshold"]) & enough
            active[withdraw] = False

    recognized = (confidence >= p["recognition_threshold"]) & active
    total_true_value = papers["scientific_value"].sum() + 1e-12
    true_value = papers["scientific_value"][recognized].sum() / total_true_value
    weighted_calibration = np.average(
        (confidence - papers["truth"]) ** 2,
        weights=papers["intrinsic_value"],
    )

    false_mask = papers["truth"] == 0.0
    mixed_mask = papers["truth"] == 0.5
    true_mask = papers["truth"] == 1.0
    false_recognition = (recognized & false_mask).sum() / max(false_mask.sum(), 1)
    mixed_recognition = (recognized & mixed_mask).sum() / max(mixed_mask.sum(), 1)
    true_recognition = (recognized & true_mask).sum() / max(true_mask.sum(), 1)

    recognized_true = recognized & (papers["truth"] > 0)
    mean_time = math.nan
    if np.any(recognized_true):
        mean_time = float(np.nanmean(first_recognition[recognized_true]))

    delay_cost = p["delay_weight"] * np.mean(publication_delay) / max(periods, 1)
    labor_cost = p["labor_weight"] * review_labor / max(n * periods, 1)
    score = (
        p["recovery_weight"] * true_value
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
            (reviewers + attempts_per_paper[initial_rejected] * reviewers).mean()
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
        "true_value_recovered": float(true_value),
        "true_recognition_rate": float(true_recognition),
        "mixed_recognition_rate": float(mixed_recognition),
        "false_recognition_rate": float(false_recognition),
        "calibration_mse": float(weighted_calibration),
        "attention_coverage": float((cumulative_evals > 0).mean()),
        "attention_gini": float(ecosystem.gini(cumulative_evals.astype(float))),
        "mean_time_to_recognition": mean_time,
        "review_labor_per_paper": float(review_labor / n),
        "mean_publication_delay": float(np.mean(publication_delay)),
        "initial_rejection_share": float(initial_rejected.mean()),
        "final_rejection_share": float((~accepted).mean()),
        "initial_reject_resubmit_fraction": initial_reject_resubmit_fraction,
        "mean_attempts_initial_rejected": mean_attempts_initial_rejected,
        "mean_delay_initial_rejected": mean_delay_initial_rejected,
        "mean_review_labor_initial_rejected": mean_review_labor_initial_rejected,
    }


def generate_world(seed: int, world: int, n: int):
    param_rng = np.random.default_rng(seed + 1000003 * world)
    p = calibrated.sample_calibration_propagated_parameters(param_rng)
    papers = calibrated.calibrated_initialize_papers(param_rng, p, n)
    review_rng = np.random.default_rng(seed + 1300033 * world + 7)
    accepted, review_score = calibrated.calibrated_formal_review(
        review_rng, papers, p, papers["clarity"].copy()
    )
    return papers, p, accepted, review_score


def effect_summary(values: np.ndarray, name: str, outcome: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / math.sqrt(n)
    return {
        "outcome": outcome,
        "effect": name,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        "worlds": n,
    }


def factorial_effects(world_results: pd.DataFrame, outcome: str) -> list[dict]:
    pivot = world_results.pivot(index="world", columns="arm", values=outcome)
    gc = pivot["gate__concentrated"].to_numpy()
    gt = pivot["gate__triage"].to_numpy()
    nc = pivot["no_gate__concentrated"].to_numpy()
    nt = pivot["no_gate__triage"].to_numpy()
    return [
        effect_summary(0.5 * ((gc - nc) + (gt - nt)), "gate_main_effect", outcome),
        effect_summary(0.5 * ((gt - gc) + (nt - nc)), "triage_allocation_main_effect", outcome),
        effect_summary((gt - gc) - (nt - nc), "gate_x_triage_interaction", outcome),
        effect_summary(gc - nc, "gate_effect_under_concentrated", outcome),
        effect_summary(gt - nt, "gate_effect_under_triage", outcome),
        effect_summary(gt - gc, "triage_effect_under_gate", outcome),
        effect_summary(nt - nc, "triage_effect_under_no_gate", outcome),
    ]


def run_factorial(seed: int, worlds: int, n: int, periods: int):
    rows = []
    for world in range(worlds):
        papers, p, accepted, review_score = generate_world(seed, world, n)
        revision_seed = seed + 1700021 * world + 11
        dynamics_seed = seed + 1900037 * world + 13
        for gate, attention in ARMS:
            outcome = simulate_arm(
                copy.deepcopy(papers), copy.deepcopy(p), accepted, review_score,
                gate, attention, n, periods, revision_seed, dynamics_seed
            )
            rows.append({"world": world, **outcome})

    world_results = pd.DataFrame(rows)
    metrics = [
        "score", "true_value_recovered", "true_recognition_rate",
        "mixed_recognition_rate", "false_recognition_rate", "calibration_mse",
        "attention_coverage", "attention_gini", "mean_time_to_recognition",
        "review_labor_per_paper", "mean_publication_delay",
        "initial_rejection_share", "final_rejection_share",
        "initial_reject_resubmit_fraction", "mean_attempts_initial_rejected",
        "mean_delay_initial_rejected", "mean_review_labor_initial_rejected",
    ]
    arm_summary = world_results.groupby("arm")[metrics].agg(["mean", "std", "median"])
    effects = pd.DataFrame(
        factorial_effects(world_results, "true_value_recovered")
        + factorial_effects(world_results, "score")
    )
    return world_results, arm_summary, effects


def run_exploration_sweep(seed: int, worlds: int, n: int, periods: int, step: float):
    shares = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)
    rows = []
    for world in range(worlds):
        papers, p, accepted, review_score = generate_world(seed + 70000001, world, n)
        revision_seed = seed + 71000003 * world + 19
        dynamics_seed = seed + 73000007 * world + 23
        for share in shares:
            for gate in (False, True):
                p2 = copy.deepcopy(p)
                p2["exploration_share"] = float(share)
                outcome = simulate_arm(
                    copy.deepcopy(papers), p2, accepted, review_score,
                    gate, "triage", n, periods, revision_seed, dynamics_seed
                )
                rows.append({
                    "world": world,
                    "gate": int(gate),
                    "exploration_share": float(share),
                    **{k: outcome[k] for k in (
                        "true_value_recovered", "false_recognition_rate",
                        "calibration_mse", "attention_gini"
                    )},
                })
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["gate", "exploration_share"]).agg(
        true_value_recovered=("true_value_recovered", "mean"),
        true_value_sd=("true_value_recovered", "std"),
        false_recognition_rate=("false_recognition_rate", "mean"),
        calibration_mse=("calibration_mse", "mean"),
        attention_gini=("attention_gini", "mean"),
    ).reset_index()
    summary["true_value_se"] = summary["true_value_sd"] / math.sqrt(worlds)
    summary["ci_low"] = summary["true_value_recovered"] - 1.96 * summary["true_value_se"]
    summary["ci_high"] = summary["true_value_recovered"] + 1.96 * summary["true_value_se"]
    return raw, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--worlds", type=int, default=300)
    parser.add_argument("--sweep-worlds", type=int, default=150)
    parser.add_argument("--papers", type=int, default=250)
    parser.add_argument("--periods", type=int, default=25)
    parser.add_argument("--exploration-step", type=float, default=0.10)
    parser.add_argument("--calibration", type=Path, default=calibrated.DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=Path("gate_attention_factorial_output"))
    args = parser.parse_args()

    calibrated._CALIBRATED_THETA = calibrated.load_theta(args.calibration)

    world_results, arm_summary, effects = run_factorial(
        args.seed, args.worlds, args.papers, args.periods
    )
    sweep_raw, sweep_summary = run_exploration_sweep(
        args.seed, args.sweep_worlds, args.papers, args.periods, args.exploration_step
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    world_results.to_csv(args.output_dir / "factorial_world_results.csv", index=False)
    arm_summary.to_csv(args.output_dir / "arm_summary.csv")
    effects.to_csv(args.output_dir / "factorial_effects.csv", index=False)
    sweep_raw.to_csv(args.output_dir / "exploration_sweep_world_results.csv", index=False)
    sweep_summary.to_csv(args.output_dir / "exploration_sweep.csv", index=False)

    metadata = {
        "seed": args.seed,
        "factorial_worlds": args.worlds,
        "sweep_worlds": args.sweep_worlds,
        "papers_per_world": args.papers,
        "periods": args.periods,
        "exploration_step": args.exploration_step,
        "prediction_file": "PREDICTION_ATTENTION_FACTORIAL.md",
        "gate_definition": "Same calibrated formal review and revision in every arm; gate arms down-weight unaccepted work by rejected_attention_fraction and incur publication delay, while no-gate arms keep review non-vetoing.",
        "attention_factor": "Peer-style concentrated allocation versus the repository's triage allocation rule. The secondary sweep isolates the randomized-exploration share within triage.",
        "interpretive_boundary": "Mechanism-isolation experiment in the current calibrated architecture, not an empirical estimate of real-world gate or triage effects.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    means = arm_summary.xs("mean", axis=1, level=1)
    print("Gate x attention factorial: mean arm outcomes")
    print(means[[
        "true_value_recovered", "score", "false_recognition_rate",
        "calibration_mse", "initial_rejection_share", "final_rejection_share",
        "mean_publication_delay", "initial_reject_resubmit_fraction",
        "mean_delay_initial_rejected",
    ]].round(5).to_string())
    print("\nPaired factorial effects")
    print(effects.round(5).to_string(index=False))
    print("\nExploration-share sweep")
    print(sweep_summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
