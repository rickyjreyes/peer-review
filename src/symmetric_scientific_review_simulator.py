#!/usr/bin/env python3
"""
Symmetric Scientific Review Simulator
=====================================

Purpose
-------
Compare four systems using the same hidden paper population and the same total
expert-labor budget:

1. peer_review
2. open_review
3. open_triage
4. hybrid

Both prepublication and post-publication review can:
- detect defects,
- improve methods,
- improve statistics,
- improve reporting,
- improve reproducibility,
- miss defects,
- introduce harmful revisions,
- consume expert labor.

The model is intentionally symmetric. Differences arise from timing, attention
allocation, gating, resubmission, and institutional signals rather than from
giving one system exclusive access to corrective mechanisms.

This is a structural simulation, not an empirical estimate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


SYSTEMS = ("peer_review", "open_review", "open_triage", "hybrid")


@dataclass(frozen=True)
class Config:
    seed: int = 20260806
    worlds: int = 2000
    papers: int = 300
    periods: int = 30
    output_dir: str = "symmetric_simulation_output"


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def gini(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(values == 0):
        return 0.0
    values = np.sort(values)
    n = values.size
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * values) / (n * np.sum(values)))


def softmax_probs(values, eligible, concentration):
    probs = np.zeros_like(values, dtype=float)
    if not np.any(eligible):
        return probs
    x = values[eligible]
    x = x - np.max(x)
    w = np.exp(np.clip(concentration * x, -50, 50))
    probs[eligible] = w / w.sum()
    return probs


def sample_parameters(rng):
    false_share = rng.uniform(0.20, 0.45)
    mixed_share = min(rng.uniform(0.15, 0.30), 0.82 - false_share)

    return {
        "false_share": false_share,
        "mixed_share": mixed_share,
        "fraud_share": rng.uniform(0.00, 0.05),
        "novelty_value_multiplier": rng.uniform(1.0, 6.0),

        # Initial manuscript quality
        "method_quality_mean": rng.uniform(0.45, 0.70),
        "statistics_quality_mean": rng.uniform(0.45, 0.70),
        "reporting_quality_mean": rng.uniform(0.45, 0.75),
        "reproducibility_mean": rng.uniform(0.40, 0.70),

        # Shared reviewer/evaluator properties
        "expert_skill": rng.uniform(0.35, 1.10),
        "expert_noise": rng.uniform(0.80, 1.60),
        "defect_detection_rate": rng.uniform(0.15, 0.55),
        "repair_success_rate": rng.uniform(0.20, 0.65),
        "harmful_revision_rate": rng.uniform(0.00, 0.15),
        "repair_magnitude": rng.uniform(0.05, 0.25),
        "harm_magnitude": rng.uniform(0.02, 0.15),

        # Biases affecting reviewer decisions/attention
        "prestige_bias": rng.uniform(0.00, 0.60),
        "novelty_penalty": rng.uniform(0.00, 0.50),
        "clarity_bias": rng.uniform(0.10, 0.60),
        "conformity_bias": rng.uniform(0.00, 0.40),

        # Gate and resubmission
        "accept_threshold": rng.uniform(-0.10, 0.40),
        "reviewers_per_paper": int(rng.integers(2, 4)),
        "rejected_attention_fraction": rng.uniform(0.10, 0.60),
        "resubmission_probability": rng.uniform(0.30, 0.80),

        # Equal labor budget per period
        "expert_actions_per_paper": rng.uniform(0.40, 1.20),
        "peer_concentration": rng.uniform(1.0, 4.0),
        "open_concentration": rng.uniform(1.0, 4.0),
        "triage_concentration": rng.uniform(0.8, 3.5),
        "hybrid_concentration": rng.uniform(0.8, 3.5),
        "exploration_share": rng.uniform(0.15, 0.50),
        "popularity_bias": rng.uniform(0.10, 0.80),

        # Evidence accumulation
        "evidence_noise": rng.uniform(0.80, 1.60),
        "learning_rate": rng.uniform(0.35, 0.95),
        "replication_probability": rng.uniform(0.05, 0.35),
        "fraud_detection_rate": rng.uniform(0.05, 0.30),
        "recognition_threshold": rng.uniform(0.68, 0.82),
        "withdraw_threshold": rng.uniform(0.05, 0.18),
        "min_withdraw_period": int(rng.integers(4, 9)),

        # Utility weights
        "recovery_weight": rng.uniform(0.9, 1.3),
        "false_weight": rng.uniform(0.9, 1.8),
        "calibration_weight": rng.uniform(0.5, 1.2),
        "delay_weight": rng.uniform(0.0, 0.05),
        "labor_weight": rng.uniform(0.0, 0.03),
    }


def initialize_papers(rng, p, n):
    truth = rng.choice(
        np.array([0.0, 0.5, 1.0]),
        size=n,
        p=[p["false_share"], p["mixed_share"], 1 - p["false_share"] - p["mixed_share"]],
    )

    novelty = rng.beta(2, 5, n)
    prestige = rng.beta(2, 5, n)
    clarity = rng.beta(4, 2, n)
    popularity = rng.beta(2, 2, n)
    fraud = (rng.random(n) < p["fraud_share"]) & (truth < 1.0)

    def beta_from_mean(mean):
        concentration = 8.0
        a = mean * concentration
        b = (1 - mean) * concentration
        return rng.beta(a, b, n)

    methods = beta_from_mean(p["method_quality_mean"])
    statistics = beta_from_mean(p["statistics_quality_mean"])
    reporting = beta_from_mean(p["reporting_quality_mean"])
    reproducibility = beta_from_mean(p["reproducibility_mean"])

    intrinsic_value = rng.lognormal(0, 1, n) * (1 + p["novelty_value_multiplier"] * novelty**2)
    scientific_value = intrinsic_value * truth

    return {
        "truth": truth,
        "novelty": novelty,
        "prestige": prestige,
        "clarity": clarity,
        "popularity": popularity,
        "fraud": fraud,
        "methods": methods,
        "statistics": statistics,
        "reporting": reporting,
        "reproducibility": reproducibility,
        "intrinsic_value": intrinsic_value,
        "scientific_value": scientific_value,
    }


def manuscript_quality(papers, state):
    return (
        0.30 * state["methods"]
        + 0.25 * state["statistics"]
        + 0.20 * state["reporting"]
        + 0.25 * state["reproducibility"]
    )


def apply_expert_actions(rng, papers, state, p, action_counts):
    """
    Shared correction mechanism for every system.
    Each expert action may detect a defect, repair it, miss it, or introduce harm.
    """
    total_actions = int(action_counts.sum())
    state["labor"] += total_actions
    if total_actions == 0:
        return

    for field in ("methods", "statistics", "reporting", "reproducibility"):
        quality = state[field]
        defect = 1.0 - quality

        seen = action_counts > 0
        if not np.any(seen):
            continue

        detection_prob = 1.0 - np.exp(
            -p["defect_detection_rate"] * action_counts[seen] * defect[seen]
        )
        detected = rng.random(seen.sum()) < detection_prob

        repair = detected & (rng.random(seen.sum()) < p["repair_success_rate"])
        harmful = rng.random(seen.sum()) < (
            1.0 - np.exp(-p["harmful_revision_rate"] * action_counts[seen])
        )

        idx = np.where(seen)[0]
        repair_idx = idx[repair]
        harm_idx = idx[harmful]

        if repair_idx.size:
            quality[repair_idx] = np.clip(
                quality[repair_idx]
                + p["repair_magnitude"] * (1.0 - quality[repair_idx]),
                0,
                1,
            )
        if harm_idx.size:
            quality[harm_idx] = np.clip(
                quality[harm_idx] - p["harm_magnitude"] * quality[harm_idx],
                0,
                1,
            )

        state[field] = quality


def review_scores(rng, papers, state, p):
    quality = manuscript_quality(papers, state)
    scores = []
    for _ in range(p["reviewers_per_paper"]):
        score = (
            p["expert_skill"] * ((papers["truth"] - 0.5) * 2)
            + 0.8 * (quality - 0.5)
            + p["clarity_bias"] * (papers["clarity"] - 0.5)
            + p["prestige_bias"] * (papers["prestige"] - 0.5)
            - p["novelty_penalty"] * papers["novelty"]
            + p["conformity_bias"] * (papers["prestige"] - papers["novelty"])
            + rng.normal(0, p["expert_noise"], len(quality))
        )
        scores.append(score)
    return np.mean(scores, axis=0)


def allocate_actions(rng, system, papers, state, p, budget):
    available = state["active"]
    conf = state["confidence"]
    quality = manuscript_quality(papers, state)

    if system == "peer_review":
        appeal = (
            1.0 * conf
            + 0.5 * state["accepted"].astype(float)
            + p["prestige_bias"] * papers["prestige"]
            + 0.2 * quality
        )
        probs = softmax_probs(appeal, available, p["peer_concentration"])
        rejected = available & (~state["accepted"])
        probs[rejected] *= p["rejected_attention_fraction"]
        if probs.sum() > 0:
            probs /= probs.sum()

    elif system == "open_review":
        appeal = (
            0.8 * conf
            + p["popularity_bias"] * papers["popularity"]
            + p["prestige_bias"] * papers["prestige"]
            + 0.2 * quality
            + 0.1 * papers["novelty"]
        )
        probs = softmax_probs(appeal, available, p["open_concentration"])

    elif system == "open_triage":
        merit = (
            0.35 * quality
            + 0.25 * conf
            + 0.20 * papers["novelty"]
            + 0.10 * (1 - papers["prestige"])
            + 0.10 * papers["clarity"]
        )
        probs = softmax_probs(merit, available, p["triage_concentration"])
        explore = np.zeros_like(probs)
        explore[available] = 1.0 / available.sum()
        probs = (1 - p["exploration_share"]) * probs + p["exploration_share"] * explore

    elif system == "hybrid":
        appeal = (
            0.75 * conf
            + 0.25 * quality
            + 0.20 * state["accepted"].astype(float)
            + 0.15 * papers["novelty"]
            + 0.10 * (1 - papers["prestige"])
        )
        probs = softmax_probs(appeal, available, p["hybrid_concentration"])
        explore = np.zeros_like(probs)
        explore[available] = 1.0 / available.sum()
        probs = (1 - 0.5 * p["exploration_share"]) * probs + 0.5 * p["exploration_share"] * explore

    else:
        raise ValueError(system)

    if probs.sum() == 0:
        return np.zeros(len(conf), dtype=int)
    probs /= probs.sum()
    return rng.multinomial(budget, probs)


def update_confidence(rng, papers, state, p, action_counts):
    seen = action_counts > 0
    if not np.any(seen):
        return

    quality = manuscript_quality(papers, state)
    truth_signal = ((papers["truth"] - 0.5) * 2)[seen]
    quality_signal = (quality[seen] - 0.5) * 0.8

    noise = rng.normal(0, p["evidence_noise"] / np.sqrt(action_counts[seen]))
    evidence = truth_signal + quality_signal + noise

    replicated = rng.random(seen.sum()) < (
        1 - np.exp(-p["replication_probability"] * action_counts[seen])
    )
    evidence += replicated * truth_signal * 0.5

    fraud_seen = papers["fraud"][seen]
    fraud_detected = fraud_seen & (
        rng.random(seen.sum())
        < (1 - np.exp(-p["fraud_detection_rate"] * action_counts[seen]))
    )
    evidence += fraud_seen * 0.35
    evidence -= fraud_detected * 1.5

    old = state["confidence"][seen]
    state["confidence"][seen] = sigmoid(logit(old) + p["learning_rate"] * evidence)

    unseen = ~seen
    state["confidence"][unseen] = 0.998 * state["confidence"][unseen] + 0.002 * 0.5


def simulate_world(rng, p, n, periods):
    papers = initialize_papers(rng, p, n)

    states = {}
    for system in SYSTEMS:
        state = {
            "methods": papers["methods"].copy(),
            "statistics": papers["statistics"].copy(),
            "reporting": papers["reporting"].copy(),
            "reproducibility": papers["reproducibility"].copy(),
            "confidence": np.full(n, 0.5),
            "active": np.ones(n, dtype=bool),
            "accepted": np.ones(n, dtype=bool),
            "labor": 0,
            "delay": np.zeros(n),
            "cumulative_actions": np.zeros(n, dtype=int),
            "first_recognition": np.full(n, np.nan),
        }

        if system in ("peer_review", "hybrid"):
            # Same formal-review labor and same correction mechanism.
            initial_actions = np.full(n, p["reviewers_per_paper"], dtype=int)
            apply_expert_actions(rng, papers, state, p, initial_actions)
            scores = review_scores(rng, papers, state, p)
            state["accepted"] = scores > p["accept_threshold"]
            if system == "peer_review":
                state["delay"] += 1

        states[system] = state

    budget = max(1, int(round(p["expert_actions_per_paper"] * n)))

    for t in range(periods):
        for system in SYSTEMS:
            state = states[system]

            # Resubmission for gated peer review.
            if system == "peer_review" and t > 0:
                rejected = ~state["accepted"]
                attempts = rejected & (
                    rng.random(n) < p["resubmission_probability"] / periods
                )
                if np.any(attempts):
                    action_counts = np.zeros(n, dtype=int)
                    action_counts[attempts] = p["reviewers_per_paper"]
                    apply_expert_actions(rng, papers, state, p, action_counts)
                    scores = review_scores(rng, papers, state, p)
                    state["accepted"][attempts] = scores[attempts] > p["accept_threshold"]
                    state["delay"][attempts] += 1

            actions = allocate_actions(rng, system, papers, state, p, budget)
            state["cumulative_actions"] += actions

            apply_expert_actions(rng, papers, state, p, actions)
            update_confidence(rng, papers, state, p, actions)

            recognized = (
                state["confidence"] >= p["recognition_threshold"]
            ) & state["active"]
            new = recognized & np.isnan(state["first_recognition"])
            state["first_recognition"][new] = t

            if t >= p["min_withdraw_period"]:
                enough = state["cumulative_actions"] >= 2
                withdraw = (
                    state["confidence"] < p["withdraw_threshold"]
                ) & enough
                state["active"][withdraw] = False

    outcomes = []
    total_true_value = papers["scientific_value"].sum() + 1e-12

    for system in SYSTEMS:
        state = states[system]
        conf = state["confidence"]
        recognized = (conf >= p["recognition_threshold"]) & state["active"]
        quality = manuscript_quality(papers, state)

        true_value_recovered = papers["scientific_value"][recognized].sum() / total_true_value
        false_mask = papers["truth"] == 0.0
        true_mask = papers["truth"] == 1.0
        mixed_mask = papers["truth"] == 0.5

        false_recognition = (recognized & false_mask).sum() / max(false_mask.sum(), 1)
        true_recognition = (recognized & true_mask).sum() / max(true_mask.sum(), 1)
        mixed_recognition = (recognized & mixed_mask).sum() / max(mixed_mask.sum(), 1)

        calibration = np.average(
            (conf - papers["truth"]) ** 2,
            weights=papers["intrinsic_value"],
        )

        recognized_true = recognized & (papers["truth"] > 0)
        mean_time = np.nan
        if np.any(recognized_true):
            mean_time = np.nanmean(state["first_recognition"][recognized_true])

        score = (
            p["recovery_weight"] * true_value_recovered
            - p["false_weight"] * false_recognition
            - p["calibration_weight"] * calibration
            - p["delay_weight"] * np.mean(state["delay"]) / max(periods, 1)
            - p["labor_weight"] * state["labor"] / max(n * periods, 1)
        )

        outcomes.append({
            "system": system,
            "score": score,
            "true_value_recovered": true_value_recovered,
            "true_recognition_rate": true_recognition,
            "mixed_recognition_rate": mixed_recognition,
            "false_recognition_rate": false_recognition,
            "calibration_mse": calibration,
            "mean_final_quality": float(np.mean(quality)),
            "attention_coverage": float((state["cumulative_actions"] > 0).mean()),
            "attention_gini": gini(state["cumulative_actions"]),
            "mean_time_to_recognition": mean_time,
            "labor_per_paper": state["labor"] / n,
            "mean_delay": float(np.mean(state["delay"])),
            "final_rejection_share": float((~state["accepted"]).mean()) if system == "peer_review" else 0.0,
        })

    return outcomes


def ci(p, n):
    se = math.sqrt(max(p * (1 - p), 0) / n)
    return max(0, p - 1.96 * se), min(1, p + 1.96 * se)


def run(config):
    rng = np.random.default_rng(config.seed)
    rows = []

    for world in range(config.worlds):
        p = sample_parameters(rng)
        for out in simulate_world(rng, p, config.papers, config.periods):
            rows.append({"world": world, **out, **p})

    results = pd.DataFrame(rows)
    scores = results.pivot(index="world", columns="system", values="score")
    winners = scores.idxmax(axis=1)

    win_rows = []
    for system in SYSTEMS:
        share = float((winners == system).mean())
        lo, hi = ci(share, config.worlds)
        win_rows.append({"system": system, "win_share": share, "ci_low": lo, "ci_high": hi})
    win_rates = pd.DataFrame(win_rows).set_index("system")

    metrics = [
        "score", "true_value_recovered", "true_recognition_rate",
        "mixed_recognition_rate", "false_recognition_rate",
        "calibration_mse", "mean_final_quality", "attention_coverage",
        "attention_gini", "mean_time_to_recognition", "labor_per_paper",
        "mean_delay", "final_rejection_share",
    ]
    summary = results.groupby("system")[metrics].mean()

    return results, summary, win_rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--worlds", type=int, default=Config.worlds)
    parser.add_argument("--papers", type=int, default=Config.papers)
    parser.add_argument("--periods", type=int, default=Config.periods)
    parser.add_argument("--output-dir", default=Config.output_dir)
    args = parser.parse_args()

    config = Config(args.seed, args.worlds, args.papers, args.periods, args.output_dir)
    results, summary, win_rates = run(config)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "world_level_results.csv", index=False)
    summary.to_csv(out / "summary.csv")
    win_rates.to_csv(out / "win_rates.csv")
    with (out / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    print("Symmetric Scientific Review Simulator")
    print(f"Worlds: {config.worlds:,}")
    print(f"Papers/world: {config.papers:,}")
    print(f"Periods: {config.periods}")
    print(f"Unique papers: {config.worlds * config.papers:,}")
    print("\nWin rates:")
    print(win_rates.round(4))
    print("\nMean outcomes:")
    print(summary.round(4))


if __name__ == "__main__":
    main()
