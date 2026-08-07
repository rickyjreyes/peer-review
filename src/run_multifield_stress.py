#!/usr/bin/env python3
"""Preregistered multifield stochastic stress test.

Extends the calibrated publication ecosystem with explicit heterogeneous fields,
persistent field-level evidence shocks, and within/cross-field belief spillover.
The extension is institution-neutral: publication systems retain their existing
review and attention rules while the epistemic environment is shared.

Prediction was committed first in PREDICTION_MULTIFIELD_STRESS.md.
"""

from __future__ import annotations

import argparse
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

DEFAULT_CALIBRATION = Path("results/empirical_calibration/best_fit_parameters.json")
DEFAULT_OUTPUT = Path("results/multifield_stress")
SYSTEMS = ecosystem.SYSTEMS

REGIMES = {
    "independent": {
        "field_heterogeneity": False,
        "shock_scale": 0.0,
        "within_spillover": 0.0,
        "cross_spillover": 0.0,
    },
    "multifield_moderate": {
        "field_heterogeneity": True,
        "shock_scale": 0.25,
        "within_spillover": 0.035,
        "cross_spillover": 0.010,
    },
    "multifield_strong": {
        "field_heterogeneity": True,
        "shock_scale": 0.50,
        "within_spillover": 0.080,
        "cross_spillover": 0.025,
    },
}


def seeded_rng(seed: int, world: int, stream: int, system_index: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, world, stream, system_index]))


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, math.nan, math.nan
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def make_field_environment(
    rng: np.random.Generator,
    n: int,
    fields: int,
    periods: int,
) -> dict:
    field_id = rng.integers(0, fields, size=n)

    # Stable field heterogeneity shared by both multifield regimes.
    noise_mult = np.clip(rng.lognormal(0.0, 0.25, fields), 0.60, 1.70)
    learning_mult = np.clip(rng.lognormal(0.0, 0.15, fields), 0.70, 1.40)
    replication_mult = np.clip(rng.lognormal(0.0, 0.25, fields), 0.50, 1.80)

    # Sparse symmetric field graph with a ring backbone so no field is isolated.
    adjacency = np.zeros((fields, fields), dtype=float)
    for f in range(fields):
        g = (f + 1) % fields
        weight = rng.uniform(0.4, 1.0)
        adjacency[f, g] = weight
        adjacency[g, f] = weight
    for f in range(fields):
        for g in range(f + 2, fields):
            if f == 0 and g == fields - 1:
                continue
            if rng.random() < 0.30:
                weight = rng.uniform(0.1, 0.7)
                adjacency[f, g] = max(adjacency[f, g], weight)
                adjacency[g, f] = adjacency[f, g]

    row_sum = adjacency.sum(axis=1, keepdims=True)
    network = np.divide(
        adjacency,
        row_sum,
        out=np.zeros_like(adjacency),
        where=row_sum > 0,
    )

    # Persistent stochastic shifts in field interpretation. Innovations are
    # common across publication systems. Neighboring fields weakly transmit
    # the prior period's shock.
    innovations = rng.normal(0.0, 1.0, size=(periods, fields))
    field_shock = np.zeros((periods, fields), dtype=float)
    for t in range(periods):
        previous = field_shock[t - 1] if t > 0 else np.zeros(fields)
        field_shock[t] = (
            0.65 * previous
            + 0.10 * (network @ previous)
            + math.sqrt(1.0 - 0.65**2) * innovations[t]
        )

    # Paper-level evidence randomness is shared across systems/regimes.
    return {
        "field_id": field_id,
        "noise_mult": noise_mult,
        "learning_mult": learning_mult,
        "replication_mult": replication_mult,
        "network": network,
        "field_shock": field_shock,
        "paper_noise": rng.normal(0.0, 1.0, size=(periods, n)),
        "replication_u": rng.random((periods, n)),
        "fraud_u": rng.random((periods, n)),
    }


def update_multifield_confidence(
    papers: dict,
    p: dict,
    confidence: np.ndarray,
    evidence_count: np.ndarray,
    active: np.ndarray,
    env: dict,
    period: int,
    regime: dict,
) -> np.ndarray:
    updated = confidence.copy()
    seen = evidence_count > 0
    field_id = env["field_id"]
    fields = env["network"].shape[0]

    if regime["field_heterogeneity"]:
        noise_mult = env["noise_mult"][field_id]
        learning_mult = env["learning_mult"][field_id]
        replication_mult = env["replication_mult"][field_id]
    else:
        noise_mult = np.ones_like(confidence)
        learning_mult = np.ones_like(confidence)
        replication_mult = np.ones_like(confidence)

    paper_signal = np.zeros_like(confidence, dtype=float)

    if np.any(seen):
        count = evidence_count[seen].astype(float)
        truth_signal = (papers["truth"][seen] - 0.5) * 2.0
        signal = (
            truth_signal
            + env["paper_noise"][period, seen]
            * p["evidence_noise"]
            * noise_mult[seen]
            / np.sqrt(count)
        )

        replication_probability = 1.0 - np.exp(
            -p["replication_bonus"] * replication_mult[seen] * count
        )
        replicated = env["replication_u"][period, seen] < replication_probability
        signal += replicated * truth_signal * 0.40

        fraud_seen = papers["fraud"][seen]
        detection_probability = 1.0 - np.exp(-p["fraud_detection_rate"] * count)
        fraud_detected = fraud_seen & (
            env["fraud_u"][period, seen] < detection_probability
        )
        signal += fraud_seen * 0.45
        signal -= fraud_detected * 1.75

        if regime["shock_scale"] > 0:
            signal += regime["shock_scale"] * env["field_shock"][period, field_id[seen]]

        paper_signal[seen] = signal
        logits = ecosystem.logit(updated[seen])
        logits += p["learning_rate"] * learning_mult[seen] * signal
        updated[seen] = ecosystem.sigmoid(logits)

    # Canonical behavior for unseen claims.
    unseen = ~seen
    updated[unseen] = 0.998 * updated[unseen] + 0.002 * 0.5

    # Spillover uses evidence generated by the current institution. Hidden truth
    # is never injected directly into the shared field signal.
    if regime["within_spillover"] > 0 or regime["cross_spillover"] > 0:
        field_signal = np.zeros(fields, dtype=float)
        for f in range(fields):
            mask = seen & (field_id == f)
            if np.any(mask):
                weights = evidence_count[mask].astype(float)
                field_signal[f] = np.average(paper_signal[mask], weights=weights)

        cross_signal = env["network"] @ field_signal
        shared = (
            regime["within_spillover"] * field_signal
            + regime["cross_spillover"] * cross_signal
        )
        shared = np.clip(shared, -0.35, 0.35)
        if np.any(active):
            logits = ecosystem.logit(updated[active])
            logits += p["learning_rate"] * shared[field_id[active]]
            updated[active] = ecosystem.sigmoid(logits)

    return updated


def initial_reviews(seed: int, world: int, papers: dict, p: dict) -> dict:
    reviews = {}
    for system_index, system in enumerate(SYSTEMS):
        rng = seeded_rng(seed, world, 20, system_index)
        accepted, score = calibrated.calibrated_formal_review(
            rng, papers, p, papers["clarity"].copy()
        )
        if system in ("open", "open_triage"):
            accepted = np.ones(len(papers["truth"]), dtype=bool)
        reviews[system] = (accepted.copy(), score.copy())
    return reviews


def simulate_regime(
    seed: int,
    world: int,
    regime_name: str,
    papers: dict,
    p: dict,
    env: dict,
    reviews: dict,
    periods: int,
) -> list[dict]:
    n = len(papers["truth"])
    regime = REGIMES[regime_name]
    budget = max(1, int(round(p["evals_per_paper_per_period"] * n)))
    outcomes = []
    total_true_value = papers["scientific_value"].sum() + 1e-12

    for system_index, system in enumerate(SYSTEMS):
        accepted0, review_score0 = reviews[system]
        s = {
            "confidence": np.full(n, 0.5),
            "clarity": papers["clarity"].copy(),
            "accepted": accepted0.copy(),
            "review_score": review_score0.copy(),
            "active": np.ones(n, dtype=bool),
            "first_recognition_period": np.full(n, np.nan),
            "cumulative_evaluations": np.zeros(n, dtype=int),
            "review_labor": n * p["reviewers_per_paper"] if system in ("peer_review", "hybrid") else 0,
            "publication_delay": np.ones(n) if system == "peer_review" else np.zeros(n),
        }

        # Same random-number streams across regimes for each system/world.
        institutional_rng = seeded_rng(seed, world, 30, system_index)
        attention_rng = seeded_rng(seed, world, 40, system_index)

        for t in range(periods):
            if system == "peer_review" and t > 0:
                rejected = ~s["accepted"]
                attempts = rejected & (
                    institutional_rng.random(n) < p["resubmission_probability"] / periods
                )
                if np.any(attempts):
                    s["clarity"][attempts] = np.clip(
                        s["clarity"][attempts]
                        + p["revision_effectiveness"] * institutional_rng.beta(2, 5, attempts.sum()),
                        0,
                        1,
                    )
                    accepted_new, score_new = calibrated.calibrated_formal_review(
                        institutional_rng, papers, p, s["clarity"]
                    )
                    s["accepted"][attempts] = accepted_new[attempts]
                    s["review_score"][attempts] = score_new[attempts]
                    s["review_labor"] += attempts.sum() * p["reviewers_per_paper"]
                    s["publication_delay"][attempts] += 1

            counts = ecosystem.allocate_attention(
                rng=attention_rng,
                system=system,
                papers=papers,
                p=p,
                confidence=s["confidence"],
                available=s["active"],
                formally_accepted=s["accepted"],
                total_budget=budget,
            )
            s["cumulative_evaluations"] += counts
            s["confidence"] = update_multifield_confidence(
                papers, p, s["confidence"], counts, s["active"], env, t, regime
            )

            recognized_now = (s["confidence"] >= p["recognition_threshold"]) & s["active"]
            newly = recognized_now & np.isnan(s["first_recognition_period"])
            s["first_recognition_period"][newly] = t

            if t >= p["min_withdraw_period"]:
                enough_evidence = s["cumulative_evaluations"] >= 2
                withdraw = (s["confidence"] < p["withdraw_threshold"]) & enough_evidence
                s["active"][withdraw] = False

        confidence = s["confidence"]
        recognized = (confidence >= p["recognition_threshold"]) & s["active"]
        false_mask = papers["truth"] == 0.0
        mixed_mask = papers["truth"] == 0.5
        true_mask = papers["truth"] == 1.0

        weighted_calibration = np.average(
            (confidence - papers["truth"]) ** 2,
            weights=papers["intrinsic_value"],
        )
        true_value_recovered = papers["scientific_value"][recognized].sum() / total_true_value
        false_recognition = (recognized & false_mask).sum() / max(false_mask.sum(), 1)
        mixed_recognition = (recognized & mixed_mask).sum() / max(mixed_mask.sum(), 1)
        true_recognition = (recognized & true_mask).sum() / max(true_mask.sum(), 1)

        recognized_true = recognized & (papers["truth"] > 0)
        mean_time_to_recognition = math.nan
        if np.any(recognized_true):
            mean_time_to_recognition = float(
                np.nanmean(s["first_recognition_period"][recognized_true])
            )

        attention_coverage = float((s["cumulative_evaluations"] > 0).mean())
        attention_gini = ecosystem.gini(s["cumulative_evaluations"].astype(float))
        delay_cost = p["delay_weight"] * np.mean(s["publication_delay"]) / max(periods, 1)
        labor_cost = p["labor_weight"] * s["review_labor"] / max(n * periods, 1)
        score = (
            p["recovery_weight"] * true_value_recovered
            - p["false_weight"] * false_recognition
            - p["calibration_weight"] * weighted_calibration
            - delay_cost
            - labor_cost
        )

        outcomes.append({
            "world": world,
            "regime": regime_name,
            "system": system,
            "score": score,
            "true_value_recovered": true_value_recovered,
            "true_recognition_rate": true_recognition,
            "mixed_recognition_rate": mixed_recognition,
            "false_recognition_rate": false_recognition,
            "calibration_mse": weighted_calibration,
            "attention_coverage": attention_coverage,
            "attention_gini": attention_gini,
            "mean_time_to_recognition": mean_time_to_recognition,
            "review_labor_per_paper": s["review_labor"] / n,
            "mean_publication_delay": float(np.mean(s["publication_delay"])),
            "rejected_final_share": float((~s["accepted"]).mean()) if system == "peer_review" else 0.0,
        })

    return outcomes


def run(seed: int, worlds: int, papers_per_world: int, periods: int, fields: int):
    rows = []
    for world in range(worlds):
        world_rng = seeded_rng(seed, world, 1)
        p = calibrated.sample_calibration_propagated_parameters(world_rng)
        papers = calibrated.calibrated_initialize_papers(world_rng, p, papers_per_world)
        env = make_field_environment(
            seeded_rng(seed, world, 2), papers_per_world, fields, periods
        )
        reviews = initial_reviews(seed, world, papers, p)
        for regime_name in REGIMES:
            rows.extend(simulate_regime(
                seed, world, regime_name, papers, p, env, reviews, periods
            ))

    results = pd.DataFrame(rows)
    metric_columns = [
        "score", "true_value_recovered", "true_recognition_rate",
        "mixed_recognition_rate", "false_recognition_rate", "calibration_mse",
        "attention_coverage", "attention_gini", "mean_time_to_recognition",
        "review_labor_per_paper", "mean_publication_delay", "rejected_final_share",
    ]
    arm_means = results.groupby(["regime", "system"])[metric_columns].mean().reset_index()

    gap_rows = []
    for regime_name in REGIMES:
        pivot = results[results["regime"] == regime_name].pivot(
            index="world", columns="system", values="true_value_recovered"
        )
        gap = (pivot["peer_review"] - pivot["open_triage"]).to_numpy()
        mean, low, high = mean_ci(gap)
        gap_rows.append({"regime": regime_name, "peer_minus_triage": mean, "ci_low": low, "ci_high": high})
    gaps = pd.DataFrame(gap_rows)

    gap_by_world = results.pivot_table(
        index=["world", "regime"], columns="system", values="true_value_recovered"
    ).reset_index()
    gap_by_world["gap"] = gap_by_world["peer_review"] - gap_by_world["open_triage"]
    wide_gap = gap_by_world.pivot(index="world", columns="regime", values="gap")
    change_rows = []
    for regime_name in ("multifield_moderate", "multifield_strong"):
        change = (wide_gap[regime_name] - wide_gap["independent"]).to_numpy()
        mean, low, high = mean_ci(change)
        change_rows.append({
            "comparison": f"{regime_name}_minus_independent",
            "gap_change": mean,
            "ci_low": low,
            "ci_high": high,
        })
    gap_changes = pd.DataFrame(change_rows)

    win_rows = []
    for regime_name in REGIMES:
        subset = results[results["regime"] == regime_name]
        true_table = subset.pivot(index="world", columns="system", values="true_value_recovered")
        score_table = subset.pivot(index="world", columns="system", values="score")
        true_winner = true_table.idxmax(axis=1)
        score_winner = score_table.idxmax(axis=1)
        for system in SYSTEMS:
            win_rows.append({
                "regime": regime_name,
                "system": system,
                "true_value_win_share": float((true_winner == system).mean()),
                "score_win_share": float((score_winner == system).mean()),
            })
    win_shares = pd.DataFrame(win_rows)
    return results, arm_means, gaps, gap_changes, win_shares


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--worlds", type=int, default=300)
    parser.add_argument("--papers", type=int, default=250)
    parser.add_argument("--periods", type=int, default=25)
    parser.add_argument("--fields", type=int, default=6)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.worlds < 1 or args.papers < 20 or args.periods < 1 or args.fields < 2:
        parser.error("worlds >= 1, papers >= 20, periods >= 1, fields >= 2")

    calibrated._CALIBRATED_THETA = calibrated.load_theta(args.calibration)
    _, arm_means, gaps, gap_changes, win_shares = run(
        args.seed, args.worlds, args.papers, args.periods, args.fields
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    arm_means.to_csv(out / "arm_means.csv", index=False)
    gaps.to_csv(out / "peer_triage_gaps.csv", index=False)
    gap_changes.to_csv(out / "gap_changes.csv", index=False)
    win_shares.to_csv(out / "win_shares.csv", index=False)

    metadata = {
        "seed": args.seed,
        "worlds": args.worlds,
        "papers_per_world": args.papers,
        "periods": args.periods,
        "fields": args.fields,
        "prediction_file": "PREDICTION_MULTIFIELD_STRESS.md",
        "regimes": REGIMES,
        "reviewer_calibration": str(args.calibration),
        "interpretive_boundary": (
            "Structural robustness test with explicit heterogeneous fields and "
            "correlated path-dependent learning, not an empirical reconstruction "
            "of named disciplines."
        ),
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nArm means")
    print(arm_means.to_string(index=False))
    print("\nPeer review minus open-triage true-value gaps")
    print(gaps.to_string(index=False))
    print("\nChange in peer-review deficit versus independent regime")
    print(gap_changes.to_string(index=False))
    print("\nWin shares")
    print(win_shares.to_string(index=False))


if __name__ == "__main__":
    main()
