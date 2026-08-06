#!/usr/bin/env python3
"""
Scientific Publication Ecosystem Simulator
==========================================

Compares four institutions under hidden truth, finite attention, noisy evidence,
reviewer incentives, partial truth, resubmission, prestige effects, and time:

1. peer_review:
   Prepublication review gates journal publication. Rejected work remains publicly
   accessible with reduced attention and may be revised/resubmitted.

2. open:
   Immediate publication. Attention follows prestige, popularity, novelty, clarity,
   and current confidence.

3. open_triage:
   Immediate publication with transparent triage, replication priority, and a
   randomized exploration allocation.

4. hybrid:
   Immediate public preprint plus optional formal review. Formal review affects
   labels and attention but never blocks public availability.

The simulator knows each paper's latent truth state. Agents do not. They only see
noisy signals and accumulate evidence over time.

This is a structural model, not an empirical estimate of real scientific systems.
Its purpose is to identify which assumptions cause each institution to win and to
produce falsifiable empirical calibration targets.

Example:
    python scientific_publication_simulator.py --worlds 5000 --papers 300 --periods 30
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


SYSTEMS = ("peer_review", "open", "open_triage", "hybrid")


@dataclass(frozen=True)
class Config:
    seed: int = 20260805
    worlds: int = 5000
    papers: int = 300
    periods: int = 30
    output_dir: str = "simulation_output"
    save_world_level: bool = True


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def normalized_softmax(values: np.ndarray, eligible: np.ndarray, concentration: float) -> np.ndarray:
    probs = np.zeros_like(values, dtype=float)
    if not np.any(eligible):
        return probs
    centered = values[eligible] - np.max(values[eligible])
    weights = np.exp(np.clip(concentration * centered, -50, 50))
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        probs[eligible] = 1.0 / eligible.sum()
    else:
        probs[eligible] = weights / total
    return probs


def sample_parameters(rng: np.random.Generator) -> dict:
    false_share = rng.uniform(0.20, 0.50)
    mixed_share = rng.uniform(0.10, 0.35)
    mixed_share = min(mixed_share, 0.82 - false_share)

    return {
        # Population
        "false_share": false_share,
        "mixed_share": mixed_share,
        "fraud_share": rng.uniform(0.00, 0.08),
        "novelty_value_multiplier": rng.uniform(1.0, 8.0),

        # Review quality and incentives
        "reviewer_skill": rng.uniform(0.20, 1.50),
        "review_noise": rng.uniform(0.60, 1.80),
        "prestige_bias": rng.uniform(0.00, 1.20),
        "clarity_bias": rng.uniform(0.10, 0.80),
        "novelty_penalty": rng.uniform(0.00, 1.20),
        "career_conformity": rng.uniform(0.00, 1.00),
        "accept_threshold": rng.uniform(-0.20, 0.60),
        "reviewers_per_paper": int(rng.integers(2, 4)),

        # Rejection is not disappearance
        "rejected_attention_fraction": rng.uniform(0.05, 0.60),
        "resubmission_probability": rng.uniform(0.20, 0.85),
        "revision_effectiveness": rng.uniform(0.05, 0.35),

        # Attention
        "evals_per_paper_per_period": rng.uniform(0.40, 1.60),
        "peer_attention_prestige": rng.uniform(0.40, 1.50),
        "open_attention_prestige": rng.uniform(0.20, 1.50),
        "popularity_bias": rng.uniform(0.00, 1.00),
        "novelty_attention": rng.uniform(-0.20, 0.80),
        "peer_concentration": rng.uniform(1.00, 5.00),
        "open_concentration": rng.uniform(0.50, 5.00),
        "triage_concentration": rng.uniform(0.50, 4.00),
        "hybrid_concentration": rng.uniform(0.50, 4.00),
        "exploration_share": rng.uniform(0.10, 0.60),

        # Evidence production
        "evidence_noise": rng.uniform(0.70, 2.00),
        "learning_rate": rng.uniform(0.35, 1.20),
        "replication_bonus": rng.uniform(0.10, 0.60),
        "fraud_detection_rate": rng.uniform(0.05, 0.45),
        "min_withdraw_period": int(rng.integers(3, 10)),
        "withdraw_threshold": rng.uniform(0.05, 0.20),
        "recognition_threshold": rng.uniform(0.65, 0.85),

        # Score weights
        "recovery_weight": rng.uniform(0.80, 1.50),
        "false_weight": rng.uniform(0.80, 2.00),
        "calibration_weight": rng.uniform(0.50, 1.50),
        "delay_weight": rng.uniform(0.00, 0.10),
        "labor_weight": rng.uniform(0.00, 0.05),
    }


def initialize_papers(rng: np.random.Generator, p: dict, n: int) -> dict:
    truth = rng.choice(
        np.array([0.0, 0.5, 1.0]),
        size=n,
        p=[
            p["false_share"],
            p["mixed_share"],
            1.0 - p["false_share"] - p["mixed_share"],
        ],
    )
    novelty = rng.beta(2, 5, n)
    prestige = rng.beta(2, 5, n)
    clarity = rng.beta(4, 2, n)
    popularity = rng.beta(2, 2, n)
    fraud = (rng.random(n) < p["fraud_share"]) & (truth < 1.0)

    intrinsic_value = rng.lognormal(0.0, 1.0, n) * (
        1.0 + p["novelty_value_multiplier"] * novelty**2
    )
    scientific_value = intrinsic_value * truth

    return {
        "truth": truth,
        "novelty": novelty,
        "prestige": prestige,
        "clarity": clarity,
        "popularity": popularity,
        "fraud": fraud,
        "intrinsic_value": intrinsic_value,
        "scientific_value": scientific_value,
    }


def formal_review(
    rng: np.random.Generator,
    papers: dict,
    p: dict,
    clarity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(papers["truth"])
    scores = []
    for _ in range(p["reviewers_per_paper"]):
        conformity_signal = (
            p["career_conformity"]
            * (papers["prestige"] - papers["novelty"])
        )
        score = (
            p["reviewer_skill"] * (papers["truth"] - 0.5) * 2.0
            + p["clarity_bias"] * (clarity - 0.5)
            + p["prestige_bias"] * (papers["prestige"] - 0.5)
            - p["novelty_penalty"] * papers["novelty"]
            + conformity_signal
            + rng.normal(0.0, p["review_noise"], n)
        )
        scores.append(score)
    mean_score = np.mean(scores, axis=0)
    accepted = mean_score > p["accept_threshold"]
    return accepted, mean_score


def allocate_attention(
    rng: np.random.Generator,
    system: str,
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

    if system == "peer_review":
        appeal = (
            1.10 * confidence
            + p["peer_attention_prestige"] * prestige
            + 0.20 * clarity
            + 0.15 * popularity
        )
        # Rejected work remains available, but receives less attention.
        weights = normalized_softmax(
            appeal, available, p["peer_concentration"]
        )
        weights[available & (~formally_accepted)] *= p["rejected_attention_fraction"]
        if weights.sum() > 0:
            weights /= weights.sum()

    elif system == "open":
        appeal = (
            0.80 * confidence
            + p["open_attention_prestige"] * prestige
            + p["popularity_bias"] * popularity
            + p["novelty_attention"] * novelty
            + 0.20 * clarity
        )
        weights = normalized_softmax(
            appeal, available, p["open_concentration"]
        )

    elif system == "open_triage":
        merit = (
            0.30 * clarity
            + 0.25 * novelty
            + 0.25 * confidence
            + 0.15 * (1.0 - prestige)
            + 0.05 * popularity
        )
        weights = normalized_softmax(
            merit, available, p["triage_concentration"]
        )
        exploration = np.zeros_like(weights)
        exploration[available] = 1.0 / available.sum()
        weights = (
            (1.0 - p["exploration_share"]) * weights
            + p["exploration_share"] * exploration
        )

    elif system == "hybrid":
        # Immediate preprint for all; formal acceptance is a signal, not a gate.
        appeal = (
            0.75 * confidence
            + 0.30 * clarity
            + 0.20 * novelty
            + 0.20 * formally_accepted.astype(float)
            + 0.15 * (1.0 - prestige)
        )
        weights = normalized_softmax(
            appeal, available, p["hybrid_concentration"]
        )
        exploration = np.zeros_like(weights)
        exploration[available] = 1.0 / available.sum()
        weights = (
            (1.0 - 0.5 * p["exploration_share"]) * weights
            + 0.5 * p["exploration_share"] * exploration
        )
    else:
        raise ValueError(f"Unknown system: {system}")

    if weights.sum() <= 0:
        return np.zeros_like(weights, dtype=int)
    weights /= weights.sum()
    return rng.multinomial(total_budget, weights)


def update_confidence(
    rng: np.random.Generator,
    papers: dict,
    p: dict,
    confidence: np.ndarray,
    evidence_count: np.ndarray,
) -> np.ndarray:
    updated = confidence.copy()
    seen = evidence_count > 0
    if not np.any(seen):
        return updated

    count = evidence_count[seen]
    truth_signal = (papers["truth"][seen] - 0.5) * 2.0

    # Multiple evaluations reduce random noise.
    signal = truth_signal + rng.normal(
        0.0,
        p["evidence_noise"] / np.sqrt(count),
        seen.sum(),
    )

    # More evaluations increase the chance of genuine replication.
    replicated = rng.random(seen.sum()) < (
        1.0 - np.exp(-p["replication_bonus"] * count)
    )
    signal += replicated * truth_signal * 0.40

    # Fraud can look convincing until detected.
    fraud_seen = papers["fraud"][seen]
    fraud_detected = fraud_seen & (
        rng.random(seen.sum())
        < (1.0 - np.exp(-p["fraud_detection_rate"] * count))
    )
    signal += fraud_seen * 0.45
    signal -= fraud_detected * 1.75

    updated_logits = logit(updated[seen]) + p["learning_rate"] * signal
    updated[seen] = sigmoid(updated_logits)

    # Unseen papers drift very slowly toward uncertainty, not rejection.
    unseen = ~seen
    updated[unseen] = 0.998 * updated[unseen] + 0.002 * 0.5
    return updated


def simulate_world(rng: np.random.Generator, p: dict, n: int, periods: int) -> list[dict]:
    papers = initialize_papers(rng, p, n)
    base_clarity = papers["clarity"].copy()

    state: Dict[str, dict] = {}
    for system in SYSTEMS:
        confidence = np.full(n, 0.5)
        clarity = base_clarity.copy()
        accepted, review_score = formal_review(rng, papers, p, clarity)

        if system in ("open", "open_triage"):
            accepted = np.ones(n, dtype=bool)

        state[system] = {
            "confidence": confidence,
            "clarity": clarity,
            "accepted": accepted,
            "review_score": review_score,
            "active": np.ones(n, dtype=bool),
            "first_recognition_period": np.full(n, np.nan),
            "cumulative_evaluations": np.zeros(n, dtype=int),
            "review_labor": (
                n * p["reviewers_per_paper"]
                if system in ("peer_review", "hybrid")
                else 0
            ),
            "publication_delay": (
                np.ones(n)
                if system == "peer_review"
                else np.zeros(n)
            ),
        }

    budget = max(1, int(round(p["evals_per_paper_per_period"] * n)))

    for t in range(periods):
        for system in SYSTEMS:
            s = state[system]

            # Peer review allows revision and resubmission.
            if system == "peer_review" and t > 0:
                rejected = ~s["accepted"]
                attempts = rejected & (
                    rng.random(n) < p["resubmission_probability"] / periods
                )
                if np.any(attempts):
                    s["clarity"][attempts] = np.clip(
                        s["clarity"][attempts]
                        + p["revision_effectiveness"]
                        * rng.beta(2, 5, attempts.sum()),
                        0,
                        1,
                    )
                    accepted_new, score_new = formal_review(
                        rng, papers, p, s["clarity"]
                    )
                    s["accepted"][attempts] = accepted_new[attempts]
                    s["review_score"][attempts] = score_new[attempts]
                    s["review_labor"] += (
                        attempts.sum() * p["reviewers_per_paper"]
                    )
                    s["publication_delay"][attempts] += 1

            counts = allocate_attention(
                rng=rng,
                system=system,
                papers=papers,
                p=p,
                confidence=s["confidence"],
                available=s["active"],
                formally_accepted=s["accepted"],
                total_budget=budget,
            )
            s["cumulative_evaluations"] += counts
            s["confidence"] = update_confidence(
                rng, papers, p, s["confidence"], counts
            )

            recognized_now = (
                (s["confidence"] >= p["recognition_threshold"])
                & s["active"]
            )
            newly = recognized_now & np.isnan(s["first_recognition_period"])
            s["first_recognition_period"][newly] = t

            # Claims can lose standing after evidence accumulates.
            if t >= p["min_withdraw_period"]:
                enough_evidence = s["cumulative_evaluations"] >= 2
                withdraw = (
                    (s["confidence"] < p["withdraw_threshold"])
                    & enough_evidence
                )
                s["active"][withdraw] = False

    outcomes = []
    total_true_value = papers["scientific_value"].sum() + 1e-12

    for system in SYSTEMS:
        s = state[system]
        confidence = s["confidence"]
        recognized = (
            (confidence >= p["recognition_threshold"])
            & s["active"]
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
        mean_time_to_recognition = np.nan
        if np.any(recognized_true):
            mean_time_to_recognition = np.nanmean(
                s["first_recognition_period"][recognized_true]
            )

        attention_coverage = (
            (s["cumulative_evaluations"] > 0).mean()
        )
        attention_gini = gini(s["cumulative_evaluations"].astype(float))
        delay_cost = (
            p["delay_weight"]
            * np.mean(s["publication_delay"])
            / max(periods, 1)
        )
        labor_cost = (
            p["labor_weight"]
            * s["review_labor"]
            / max(n * periods, 1)
        )

        score = (
            p["recovery_weight"] * true_value_recovered
            - p["false_weight"] * false_recognition
            - p["calibration_weight"] * weighted_calibration
            - delay_cost
            - labor_cost
        )

        outcomes.append(
            {
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
                "mean_publication_delay": np.mean(s["publication_delay"]),
                "rejected_final_share": (
                    (~s["accepted"]).mean()
                    if system == "peer_review"
                    else 0.0
                ),
            }
        )

    return outcomes


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(values == 0):
        return 0.0
    values = np.sort(values)
    n = values.size
    index = np.arange(1, n + 1)
    return float(
        (np.sum((2 * index - n - 1) * values))
        / (n * np.sum(values))
    )


def confidence_interval(p: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    se = math.sqrt(max(p * (1 - p), 0) / n)
    return (max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se))


def run(config: Config) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    rows: list[dict] = []

    for world in range(config.worlds):
        params = sample_parameters(rng)
        outcomes = simulate_world(
            rng, params, config.papers, config.periods
        )
        for outcome in outcomes:
            rows.append({"world": world, **outcome, **params})

    results = pd.DataFrame(rows)
    score_table = results.pivot(
        index="world", columns="system", values="score"
    )
    winners = score_table.idxmax(axis=1)

    win_rows = []
    for system in SYSTEMS:
        share = float((winners == system).mean())
        low, high = confidence_interval(share, config.worlds)
        win_rows.append(
            {
                "system": system,
                "win_share": share,
                "ci_low": low,
                "ci_high": high,
            }
        )
    win_rates = pd.DataFrame(win_rows).set_index("system")

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

    # Winner-condition diagnostics: which parameters distinguish winning worlds?
    winner_map = winners.rename("winner")
    diagnostics = results.drop_duplicates("world").set_index("world")
    diagnostics = diagnostics.join(winner_map)
    numeric_params = [
        c for c in diagnostics.columns
        if c not in metric_columns + ["system", "winner"]
        and pd.api.types.is_numeric_dtype(diagnostics[c])
    ]
    winner_conditions = diagnostics.groupby("winner")[numeric_params].mean().T

    return {
        "results": results,
        "win_rates": win_rates,
        "summary": summary,
        "winner_conditions": winner_conditions,
    }


def save_outputs(config: Config, tables: dict[str, pd.DataFrame]) -> Path:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if config.save_world_level:
        tables["results"].to_csv(out / "world_level_results.csv", index=False)
    tables["win_rates"].to_csv(out / "win_rates.csv")
    tables["summary"].to_csv(out / "summary.csv")
    tables["winner_conditions"].to_csv(out / "winner_conditions.csv")

    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    return out


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Simulate alternative scientific publication institutions."
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--worlds", type=int, default=Config.worlds)
    parser.add_argument("--papers", type=int, default=Config.papers)
    parser.add_argument("--periods", type=int, default=Config.periods)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument(
        "--no-world-level",
        action="store_true",
        help="Do not save the large world-level results file.",
    )
    args = parser.parse_args()

    if args.worlds < 1 or args.papers < 10 or args.periods < 1:
        parser.error("worlds >= 1, papers >= 10, and periods >= 1 are required.")

    return Config(
        seed=args.seed,
        worlds=args.worlds,
        papers=args.papers,
        periods=args.periods,
        output_dir=args.output_dir,
        save_world_level=not args.no_world_level,
    )


def main() -> None:
    config = parse_args()
    tables = run(config)
    out = save_outputs(config, tables)

    print("\nScientific Publication Ecosystem Simulator")
    print("=" * 44)
    print(f"Worlds: {config.worlds:,}")
    print(f"Papers/world: {config.papers:,}")
    print(f"Periods/world: {config.periods:,}")
    print(f"Unique simulated papers: {config.worlds * config.papers:,}")
    print("\nWin rates:")
    print(tables["win_rates"].round(4))
    print("\nMean outcomes:")
    mean_summary = tables["summary"].xs("mean", axis=1, level=1)
    print(mean_summary.round(4))
    print(f"\nOutputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
