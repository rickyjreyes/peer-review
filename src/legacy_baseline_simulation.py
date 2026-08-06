"""
Known-truth simulation comparing:
1. conventional prepublication peer review,
2. open publication with decentralized attention,
3. open publication with transparent triage and randomized exploration.

This is a structural experiment, not an empirical estimate of real-world peer review.
Run with: python peer_review_simulation.py
"""

import numpy as np
import pandas as pd

SEED = 42
WORLDS = 2000
PAPERS_PER_WORLD = 250
rng = np.random.default_rng(SEED)


def simulate_world(p: dict, n: int) -> tuple:
    true = rng.random(n) < p["truth_rate"]
    novelty = rng.beta(2, 5, n)
    prestige = rng.beta(2, 5, n)
    clarity = rng.beta(4, 2, n)

    # Scientific value is heavy-tailed, with some extra value assigned to novel true work.
    value = rng.lognormal(0, 1, n) * (1 + p["novelty_value_multiplier"] * novelty**2) * true
    harm = rng.lognormal(-0.2, 0.8, n) * (~true)
    total_evaluations = 2 * n

    # Conventional peer review: two selected reviewers per manuscript.
    reviewer_scores = []
    for _ in range(2):
        reviewer_scores.append(
            p["reviewer_accuracy"] * (true.astype(float) - 0.5) * 2
            + p["clarity_weight"] * (clarity - 0.5)
            + p["prestige_bias"] * (prestige - 0.5)
            - p["novelty_penalty"] * novelty
            + rng.normal(0, p["reviewer_noise"], n)
        )

    mean_review = np.mean(reviewer_scores, axis=0)
    accepted = mean_review > p["acceptance_threshold"]

    downstream_signal = (
        p["community_accuracy"] * (true.astype(float) - 0.5) * 2
        + 0.4 * prestige
        + 0.2 * clarity
        + rng.normal(0, 1, n)
    )
    peer_recognized = accepted & (downstream_signal > p["recognition_threshold"])

    # Open publication: equal total evaluation effort, but attention is unequal.
    appeal = (
        p["open_prestige_bias"] * prestige
        + p["topic_bias"] * rng.beta(2, 2, n)
        + 0.25 * clarity
        + p["novelty_attention"] * novelty
    )
    attention_prob = np.exp(
        p["attention_concentration"] * (appeal - appeal.max())
    )
    attention_prob /= attention_prob.sum()
    counts = rng.multinomial(total_evaluations, attention_prob)

    open_signal = np.full(n, -99.0)
    seen = counts > 0
    open_signal[seen] = (
        p["community_accuracy"] * (true[seen].astype(float) - 0.5) * 2
        + rng.normal(0, 1 / np.sqrt(counts[seen]))
    )
    open_recognized = seen & (open_signal > p["open_threshold"])

    # Open publication plus transparent triage and randomized exploration.
    merit = (
        0.45 * clarity
        + 0.25 * novelty
        + 0.15 * (1 - prestige)
        + 0.15 * rng.random(n)
    )
    triage_prob = np.exp(
        p["triage_concentration"] * (merit - merit.max())
    )
    triage_prob /= triage_prob.sum()
    triage_prob = (
        (1 - p["exploration_share"]) * triage_prob
        + p["exploration_share"] / n
    )
    triage_counts = rng.multinomial(total_evaluations, triage_prob)

    triage_signal = np.full(n, -99.0)
    triaged = triage_counts > 0
    triage_signal[triaged] = (
        p["community_accuracy"] * (true[triaged].astype(float) - 0.5) * 2
        + rng.normal(0, 1 / np.sqrt(triage_counts[triaged]))
    )
    triage_recognized = triaged & (triage_signal > p["open_threshold"])

    def metrics(recognized: np.ndarray, delay_cost: float = 0.0) -> tuple:
        true_value = value[recognized & true].sum()
        false_harm = harm[recognized & (~true)].sum()
        missed_value = value[(~recognized) & true].sum()

        score = (
            true_value
            - p["false_harm_weight"] * false_harm
            - p["missed_value_weight"] * missed_value
            - delay_cost
        )
        return (
            score,
            true_value / (value.sum() + 1e-9),
            (recognized & (~true)).sum() / max((~true).sum(), 1),
            (recognized & true).sum() / max(true.sum(), 1),
        )

    return (
        metrics(peer_recognized, p["delay_cost"] * n),
        metrics(open_recognized),
        metrics(triage_recognized),
    )


rows = []
for world in range(WORLDS):
    parameters = {
        "truth_rate": rng.uniform(0.35, 0.75),
        "novelty_value_multiplier": rng.uniform(1, 8),
        "reviewer_accuracy": rng.uniform(0.2, 1.5),
        "reviewer_noise": rng.uniform(0.5, 1.8),
        "clarity_weight": rng.uniform(0.1, 0.8),
        "prestige_bias": rng.uniform(0, 1.2),
        "novelty_penalty": rng.uniform(0, 1.2),
        "acceptance_threshold": rng.uniform(-0.2, 0.6),
        "community_accuracy": rng.uniform(0.4, 1.6),
        "recognition_threshold": rng.uniform(-0.2, 0.8),
        "open_prestige_bias": rng.uniform(0, 1.5),
        "topic_bias": rng.uniform(0, 0.8),
        "novelty_attention": rng.uniform(-0.2, 0.8),
        "attention_concentration": rng.uniform(0.5, 5),
        "open_threshold": rng.uniform(-0.2, 0.8),
        "triage_concentration": rng.uniform(0.5, 4),
        "exploration_share": rng.uniform(0.1, 0.5),
        "false_harm_weight": rng.uniform(0.5, 2),
        "missed_value_weight": rng.uniform(0.1, 0.5),
        "delay_cost": rng.uniform(0, 0.05),
    }

    outcomes = simulate_world(parameters, PAPERS_PER_WORLD)
    for system, result in zip(("peer", "open", "triage"), outcomes):
        rows.append(
            {
                "world": world,
                "system": system,
                "score": result[0],
                "true_value_recovered": result[1],
                "false_recognition_rate": result[2],
                "true_recognition_rate": result[3],
                **parameters,
            }
        )

results = pd.DataFrame(rows)
scores = results.pivot(index="world", columns="system", values="score")
winner_share = scores.idxmax(axis=1).value_counts(normalize=True).rename("win_share")
means = results.groupby("system")[
    ["score", "true_value_recovered", "false_recognition_rate", "true_recognition_rate"]
].mean()

print("\nWin share across simulated worlds:")
print(winner_share)
print("\nMean outcomes:")
print(means)

results.to_csv("peer_review_simulation_results.csv", index=False)
winner_share.to_csv("peer_review_simulation_win_share.csv")
means.to_csv("peer_review_simulation_mean_outcomes.csv")
