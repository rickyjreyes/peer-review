#!/usr/bin/env python3
"""Calibrate a reviewer-behavior submodel to controlled-study observables.

This script deliberately does not assign empirical percentages directly to latent
coefficients such as ``reviewer_skill``. Instead, it samples internal parameter
sets, simulates reviewer behavior under fixed common random numbers, and retains
sets that reproduce observable targets:

* major-error detection rate;
* continuous inter-reviewer reliability;
* binary recommendation agreement (Cohen's kappa);
* positive- and null-result recommendation rates;
* major-error detection among reviewers recommending rejection.

The procedure is a lightweight rejection-ABC / simulation-based calibration.
It is not a full Bayesian posterior and should be interpreted as a set of
parameter combinations consistent with the selected targets and tolerances.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationConfig:
    seed: int = 20260806
    candidates: int = 4000
    retain: int = 200
    refinement_rounds: int = 4
    refinement_candidates: int = 1500
    manuscripts: int = 5000
    errors: int = 9
    output_dir: str = "results/empirical_calibration"
    targets: str = "data/empirical_targets.json"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    observed = np.mean(a == b)
    pa = np.mean(a)
    pb = np.mean(b)
    expected = pa * pb + (1.0 - pa) * (1.0 - pb)
    if expected >= 1.0 - 1e-12:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def load_targets(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def target_value(targets: dict, name: str) -> float:
    return float(targets["targets"][name]["value"])


def target_tolerance(targets: dict, name: str) -> float:
    return float(targets["targets"][name]["tolerance"])


def fixed_random_design(rng: np.random.Generator, n: int, errors: int) -> dict[str, np.ndarray]:
    """Generate common random numbers reused for every candidate parameter set."""
    return {
        "error_difficulty": rng.normal(size=errors),
        "ability": rng.normal(size=(n, 2)),
        "detect_uniform": rng.random(size=(n, 2, errors)),
        "latent_merit": rng.normal(size=n),
        "rating_noise": rng.normal(size=(n, 2)),
        "positive_noise": rng.normal(size=n),
        "null_noise": rng.normal(size=n),
        "positive_ability": rng.normal(size=n),
        "null_ability": rng.normal(size=n),
        "positive_detect_uniform": rng.random(size=(n, errors)),
        "null_detect_uniform": rng.random(size=(n, errors)),
    }


def simulate_observables(theta: dict[str, float], design: dict[str, np.ndarray]) -> dict[str, float]:
    n = design["latent_merit"].size
    errors = design["error_difficulty"].size

    logits = (
        theta["detection_intercept"]
        + theta["reviewer_ability_sd"] * design["ability"][..., None]
        - theta["error_difficulty_sd"] * design["error_difficulty"][None, None, :]
    )
    detected = design["detect_uniform"] < sigmoid(logits)
    detected_fraction = detected.mean(axis=2)

    ratings = (
        theta["merit_signal"] * design["latent_merit"][:, None]
        - theta["detected_error_penalty"] * detected_fraction
        + theta["rating_noise_sd"] * design["rating_noise"]
    )
    reliability = float(np.corrcoef(ratings[:, 0], ratings[:, 1])[0, 1])
    recommend = ratings > theta["recommendation_threshold"]
    kappa = cohen_kappa(recommend[:, 0], recommend[:, 1])

    # Emerson-style positive-vs-null experiment: identical underlying manuscript,
    # differing only in reported principal outcome.
    base_error_logits_pos = (
        theta["detection_intercept"]
        + theta["reviewer_ability_sd"] * design["positive_ability"][:, None]
        - theta["error_difficulty_sd"] * design["error_difficulty"][None, :]
    )
    base_error_logits_null = (
        theta["detection_intercept"]
        + theta["reviewer_ability_sd"] * design["null_ability"][:, None]
        - theta["error_difficulty_sd"] * design["error_difficulty"][None, :]
    )
    detected_pos = (design["positive_detect_uniform"] < sigmoid(base_error_logits_pos)).mean(axis=1)
    detected_null = (design["null_detect_uniform"] < sigmoid(base_error_logits_null)).mean(axis=1)

    positive_scores = (
        theta["trial_base_merit"]
        + theta["positive_outcome_bias"]
        - theta["detected_error_penalty"] * detected_pos
        + theta["rating_noise_sd"] * design["positive_noise"]
    )
    null_scores = (
        theta["trial_base_merit"]
        - theta["detected_error_penalty"] * detected_null
        + theta["rating_noise_sd"] * design["null_noise"]
    )
    positive_rate = float(np.mean(positive_scores > theta["recommendation_threshold"]))
    null_rate = float(np.mean(null_scores > theta["recommendation_threshold"]))

    reject = ~recommend
    if np.any(reject):
        reject_detection = float(detected_fraction[reject].mean())
    else:
        reject_detection = float("nan")

    return {
        "major_error_detection_rate": float(detected.mean()),
        "interreviewer_reliability": reliability,
        "interreviewer_kappa": kappa,
        "positive_result_recommendation_rate": positive_rate,
        "null_result_recommendation_rate": null_rate,
        "positive_result_recommendation_gap": positive_rate - null_rate,
        "major_error_detection_among_rejectors": reject_detection,
        "major_error_miss_rate": 1.0 - float(detected.mean()),
        "simulated_manuscripts": float(n),
        "simulated_errors_per_manuscript": float(errors),
    }


PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "detection_intercept": (-2.8, -0.2),
    "reviewer_ability_sd": (0.0, 1.5),
    "error_difficulty_sd": (0.0, 1.5),
    "merit_signal": (0.4, 3.2),
    "rating_noise_sd": (0.3, 3.8),
    "recommendation_threshold": (-1.5, 1.8),
    "positive_outcome_bias": (0.0, 3.0),
    "detected_error_penalty": (0.0, 8.0),
    "trial_base_merit": (0.5, 3.5),
}


def sample_prior(rng: np.random.Generator) -> dict[str, float]:
    return {name: rng.uniform(low, high) for name, (low, high) in PARAMETER_BOUNDS.items()}


def mutate_elite(
    rng: np.random.Generator,
    elite: pd.DataFrame,
    scale_fraction: float,
) -> dict[str, float]:
    parent = elite.iloc[int(rng.integers(0, len(elite)))]
    child: dict[str, float] = {}
    for name, (low, high) in PARAMETER_BOUNDS.items():
        scale = (high - low) * scale_fraction
        child[name] = float(np.clip(rng.normal(float(parent[name]), scale), low, high))
    return child


def distance(observed: dict[str, float], targets: dict) -> float:
    fitted = [
        "major_error_detection_rate",
        "interreviewer_reliability",
        "interreviewer_kappa",
        "positive_result_recommendation_rate",
        "null_result_recommendation_rate",
        "major_error_detection_among_rejectors",
    ]
    total = 0.0
    for name in fitted:
        value = observed[name]
        if not np.isfinite(value):
            return float("inf")
        z = (value - target_value(targets, name)) / target_tolerance(targets, name)
        total += z * z
    return float(total)


def run(config: CalibrationConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(config.seed)
    targets = load_targets(Path(config.targets))
    design = fixed_random_design(rng, config.manuscripts, config.errors)

    rows: list[dict[str, float | int]] = []

    def evaluate(theta: dict[str, float], stage: str) -> None:
        observed = simulate_observables(theta, design)
        rows.append({
            "candidate": len(rows),
            "stage": stage,
            "distance": distance(observed, targets),
            **theta,
            **{f"sim_{key}": value for key, value in observed.items()},
        })

    for _ in range(config.candidates):
        evaluate(sample_prior(rng), "global")

    for round_index in range(config.refinement_rounds):
        current = pd.DataFrame(rows).sort_values("distance", kind="stable")
        elite = current.head(min(30, len(current)))
        scale_fraction = 0.12 * (0.5 ** round_index)
        for _ in range(config.refinement_candidates):
            evaluate(
                mutate_elite(rng, elite, scale_fraction),
                f"refine_{round_index + 1}",
            )

    candidates = pd.DataFrame(rows).sort_values("distance", kind="stable").reset_index(drop=True)
    retained = candidates.head(min(config.retain, len(candidates))).copy()
    retained["abc_weight"] = np.exp(-0.5 * retained["distance"])
    if retained["abc_weight"].sum() > 0:
        retained["abc_weight"] /= retained["abc_weight"].sum()

    best_row = retained.iloc[0].to_dict()
    observable_names = [
        "major_error_detection_rate",
        "interreviewer_reliability",
        "interreviewer_kappa",
        "positive_result_recommendation_rate",
        "null_result_recommendation_rate",
        "positive_result_recommendation_gap",
        "major_error_detection_among_rejectors",
    ]
    summary_rows = []
    for name in observable_names:
        target = targets["targets"][name]
        values = retained[f"sim_{name}"].to_numpy()
        weights = retained["abc_weight"].to_numpy()
        if weights.sum() <= 0:
            weights = np.full_like(values, 1.0 / len(values))
        mean = float(np.sum(values * weights))
        summary_rows.append({
            "observable": name,
            "target": target["value"],
            "tolerance": target.get("tolerance"),
            "best_fit": best_row[f"sim_{name}"],
            "retained_weighted_mean": mean,
            "retained_p05": float(np.quantile(values, 0.05)),
            "retained_p95": float(np.quantile(values, 0.95)),
            "status": target["status"],
            "fit_role": target["fit_role"],
        })

    summary = pd.DataFrame(summary_rows)
    metadata = {
        "config": asdict(config),
        "method": "rejection_abc_with_common_random_numbers",
        "interpretation": "Retained parameter sets reproduce selected reviewer observables within a simulation-based distance. They are not direct empirical estimates of latent coefficients and are not a full Bayesian posterior.",
        "best_distance": float(best_row["distance"]),
        "retained_distance_max": float(retained["distance"].max()),
    }
    return retained, summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=CalibrationConfig.seed)
    parser.add_argument("--candidates", type=int, default=CalibrationConfig.candidates)
    parser.add_argument("--retain", type=int, default=CalibrationConfig.retain)
    parser.add_argument("--refinement-rounds", type=int, default=CalibrationConfig.refinement_rounds)
    parser.add_argument("--refinement-candidates", type=int, default=CalibrationConfig.refinement_candidates)
    parser.add_argument("--manuscripts", type=int, default=CalibrationConfig.manuscripts)
    parser.add_argument("--errors", type=int, default=CalibrationConfig.errors)
    parser.add_argument("--targets", default=CalibrationConfig.targets)
    parser.add_argument("--output-dir", default=CalibrationConfig.output_dir)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    candidates = 300 if args.quick else args.candidates
    manuscripts = 1200 if args.quick else args.manuscripts
    retain = min(40, args.retain) if args.quick else args.retain
    refinement_rounds = 1 if args.quick else args.refinement_rounds
    refinement_candidates = 150 if args.quick else args.refinement_candidates
    config = CalibrationConfig(
        seed=args.seed,
        candidates=candidates,
        retain=retain,
        refinement_rounds=refinement_rounds,
        refinement_candidates=refinement_candidates,
        manuscripts=manuscripts,
        errors=args.errors,
        output_dir=args.output_dir,
        targets=args.targets,
    )
    retained, summary, metadata = run(config)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    retained.to_csv(out / "accepted_parameters.csv", index=False)
    summary.to_csv(out / "calibration_summary.csv", index=False)
    with (out / "calibration_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    parameter_columns = [
        "detection_intercept",
        "reviewer_ability_sd",
        "error_difficulty_sd",
        "merit_signal",
        "rating_noise_sd",
        "recommendation_threshold",
        "positive_outcome_bias",
        "detected_error_penalty",
        "trial_base_merit",
    ]
    retained.iloc[[0]][parameter_columns].to_json(
        out / "best_fit_parameters.json", orient="records", indent=2
    )

    print("Empirical reviewer calibration")
    print(f"Candidates: {config.candidates:,}")
    print(f"Total parameter evaluations: {config.candidates + config.refinement_rounds * config.refinement_candidates:,}")
    print(f"Retained: {len(retained):,}")
    print(f"Calibration manuscripts: {config.manuscripts:,}")
    print(f"Best distance: {metadata['best_distance']:.4f}")
    print("\nObservable fit:")
    print(summary[["observable", "target", "best_fit", "retained_weighted_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
