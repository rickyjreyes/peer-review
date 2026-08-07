#!/usr/bin/env python3
"""Sweep rejected-paper attention retention to locate the peer-review/open-triage breakeven.

Reviewer behavior is fixed to the committed empirical calibration. For each simulated
world, structural ecosystem parameters are sampled once, then the same world is run
under open triage and under peer review across a grid of rejected-paper attention
fractions. Separate deterministic RNG streams are used for each system so the open-
triage baseline is invariant to the attention-retention setting.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

import scientific_publication_simulator as ecosystem
import run_calibrated_ecosystem as calibrated


def run_system(system: str, seed: int, parameters: dict, papers: int, periods: int) -> dict:
    old_systems = ecosystem.SYSTEMS
    try:
        ecosystem.SYSTEMS = (system,)
        outcome = ecosystem.simulate_world(
            np.random.default_rng(seed), copy.deepcopy(parameters), papers, periods
        )[0]
    finally:
        ecosystem.SYSTEMS = old_systems
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260806)
    parser.add_argument('--worlds', type=int, default=300)
    parser.add_argument('--papers', type=int, default=250)
    parser.add_argument('--periods', type=int, default=25)
    parser.add_argument('--step', type=float, default=0.05)
    parser.add_argument('--calibration', type=Path, default=calibrated.DEFAULT_CALIBRATION)
    parser.add_argument('--output-dir', type=Path, default=Path('results/attention_breakeven'))
    args = parser.parse_args()

    calibrated._CALIBRATED_THETA = calibrated.load_theta(args.calibration)
    ecosystem.initialize_papers = calibrated.calibrated_initialize_papers
    ecosystem.formal_review = calibrated.calibrated_formal_review

    rho_grid = np.round(np.arange(0.0, 1.0 + args.step / 2.0, args.step), 10)
    rows = []

    for world in range(args.worlds):
        param_rng = np.random.default_rng(args.seed + 1000003 * world)
        base = calibrated.sample_calibration_propagated_parameters(param_rng)

        triage_seed = args.seed + 2000003 * world + 17
        triage = run_system('open_triage', triage_seed, base, args.papers, args.periods)

        for j, rho in enumerate(rho_grid):
            p = copy.deepcopy(base)
            p['rejected_attention_fraction'] = float(rho)
            peer_seed = args.seed + 3000017 * world + 31
            peer = run_system('peer_review', peer_seed, p, args.papers, args.periods)
            rows.append({
                'world': world,
                'rejected_attention_fraction': float(rho),
                'peer_true_value': peer['true_value_recovered'],
                'triage_true_value': triage['true_value_recovered'],
                'delta_peer_minus_triage': peer['true_value_recovered'] - triage['true_value_recovered'],
                'peer_false_recognition': peer['false_recognition_rate'],
                'triage_false_recognition': triage['false_recognition_rate'],
                'peer_calibration_mse': peer['calibration_mse'],
                'triage_calibration_mse': triage['calibration_mse'],
            })

    world_results = pd.DataFrame(rows)
    summary = world_results.groupby('rejected_attention_fraction').agg(
        peer_true_value=('peer_true_value', 'mean'),
        triage_true_value=('triage_true_value', 'mean'),
        delta_peer_minus_triage=('delta_peer_minus_triage', 'mean'),
        delta_sd=('delta_peer_minus_triage', 'std'),
        peer_win_fraction=('delta_peer_minus_triage', lambda x: float((x > 0).mean())),
        peer_false_recognition=('peer_false_recognition', 'mean'),
        triage_false_recognition=('triage_false_recognition', 'mean'),
        peer_calibration_mse=('peer_calibration_mse', 'mean'),
        triage_calibration_mse=('triage_calibration_mse', 'mean'),
    ).reset_index()
    summary['delta_se'] = summary['delta_sd'] / np.sqrt(args.worlds)
    summary['ci_low'] = summary['delta_peer_minus_triage'] - 1.96 * summary['delta_se']
    summary['ci_high'] = summary['delta_peer_minus_triage'] + 1.96 * summary['delta_se']

    # Linear interpolation between adjacent grid points if the mean delta crosses zero.
    crossing = None
    d = summary['delta_peer_minus_triage'].to_numpy()
    r = summary['rejected_attention_fraction'].to_numpy()
    for i in range(len(d) - 1):
        if d[i] == 0:
            crossing = float(r[i]); break
        if d[i] * d[i + 1] < 0:
            crossing = float(r[i] + (0 - d[i]) * (r[i + 1] - r[i]) / (d[i + 1] - d[i]))
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    world_results.to_csv(args.output_dir / 'world_results.csv', index=False)
    summary.to_csv(args.output_dir / 'summary.csv', index=False)
    metadata = {
        'seed': args.seed,
        'worlds': args.worlds,
        'papers_per_world': args.papers,
        'periods': args.periods,
        'attention_step': args.step,
        'breakeven_attention_retention': crossing,
        'breakeven_attention_loss': None if crossing is None else 1.0 - crossing,
        'interpretation': 'Breakeven is based on mean true-value recovery with reviewer behavior fixed to the committed empirical calibration; other ecosystem parameters remain structural draws.'
    }
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(summary[['rejected_attention_fraction','peer_true_value','triage_true_value','delta_peer_minus_triage','ci_low','ci_high','peer_win_fraction']].to_string(index=False))
    print('\nBreakeven attention retention:', crossing)
    if crossing is None:
        print('No mean-recovery crossing occurred on [0, 1].')
    else:
        print('Equivalent attention loss:', 1.0 - crossing)


if __name__ == '__main__':
    main()
