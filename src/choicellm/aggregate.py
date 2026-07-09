"""Aggregate comparative results (one row per comparison) into one score per item.

Stdlib only -- no pandas. From comparative output columns
(target_id, comparison_id, position, target, ..., prob, probs) to (target_id, target, rating, entropy).
This is a one-off post-processing step, so plain csv/dict handling is plenty fast.
"""
import argparse
import csv
import logging
import math
import random
import sys
from collections import defaultdict

from .results import N_DECIMALS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser('choicellm-aggregate', description='Aggregate comparative results into one score per item.')
    p.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                   help='CSV of comparative results (default: stdin).')
    p.add_argument('--onlyscore', action='store_true', help='Output only target_id and rating rather than the full csv.')
    p.add_argument('--n-positions', dest='n_positions', type=int, default=None,
                   help='If the data has all positions, randomly sample this many positions per comparison.')
    p.add_argument('--n-comparisons', dest='n_comparisons', type=int, default=None,
                   help='Randomly sample only this many comparisons per item.')
    p.add_argument('--scale', default='1,5', help='start,end of the scale to map scores onto (default: 1,5).')
    p.add_argument('--seed', type=int, default=None, help='Seed for --n-positions / --n-comparisons sampling.')
    return p


def read_rows(file) -> list[dict]:
    """Parse a comparative-results CSV into row dicts with typed prob/probs fields."""
    rows = []
    for row in csv.DictReader(file):
        rows.append({
            'target_id': int(row['target_id']),
            'comparison_id': int(row['comparison_id']),
            'target': row['target'],
            'prob': float(row['prob']),
            'probs': tuple(float(s) for s in row['probs'].split(';')),
        })
    return rows


def entropy(probs) -> float:
    return -sum(p * math.log2(max(p, 1e-8)) for p in probs)


def sample_positions(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Keep only n randomly chosen rows (positions) per (target_id, comparison_id)."""
    groups: dict = defaultdict(list)
    for row in rows:
        groups[(row['target_id'], row['comparison_id'])].append(row)
    sampled = []
    for group in groups.values():
        sampled.extend(rng.sample(group, min(n, len(group))))
    return sampled


def sample_comparisons(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Keep only n randomly chosen comparisons (all their positions) per target_id."""
    comparison_ids: dict = defaultdict(list)   # target_id -> distinct comparison_ids, in order of appearance
    rows_by: dict = defaultdict(list)          # (target_id, comparison_id) -> rows
    for row in rows:
        key = (row['target_id'], row['comparison_id'])
        if not rows_by[key]:
            comparison_ids[row['target_id']].append(row['comparison_id'])
        rows_by[key].append(row)

    sampled = []
    for target_id, comps in comparison_ids.items():
        for comparison_id in rng.sample(comps, min(n, len(comps))):
            sampled.extend(rows_by[(target_id, comparison_id)])
    return sampled


def aggregate(rows: list[dict], n_choices: int, scale_start: float, scale_end: float) -> list[dict]:
    """One score per item: mean win-probability mapped onto the scale, plus mean entropy."""
    probs_per_item: dict = defaultdict(list)
    entropies_per_item: dict = defaultdict(list)
    for row in rows:
        key = (row['target_id'], row['target'])
        probs_per_item[key].append(row['prob'])
        entropies_per_item[key].append(entropy(row['probs']))

    results = []
    for (target_id, target), probs in probs_per_item.items():
        mean_prob = sum(probs) / len(probs)
        mean_entropy = sum(entropies_per_item[(target_id, target)]) / len(probs)
        # rating from the normalized win probability: prob(1 vs n_choices) = prob(1 vs 1) ** (n_choices - 1)
        rating = (mean_prob ** (1 / (n_choices - 1))) * (scale_end - scale_start) + scale_start
        results.append({
            'target_id': target_id,
            'target': target,
            'rating': round(rating, N_DECIMALS),
            'entropy': round(mean_entropy, N_DECIMALS),
        })
    results.sort(key=lambda r: r['target_id'])
    return results


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = build_parser().parse_args(argv)
    scale_start, scale_end = (int(i) for i in args.scale.split(','))

    rows = read_rows(args.file)
    if not rows:
        return
    n_choices = len(rows[0]['probs'])

    if args.n_positions is not None or args.n_comparisons is not None:
        rng = random.Random(args.seed)
        if args.seed is not None:
            logging.info(f'seed = {args.seed}')
        if args.n_positions is not None:
            rows = sample_positions(rows, args.n_positions, rng)
        if args.n_comparisons is not None:
            rows = sample_comparisons(rows, args.n_comparisons, rng)

    aggregated = aggregate(rows, n_choices, scale_start, scale_end)

    columns = ['target_id', 'rating'] if args.onlyscore else ['target_id', 'target', 'rating', 'entropy']
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(aggregated)


if __name__ == '__main__':
    main()
