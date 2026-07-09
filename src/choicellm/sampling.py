"""Turn raw inputs into the individual items to score. Pure; all randomness goes through an injected RNG."""
import itertools
import logging
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional


@dataclass
class Item:
    """One thing to score. For scalar/categorical only target is set; for comparative, choices/position are too."""
    target_id: Optional[int] = None
    target: Optional[str] = None
    choices: Optional[list] = None
    comparison_id: Optional[int] = None
    position: Optional[int] = None

    @property
    def format_args(self) -> list:
        """The positional arguments to pass to PromptTemplate.format."""
        return list(self.choices) if self.choices is not None else [self.target]


def basic_items(lines: Iterable[str]) -> Iterator[Item]:
    """Scalar/categorical: one item per input line."""
    for n, line in enumerate(lines):
        yield Item(target_id=n, target=line)


def multicolumn_items(rows: Iterable[list]) -> Iterator[Item]:
    """Comparative with predetermined choices: each input row is one full set of choices."""
    for row in rows:
        yield Item(choices=list(row))


def comparison_items(items: Iterable[str], compare_to: list, n_choices: int, n_comparisons: int,
                     all_positions: bool, rng: random.Random, seed_per_item: Optional[int] = None) -> Iterator[Item]:
    """
    Comparative: for each item, sample `n_comparisons` sets of alternatives from `compare_to`, and place the
    target item in a random position (or, with all_positions, every position) among each set of alternatives.
    """
    n_alternatives = n_choices - 1
    logging.info(f'Will do {n_comparisons * (n_choices if all_positions else 1)} comparisons per input line.')

    for item_id, item in enumerate(items):
        if seed_per_item is not None:
            rng.seed(seed_per_item)

        all_alternatives = sample_excluding(compare_to, n_comparisons * n_alternatives, item, rng)

        for comp_id, alternatives in enumerate(batched(all_alternatives, n_alternatives)):
            positions = range(n_choices) if all_positions else [rng.randint(0, n_alternatives)]
            for pos in positions:
                choices = alternatives[:pos] + [item] + alternatives[pos:]
                yield Item(target_id=item_id, target=item, choices=choices, comparison_id=comp_id, position=pos)


def sample_excluding(items: list, k: int, to_exclude, rng: random.Random) -> list:
    """Like rng.sample(items, k) but excluding one specific element. Original order not preserved."""
    sample = rng.sample(items, k=min(k + 1, len(items)))  # one extra, in case the excluded item is drawn
    try:
        sample.remove(to_exclude)
    except ValueError:  # excluded item wasn't drawn
        if len(sample) > k:
            sample.pop()

    if len(sample) < k:
        raise ValueError(
            'Not enough comparison items for n_choices x n_comparisons comparisons per item. '
            'Decrease n_choices or --n-comparisons, or provide a longer list to compare to (--compare-to).'
        )
    return sample


def batched(iterable, n: int) -> Iterator[list]:
    """itertools.batched was only added in Python 3.12; small shim to keep supporting 3.10/3.11."""
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch
