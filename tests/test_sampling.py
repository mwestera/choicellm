import random

from choicellm.sampling import Item, basic_items, batched, comparison_items, sample_excluding


def test_basic_items():
    items = list(basic_items(['a', 'b', 'c']))
    assert [i.target for i in items] == ['a', 'b', 'c']
    assert [i.target_id for i in items] == [0, 1, 2]
    assert items[0].format_args == ['a']


def test_batched():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_sample_excluding_never_returns_excluded():
    s = sample_excluding(list(range(20)), 5, to_exclude=3, rng=random.Random(0))
    assert 3 not in s
    assert len(s) == 5


def test_comparison_items_deterministic_for_same_seed():
    pool = [f'w{i}' for i in range(20)]
    a = list(comparison_items(['w0'], pool, n_choices=3, n_comparisons=2, all_positions=False, rng=random.Random(42)))
    b = list(comparison_items(['w0'], pool, n_choices=3, n_comparisons=2, all_positions=False, rng=random.Random(42)))
    assert [i.choices for i in a] == [i.choices for i in b]
    assert all('w0' in i.choices and len(i.choices) == 3 for i in a)
