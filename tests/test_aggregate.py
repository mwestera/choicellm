import io
import random

from choicellm.aggregate import aggregate, entropy, read_rows, sample_comparisons, sample_positions


def test_entropy_uniform_over_four_is_two_bits():
    assert abs(entropy([0.25] * 4) - 2.0) < 1e-9


def test_entropy_of_certain_outcome_is_zero():
    assert abs(entropy([1.0, 0.0])) < 1e-6


def test_aggregate_maps_mean_prob_onto_scale():
    # n_choices=2, so exponent 1/(2-1)=1; mean prob 0.25 -> 0.25*(5-1)+1 = 2.0
    rows = [
        {'target_id': 0, 'target': 'a', 'prob': 0.25, 'probs': (0.25, 0.75)},
        {'target_id': 0, 'target': 'a', 'prob': 0.25, 'probs': (0.25, 0.75)},
    ]
    out = aggregate(rows, n_choices=2, scale_start=1, scale_end=5)
    assert len(out) == 1
    assert out[0]['rating'] == 2.0
    assert list(out[0]) == ['target_id', 'target', 'rating', 'entropy']


def test_aggregate_sorts_by_target_id():
    rows = [
        {'target_id': 2, 'target': 'c', 'prob': 0.5, 'probs': (0.5, 0.5)},
        {'target_id': 0, 'target': 'a', 'prob': 0.5, 'probs': (0.5, 0.5)},
    ]
    assert [r['target_id'] for r in aggregate(rows, 2, 1, 5)] == [0, 2]


def test_read_rows_parses_types():
    csv_text = "target_id,comparison_id,position,target,choices,pred,prob,probs\n0,0,1,a,a;b,a,0.6,0.6;0.4\n"
    rows = read_rows(io.StringIO(csv_text))
    assert rows[0]['target_id'] == 0
    assert rows[0]['prob'] == 0.6
    assert rows[0]['probs'] == (0.6, 0.4)


def test_sample_comparisons_keeps_all_positions_of_chosen_comparisons():
    rows = [{'target_id': 0, 'comparison_id': c, 'target': 'a', 'prob': 0.5, 'probs': (0.5, 0.5)}
            for c in range(5) for _ in range(2)]
    out = sample_comparisons(rows, 3, random.Random(0))
    assert len({r['comparison_id'] for r in out}) == 3
    assert len(out) == 6  # 3 comparisons x 2 positions each


def test_sample_positions_limits_rows_per_comparison():
    rows = [{'target_id': 0, 'comparison_id': 0, 'target': 'a', 'prob': 0.5, 'probs': (0.5, 0.5)} for _ in range(4)]
    assert len(sample_positions(rows, 2, random.Random(0))) == 2
