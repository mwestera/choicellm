from choicellm.results import categorical_row, comparative_row, join, scalar_row
from choicellm.sampling import Item


def test_join_semicolon():
    assert join([0.1, 0.2, 0.3]) == '0.1;0.2;0.3'


def test_scalar_row():
    row = scalar_row(Item(target_id=0, target='great'), [0.1, 0.2, 0.7], scale=[1, 2, 3])
    assert row['pred'] == 3
    assert row['prob'] == 0.7
    assert row['rating'] == round(0.1 * 1 + 0.2 * 2 + 0.7 * 3, 6)
    assert list(row) == ['target_id', 'target', 'pred', 'prob', 'rating', 'probs']


def test_categorical_row():
    row = categorical_row(Item(target_id=1, target='x'), [0.5, 0.3, 0.2], category_names=['a', 'b', 'c'])
    assert row['pred'] == 'a'
    assert list(row) == ['target_id', 'target', 'pred', 'prob', 'probs']


def test_comparative_row_sampled():
    item = Item(target_id=2, target='t', choices=['t', 'u', 'v', 'w'], comparison_id=0, position=1)
    row = comparative_row(item, [0.1, 0.5, 0.2, 0.2])
    assert list(row) == ['target_id', 'comparison_id', 'position', 'target', 'choices', 'pred', 'prob', 'probs']
    assert row['pred'] == 'u'
    assert row['prob'] == 0.5  # probability at the target's position


def test_comparative_row_multicolumn_has_fewer_columns():
    row = comparative_row(Item(choices=['a', 'b']), [0.7, 0.3])
    assert list(row) == ['choices', 'pred', 'probs']
    assert row['pred'] == 'a'
