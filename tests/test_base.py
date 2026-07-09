"""Pure logit/probability maths -- no torch needed."""
from choicellm.backends.base import Backend, combine_label_probs, label_prefixes, softmax


def test_softmax_sums_to_one():
    out = softmax([1.0, 2.0, 3.0])
    assert abs(sum(out) - 1.0) < 1e-12
    assert out[2] > out[1] > out[0]


def test_softmax_is_shift_invariant():
    a = softmax([0.0, 1.0, 2.0])
    b = softmax([10.0, 11.0, 12.0])
    assert all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def test_label_prefixes_single_tokens():
    assert label_prefixes(((2,), (3,))) == [((), [0, 1], [2, 3])]


def test_combine_single_token_labels():
    def prob_for_prefix(prefix, next_token_ids):
        table = {5: 0.2, 6: 0.3, 7: 0.5}
        return {t: table[t] for t in set(next_token_ids)}

    assert combine_label_probs([[5], [6], [7]], prob_for_prefix) == [0.2, 0.3, 0.5]


def test_combine_multi_token_label_multiplies_conditionals():
    # label 0 = tokens (1, 2); label 1 = token (3,)
    table = {(): {1: 0.4, 3: 0.6}, (1,): {2: 0.5}}

    def prob_for_prefix(prefix, next_token_ids):
        return {t: table[tuple(prefix)][t] for t in set(next_token_ids)}

    out = combine_label_probs([[1, 2], [3]], prob_for_prefix)
    assert out[0] == 0.4 * 0.5
    assert out[1] == 0.6


def test_fake_backend_satisfies_protocol():
    class Fake:
        def probs(self, prompt):
            return [1.0]

    assert isinstance(Fake(), Backend)
