"""Backend abstraction and the pure (dependency-free) logit/probability maths.

Nothing here imports torch, transformers or openai, so it can be imported and unit-tested
without any of the heavy optional dependencies installed.
"""
import functools
import math
from typing import Callable, Protocol, Union, runtime_checkable

# A prompt is either a plain string (base models) or an OpenAI-style list of message dicts (chat models).
Prompt = Union[str, list[dict]]


@runtime_checkable
class Backend(Protocol):
    """Anything that can turn a prompt into a probability distribution over the choice labels."""

    def probs(self, prompt: Prompt) -> list[float]:
        ...


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a list of logits (stdlib only)."""
    largest = max(logits)
    exps = [math.exp(x - largest) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


@functools.cache
def label_prefixes(labels_tokenized: tuple[tuple[int, ...], ...]) -> list[tuple[tuple[int, ...], list[int], list[int]]]:
    """
    Group the choice labels by their shared token prefixes, for computing probabilities of
    multi-token labels (e.g. the two-token string '-1' for negative sentiment).

    Ordinary beam search isn't suitable, so we effectively run one beam per possible prefix.
    For each shared prefix this yields the prefix, the indices of the labels that share it, and
    the respective 'next token' each of those labels wants after the prefix. Multiplying the
    conditional next-token probabilities along each label's path gives that label's probability.
    """
    max_length = max(len(t) for t in labels_tokenized)
    unique_prefixes = {l[:n] for n in range(max_length) for l in labels_tokenized}
    result = []
    for prefix in unique_prefixes:
        label_ids = []
        next_token_ids = []
        for i, l in enumerate(labels_tokenized):
            if l[:len(prefix)] == prefix and len(l) > len(prefix):
                label_ids.append(i)
                next_token_ids.append(l[len(prefix)])
        result.append((prefix, label_ids, next_token_ids))
    return result


def combine_label_probs(
    labels_tokenized: list[list[int]],
    prob_for_prefix: Callable[[tuple[int, ...], list[int]], dict[int, float]],
) -> list[float]:
    """
    Compute the probability of each (possibly multi-token) label as the product of its
    conditional next-token probabilities.

    `prob_for_prefix(prefix, next_token_ids)` must return, for the prompt followed by `prefix`,
    a mapping from each candidate next token id to its probability (softmaxed over the candidates).
    This is the only part that needs an actual model, so it is injected -- which makes the
    combination logic here pure and unit-testable with a stub.
    """
    choice_probabilities = [1.0 for _ in labels_tokenized]
    for prefix, label_ids, next_token_ids in label_prefixes(tuple(tuple(t) for t in labels_tokenized)):
        token_to_prob = prob_for_prefix(prefix, next_token_ids)
        for label_id, next_token_id in zip(label_ids, next_token_ids):
            choice_probabilities[label_id] *= token_to_prob[next_token_id]
    return choice_probabilities
