"""Prompt templates: assemble the system prompt, few-shot examples and final question for each mode.

Pure (stdlib only). A template is a dataclass exposing `format(*items) -> str | messages`, a
`prompt_start_for_cache` (everything but the final question, for KV caching), and a `result(item, probs)`
method that turns choice probabilities into an output row.
"""
import json
import logging
import random
import string
from dataclasses import dataclass, field
from typing import Optional, Union

from . import results

Prompt = Union[str, list[dict]]


@dataclass(kw_only=True)
class PromptTemplate:
    mode: str
    is_chat: bool
    system_prompt: str
    examples: list           # list of (user_content, assistant_response) tuples, already rendered
    prompt: str              # the final question, with `{}` placeholder(s) for the item(s)
    labels_for_logits: list  # the tokens whose probabilities the backend should read

    def format(self, *items) -> Prompt:
        final_user = self.prompt.format(*items)
        if self.is_chat:
            return self._messages(final_user)
        return '\n\n'.join([self.system_prompt] + [f'{u} {a}' for u, a in self.examples] + [final_user])

    def _messages(self, final_user: str) -> list[dict]:
        messages = [{'role': 'system', 'content': self.system_prompt}]
        for user, assistant in self.examples:
            messages.append({'role': 'user', 'content': user})
            messages.append({'role': 'assistant', 'content': assistant})
        messages.append({'role': 'user', 'content': final_user})
        return messages

    @property
    def prompt_start_for_cache(self) -> Prompt:
        """Everything up to (but excluding) the final question -- shared across items, so cacheable."""
        if self.is_chat:
            return self._messages(final_user='')[:-1]
        return '\n\n'.join([self.system_prompt] + [f'{u} {a}' for u, a in self.examples])

    def result(self, item, probs) -> dict:
        raise NotImplementedError

    def __str__(self):
        if self.is_chat:
            return json.dumps(self._messages(self.prompt), indent=2)
        return '\n\n'.join([self.system_prompt] + [f'{u} {a}' for u, a in self.examples] + [self.prompt])

    @classmethod
    def from_json(cls, file, rng: Optional[random.Random] = None) -> "PromptTemplate":
        data = json.load(file) if hasattr(file, 'read') else file
        return cls.from_dict(data, rng=rng)

    @classmethod
    def from_dict(cls, data: dict, rng: Optional[random.Random] = None) -> "PromptTemplate":
        data = dict(data)
        data.pop('_comment', None)
        if 'chat' not in data:
            logging.warning('WARNING: The prompt .json file does not specify whether to use chat-style prompting; assuming "chat": false')
        is_chat = data.pop('chat', False)
        mode = data.pop('mode')

        if mode in ('categorical', 'comparative') and 'labels' not in data:
            # TODO: allow using the categories/items themselves as labels; for now, just A, B, C, ...
            data['labels'] = list(string.ascii_uppercase)

        if mode == 'scalar':
            template_cls, built = ScalarTemplate, _build_scalar(**data)
        elif mode == 'categorical':
            template_cls, built = CategoricalTemplate, _build_categorical(**data)
        elif mode == 'comparative':
            template_cls, built = ComparativeTemplate, _build_comparative(rng=rng or random.Random(), **data)
        else:
            raise ValueError(f'Unknown prompt mode: {mode!r} (expected scalar, categorical or comparative)')

        labels_for_logits = built.pop('labels_for_logits')
        if not is_chat:
            labels_for_logits = [' ' + l for l in labels_for_logits]

        return template_cls(mode=mode, is_chat=is_chat, labels_for_logits=labels_for_logits, **built)


@dataclass(kw_only=True)
class ScalarTemplate(PromptTemplate):
    scale: list

    @property
    def n_choices(self) -> int:
        return len(self.scale)

    def result(self, item, probs) -> dict:
        return results.scalar_row(item, probs, self.scale)


@dataclass(kw_only=True)
class CategoricalTemplate(PromptTemplate):
    category_names: list

    def result(self, item, probs) -> dict:
        return results.categorical_row(item, probs, self.category_names)


@dataclass(kw_only=True)
class ComparativeTemplate(PromptTemplate):
    n_choices: int

    def result(self, item, probs) -> dict:
        return results.comparative_row(item, probs)


def _build_scalar(system_prompt, prompt_format, examples, scale) -> dict:
    original_scale = scale
    if all(isinstance(n, int) for n in scale):
        rating_as_int = True
        scale = list(scale)
    else:
        rating_as_int = False
        scale = [float(n) for n in scale]

    labels = [str(number) for number in scale]
    scale_min, scale_max = scale[0], scale[-1]
    system_prompt = system_prompt.format(scale=', '.join(labels), scale_min=scale_min, scale_max=scale_max)
    label_hint = f' (on scale {scale_min}-{scale_max})'

    examples_list = []
    n = 0  # in case there are no examples
    for n, example in enumerate(examples, start=1):
        rating_scaled = example['target_value'] * (scale_max - scale_min) + scale_min
        rating_scaled = int(rating_scaled) if rating_as_int else float(rating_scaled)
        examples_list.append((
            prompt_format.format(n=n, item=example['item'], label_hint=label_hint),
            str(rating_scaled),
        ))
    prompt = prompt_format.format(n=n + 1, item='{}', label_hint=label_hint)

    return dict(
        system_prompt=system_prompt,
        examples=examples_list,
        prompt=prompt,
        labels_for_logits=[str(i) for i in original_scale],
        scale=list(original_scale),
    )


def _build_categorical(system_prompt, prompt_format, examples, categories, labels=None) -> dict:
    if labels:
        categories_full = '\n'.join(f'{l}. {c}: {d}' for l, (c, d) in zip(labels, categories.items()))
    else:
        categories_full = '\n'.join(f'- {c}: {d}' for c, d in categories.items())
    category_names = list(categories.keys())
    system_prompt = system_prompt.format(categories=categories_full)
    label_hint = f' (choose from {"/".join(labels)})' if labels else ''

    examples_list = []
    n = 0
    for n, example in enumerate(examples, start=1):
        examples_list.append((
            prompt_format.format(n=n, item=example['item'], label_hint=label_hint),
            f"{labels[example['target_index']]} ({category_names[example['target_index']]})",
        ))
    prompt = prompt_format.format(n=n + 1, item='{}', label_hint=label_hint)

    return dict(
        system_prompt=system_prompt,
        examples=examples_list,
        prompt=prompt,
        labels_for_logits=labels[:len(categories)],
        category_names=category_names,
    )


def _build_comparative(system_prompt, prompt_format, n_choices, examples, labels=None, rng=None) -> dict:
    rng = rng or random.Random()
    if labels:
        labels = labels[:n_choices]

    def make_choices_str(choices, labels=None):
        if labels:
            return '\n'.join(f'{label}. {choice}' for label, choice in zip(labels, choices))
        return '\n'.join(f'- {choice}' for choice in choices)

    label_hint = f' (choose from {"/".join(labels)})' if labels else ''

    examples_list = []
    n = 0
    for n, example in enumerate(examples, start=1):
        example = dict(example)
        example['options'] = list(example['options'])
        if example['target_index'] >= n_choices:
            target = example['options'][example['target_index']]
            example['target_index'] = rng.randint(0, n_choices - 1)
            example['options'][example['target_index']] = target
        example['options'] = example['options'][:n_choices]

        examples_list.append((
            prompt_format.format(n=n, choices=make_choices_str(example['options'], labels), label_hint=label_hint),
            labels[example['target_index']],
        ))
    prompt = prompt_format.format(n=n + 1, choices=make_choices_str(['{}'] * n_choices, labels), label_hint=label_hint)

    return dict(
        system_prompt=system_prompt,
        examples=examples_list,
        prompt=prompt,
        labels_for_logits=labels[:n_choices],
        n_choices=n_choices,
    )


# ---------------------------------------------------------------------------
# Example templates (for the `choicellm-template` helper command).
# ---------------------------------------------------------------------------

TEMPLATE_SCALAR = {
    "mode": "scalar",
    "chat": False,
    "system_prompt": "# Concrete vs. abstract\n\nSome words and phrases are more concrete, some are more abstract. "
                     "We can indicate how concrete a given word or phrase is, as a rating on a scale {scale}, "
                     "with {scale_min} very abstract, and {scale_max} very concrete.",
    "prompt_format": "## Example {n}.\n\nWord/phrase: {item}\n\nConcreteness rating{label_hint}:",
    "scale": [1, 2, 3, 4, 5],
    "examples": [
        {"item": "essentialness", "target_value": 0.0},
        {"item": "frangipane", "target_value": 1.0},
        {"item": "although", "target_value": 0.0},
        {"item": "blackbird", "target_value": 1.0},
        {"item": "bat", "target_value": 1.0},
        {"item": "hope", "target_value": 0.0},
    ],
    "_comment": "In the examples, \"target_value\" is the target rating as a float between [0, 1]. This will be automatically mapped to whichever scale is used. This facilitates trying different scales with the same examples."
}

TEMPLATE_SCALAR_CHAT = TEMPLATE_SCALAR.copy()
TEMPLATE_SCALAR_CHAT.update({
    'chat': True,
    'system_prompt': 'Some words and phrases are more concrete, some are more abstract. You are a helpful assistant, '
                     'who is an expert on rating how *concrete* a given word or phrase is, as a rating on a scale '
                     '{scale}, with {scale_min} very abstract, and {scale_max} very concrete.',
    'prompt_format': '## Question {n}.\n\nWord/phrase: {item}\n\nHow concrete is this{label_hint}?',
})

TEMPLATE_COMPARATIVE = {
    "mode": "comparative",
    "chat": False,
    "system_prompt": "# Concrete vs. abstract\n\nSome words and phrases are more concrete, some are more abstract. "
                     "We can often tell which word or phrase, from a given set, is the _most concrete_ one.",
    "prompt_format": "## Example {n}.\n\n{choices}\n\nThe most concrete is{label_hint}:",
    "n_choices": 4,
    "labels": ["A", "B", "C", "D"],
    "examples": [
        {"options": ["essentialness", "simulation", "bat", "living"], "target_index": 2},
        {"options": ["blackbird", "high", "cause", "although"], "target_index": 0},
        {"options": ["signature", "frangipane", "hope", "simulation"], "target_index": 1},
    ],
    "_comment": "If \"labels\" is omitted, will use the options themselves as responses (recommended only if the options are single words). Under \"examples\", \"target_index\" is always the integer index of the correct choice in the list, 0-based."
}

TEMPLATE_COMPARATIVE_CHAT = TEMPLATE_COMPARATIVE.copy()
TEMPLATE_COMPARATIVE_CHAT.update({
    'chat': True,
    'system_prompt': 'Some words and phrases are more concrete, some are more abstract. You are a helpful assistant, '
                     'who is an expert on deciding which of several words or phrases is the _most concrete_.',
    'prompt_format': '## Question {n}.\n\n{choices}\n\nWhich of these is the most concrete{label_hint}?',
})

TEMPLATE_CATEGORICAL = {
    "mode": "categorical",
    "chat": False,
    "system_prompt": "# Concrete vs. abstract\n\nSome words and phrases are more concrete, some are more abstract. "
                     "We distinguish the following categories:\n\n{categories}",
    "prompt_format": "## Example {n}.\n\nWord/phrase: {item}\n\nThis word/phrase fits best in category{label_hint}:",
    "labels": ["A", "B", "C"],
    "categories": {
        "concrete": "the word refers to something actual, concrete, empirical",
        "neutral": "the word is neither abstract nor concrete",
        "abstract": "the word refers to something conceptual, intangible, theoretical or vague"
    },
    "examples": [
        {"item": "essentialness", "target_index": 0},
        {"item": "frangipane", "target_index": 2},
        {"item": "although", "target_index": 0},
        {"item": "blackbird", "target_index": 2},
        {"item": "bat", "target_index": 2},
        {"item": "hope", "target_index": 0}
    ],
    "_comment": "If \"labels\" is omitted, will use the options themselves as responses (recommended only if the category names are single words). Under \"examples\", \"target_index\" is always the integer index of the correct choice in the list, 0-based."
}

TEMPLATE_CATEGORICAL_CHAT = TEMPLATE_CATEGORICAL.copy()
TEMPLATE_CATEGORICAL_CHAT.update({
    'chat': True,
    'system_prompt': 'Some words and phrases are more concrete, some are more abstract. You are a helpful assistant, '
                     'who is an expert on categorizing words and phrases into one of three categories:\n\n{categories}',
    'prompt_format': '## Question {n}.\n\n{item}\n\nIn which category does this word/phrase fit best{label_hint}?',
})


def main():
    import argparse

    argparser = argparse.ArgumentParser('choicellm-template', description='Generate a prompt template .json file to adapt.')
    argparser.add_argument('--chat', action='store_true', help='Generate a chat/instruct-style template; otherwise plain text generation.')
    group = argparser.add_mutually_exclusive_group(required=True)
    group.add_argument('--comparative', action='store_true', help='Template for comparative prompting.')
    group.add_argument('--scalar', action='store_true', help='Template for scalar (rating) prompting.')
    group.add_argument('--categorical', action='store_true', help='Template for categorical (classification) prompting.')

    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    mode = 'comparative' if args.comparative else 'scalar' if args.scalar else 'categorical'
    template = globals()[f'TEMPLATE_{mode.upper()}{"_CHAT" if args.chat else ""}']

    logging.info(f'Creating a JSON-format prompt template for {mode}, '
                 f'{"chat-style" if args.chat else "plain text generation"} prompting. '
                 f'Save this to a .json file, and modify to suit your needs.')
    print(json.dumps(template, indent=2))


if __name__ == '__main__':
    main()
