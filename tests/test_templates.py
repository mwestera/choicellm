import random

from choicellm.templates import ComparativeTemplate, PromptTemplate, ScalarTemplate

SCALAR = {
    "mode": "scalar", "chat": False,
    "system_prompt": "Rate on {scale} from {scale_min} to {scale_max}.",
    "prompt_format": "Item: {item}{label_hint}:",
    "scale": [1, 2, 3],
    "examples": [{"item": "good", "target_value": 1.0}],
}


def test_scalar_template_labels_and_format():
    t = PromptTemplate.from_dict(SCALAR)
    assert isinstance(t, ScalarTemplate)
    assert t.scale == [1, 2, 3]
    assert t.labels_for_logits == [' 1', ' 2', ' 3']  # non-chat gets a leading space
    out = t.format('sunshine')
    assert isinstance(out, str)
    assert 'sunshine' in out
    assert 'Rate on 1, 2, 3 from 1 to 3.' in out


def test_scalar_chat_builds_messages_without_space_prefix():
    t = PromptTemplate.from_dict(dict(SCALAR, chat=True))
    messages = t.format('sunshine')
    assert isinstance(messages, list)
    assert messages[0]['role'] == 'system'
    assert messages[-1]['role'] == 'user' and 'sunshine' in messages[-1]['content']
    assert t.labels_for_logits == ['1', '2', '3']
    # cache prefix is everything but the final user turn
    assert t.prompt_start_for_cache == messages[:-1]


def test_comparative_template_is_deterministic_and_formats_choices():
    data = {
        "mode": "comparative", "chat": False,
        "system_prompt": "Pick.", "prompt_format": "{choices}{label_hint}:",
        "n_choices": 2, "labels": ["A", "B"],
        "examples": [{"options": ["x", "y"], "target_index": 0}],
    }
    t = PromptTemplate.from_dict(data, rng=random.Random(0))
    assert isinstance(t, ComparativeTemplate)
    assert t.n_choices == 2
    out = t.format('foo', 'bar')
    assert 'A. foo' in out and 'B. bar' in out
