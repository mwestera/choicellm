"""OpenAI-compatible chat API backend (requires the 'openai' extra: openai + tiktoken).

Works against the OpenAI API itself or any OpenAI-compatible server (llama.cpp, vLLM, Ollama, ...)
by pointing `base_url` at it. Torch-free: probabilities come from the API's top_logprobs, softmaxed
with the stdlib softmax in base.py.
"""
import logging
import os
from typing import Callable, Optional

from openai import OpenAI

from .base import softmax


class OpenAICompatibleBackend:
    """
    The choice labels must map onto single tokens. Their token ids (needed for `logit_bias`) are
    resolved by, in order: an explicit `tokenizer_name` (Huggingface); tiktoken (genuine OpenAI
    models); or the `model_name` treated as a Huggingface repo id (works for vLLM/TGI). If none of
    those work (e.g. a llama.cpp .gguf name), logit_bias is skipped and the label probabilities are
    read straight from the top_logprobs -- usually fine for single-token labels.

    logit_bias adds the same constant to every label, so it does not change the labels' relative
    probabilities (softmax is shift-invariant, and we subtract it back); it only nudges the model to
    emit a label as its first token and ensures the labels appear in the returned top_logprobs.
    """

    LOGIT_BIAS = 10
    TOP_LOGPROBS = 20  # the maximum the OpenAI API allows

    def __init__(self, model_name, labels, base_url=None, api_key=None, tokenizer_name=None):
        self.model_name = model_name
        self.labels = [l.strip() for l in labels]    # labels are read back from the API as bare tokens

        api_key = api_key or os.environ.get('OPENAI_API_KEY') or ('no-key-required' if base_url else None)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.label_ids = _resolve_label_ids(model_name, self.labels, tokenizer_name)

    def probs(self, prompt) -> list[float]:
        if not isinstance(prompt, list):
            raise ValueError('The OpenAI-compatible backend expects chat-style (message list) prompts; '
                             'use a prompt template with "chat": true.')

        request = dict(
            model=self.model_name,
            messages=prompt,
            logprobs=True,
            top_logprobs=self.TOP_LOGPROBS,
            max_completion_tokens=10,
        )
        if self.label_ids is not None:
            request['logit_bias'] = {label_id: self.LOGIT_BIAS for label_id in self.label_ids}

        completion = self.client.chat.completions.create(**request)

        bias_correction = self.LOGIT_BIAS if self.label_ids is not None else 0
        label_logprobs = {}
        for logprob_dict in completion.choices[0].logprobs.content[0].top_logprobs:
            token, logprob = logprob_dict.token, logprob_dict.logprob
            if token in self.labels:
                label_logprobs[token] = logprob - bias_correction    # undo the logit_bias we applied

        newline = "\n"
        logging.info(f'{prompt[-1]["content"].replace(newline, "/")} -> {completion.choices[0].message.content}')

        logits = [label_logprobs.get(label, -90) for label in self.labels]
        return softmax(logits)


def _resolve_label_ids(model_name, labels, tokenizer_name) -> Optional[list[int]]:
    """Token id of each (single-token) label, for logit_bias. Returns None if no tokenizer is available."""
    encode = _make_encoder(model_name, tokenizer_name)
    if encode is None:
        logging.info(
            f'No tokenizer available for {model_name!r}; not using logit_bias. '
            'Reading the label probabilities straight from the top logprobs instead, which is usually '
            'fine for single-token labels. Pass --tokenizer <hf-name> to force logit_bias (e.g. for llama.cpp).'
        )
        return None

    labels_tokenized = [encode(label) for label in labels]
    problematic_labels = [label for label, tokenized in zip(labels, labels_tokenized) if len(tokenized) != 1]
    if problematic_labels:
        raise ValueError(f'Some of the choice labels do not map onto single tokens for your selected model/tokenizer: {", ".join(problematic_labels)}')

    label_ids = [tokenized[0] for tokenized in labels_tokenized]
    logging.info(f'Using label ids: {label_ids}')
    return label_ids


def _make_encoder(model_name, tokenizer_name) -> Optional[Callable[[str], list]]:
    """A `str -> token ids` encoder, or None if we can't build one for this model."""
    if tokenizer_name:  # explicit override
        import transformers
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        return lambda s: hf_tokenizer.encode(s, add_special_tokens=False)

    import tiktoken
    try:  # genuine OpenAI models
        return tiktoken.encoding_for_model(model_name).encode    # for o1, mind https://github.com/openai/tiktoken/issues/367
    except KeyError:
        pass

    try:  # the model name as a Huggingface repo id (works for vLLM / TGI, not llama.cpp .gguf names)
        import transformers
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        logging.info(f'Resolved label token ids via the Huggingface tokenizer for {model_name!r}.')
        return lambda s: hf_tokenizer.encode(s, add_special_tokens=False)
    except Exception:  # not an HF id, offline, or transformers not installed -> fall back to no logit_bias
        return None
