"""Local Huggingface transformers backend (requires the 'local' extra: torch + transformers)."""
import copy
import logging

import torch
import transformers

from .base import combine_label_probs

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TransformersBackend:
    """
    Local Huggingface causal LM (base or chat). Reads the model's logits directly to obtain the
    probability of each multiple-choice label -- no sampling, so results are deterministic.
    """

    def __init__(self, model_name, labels, prompt_start_for_cache=None, dtype=None):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
        self.labels_tokenized = [self.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        self.model = load_causal_lm(model_name, dtype=dtype)
        self.model.eval()
        if self.model.generation_config is not None:
            # we set max_new_tokens per call; clearing max_length avoids transformers' redundant-setting warning
            self.model.generation_config.max_length = None
        self.cache = create_cache(self.model, self.tokenizer, prompt_start_for_cache) if prompt_start_for_cache else None

    def probs(self, prompt) -> list[float]:
        return multiple_choice_probs(prompt, self.model, self.tokenizer, self.labels_tokenized, self.cache)


def load_causal_lm(model_name, dtype=None):
    """
    Load a causal LM, placing it on the available device. Quantized models (e.g. bitsandbytes 4/8-bit)
    must not be moved with `.to(...)`, so we let accelerate place them via device_map instead.
    """
    logging.info(f'Loading model on device: {DEVICE}')
    if DEVICE == 'cpu':
        logging.warning(
            'WARNING: No CUDA GPU available to PyTorch, running on CPU (this can be very slow). '
            'If you do have an NVIDIA GPU, your installed PyTorch build likely does not match your driver: '
            'a torch built for a newer CUDA than your driver supports silently falls back to CPU. '
            'Install a matching build from https://pytorch.org/get-started/locally/ '
            '(check your driver\'s max CUDA version with `nvidia-smi`).'
        )

    config = transformers.AutoConfig.from_pretrained(model_name)
    is_quantized = getattr(config, 'quantization_config', None) is not None

    kwargs = {}
    if dtype:
        kwargs['torch_dtype'] = dtype    # e.g. 'auto', 'float16', 'bfloat16'

    if is_quantized:
        return transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', **kwargs)
    return transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(DEVICE)


def multiple_choice_probs(prompt, model, tokenizer, labels_tokenized, cache=None) -> list[float]:
    """
    Feed the prompt into a local transformers model and obtain the probability of each choice label.

    Labels may be multi-token, so we grow the prompt by each shared label prefix and multiply the
    conditional next-token probabilities (see base.combine_label_probs). For single-token labels this
    reduces to a single forward pass.
    """
    if isinstance(prompt, str):
        prompt_encoded = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    else:   # openai-style messages
        prompt_encoded = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True, return_dict=True, enable_thinking=False,).to(DEVICE)

    def prob_for_prefix(prefix, next_token_ids):
        prompt_plus_prefix = torch.cat(
            (prompt_encoded['input_ids'], torch.tensor([list(prefix)], dtype=int).to(DEVICE)),
            dim=-1,
        )
        attention_mask_plus_prefix = torch.cat(
            (prompt_encoded['attention_mask'], torch.tensor([[1] * len(prefix)], dtype=int).to(DEVICE)),
            dim=-1,
        )
        model_output = model.generate(
            input_ids=prompt_plus_prefix,
            attention_mask=attention_mask_plus_prefix,  # only to avoid a warning
            pad_token_id=tokenizer.eos_token_id,
            output_logits=True,
            return_dict_in_generate=True,
            do_sample=False,
            max_new_tokens=5,  # just in case...
            past_key_values=copy.deepcopy(cache) if cache else None,
            temperature=None,  # only to avoid a warning
            top_p=None,
        )
        unique = list(set(next_token_ids))
        logits = model_output.logits[0]  # [0] is the first (and only) generated token
        probs = torch.nn.functional.softmax(logits[:, unique], dim=-1)[0].tolist()  # [0] is the first (and only) batch
        return dict(zip(unique, probs))

    return combine_label_probs(labels_tokenized, prob_for_prefix)


def create_cache(model, tokenizer, common_start) -> "transformers.DynamicCache":
    with torch.no_grad():
        if isinstance(common_start, str):
            inputs = tokenizer(common_start, return_tensors="pt").to(DEVICE)
        else:
            inputs = tokenizer.apply_chat_template(common_start, return_tensors="pt", return_dict=True, add_generation_prompt=True, enable_thinking=False,).to(DEVICE)

        cache = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  # only to avoid a warning
            pad_token_id=tokenizer.eos_token_id,
        ).past_key_values
    return cache
