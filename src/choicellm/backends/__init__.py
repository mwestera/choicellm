"""Backend selection. Heavy dependencies are imported lazily so the pure core stays importable."""
from .base import Backend, Prompt, softmax


def make_backend(model_name, labels, *, use_api=False, base_url=None, api_key=None,
                 tokenizer_name=None, prompt_start_for_cache=None, dtype=None) -> Backend:
    """Construct the appropriate backend, with a helpful message if its optional extra is missing."""
    if use_api or base_url:
        try:
            from .openai import OpenAICompatibleBackend
        except ImportError as e:
            raise SystemExit(
                "The OpenAI-compatible backend needs the 'openai' extra: pip install 'choicellm[openai]'"
            ) from e
        return OpenAICompatibleBackend(
            model_name, labels, base_url=base_url, api_key=api_key, tokenizer_name=tokenizer_name,
        )

    try:
        from .hf import TransformersBackend
    except ImportError as e:
        raise SystemExit(
            "Local (transformers) models need the 'local' extra: pip install 'choicellm[local]'"
        ) from e
    return TransformersBackend(
        model_name, labels, prompt_start_for_cache=prompt_start_for_cache, dtype=dtype,
    )
