import types

import pytest

pytest.importorskip("openai")
pytest.importorskip("tiktoken")

from choicellm.backends.openai import OpenAICompatibleBackend


def _fake_completion(top):
    ns = types.SimpleNamespace
    content0 = ns(top_logprobs=[ns(token=t, logprob=lp) for t, lp in top])
    return ns(choices=[ns(logprobs=ns(content=[content0]), message=ns(content="x"))])


def test_openai_model_resolves_label_ids_via_tiktoken():
    b = OpenAICompatibleBackend("gpt-4o", [" 1", " 2", " 3"], api_key="dummy")
    assert b.label_ids is not None and len(b.label_ids) == 3


def test_unresolvable_model_disables_logit_bias():
    b = OpenAICompatibleBackend("not a real model", ["1", "2"], base_url="http://localhost:1/v1", api_key="x")
    assert b.label_ids is None


def test_probs_reads_top_logprobs_and_softmaxes():
    b = OpenAICompatibleBackend("gpt-4o", [" a", " b"], api_key="dummy")
    # with logit_bias on, the API logprobs come back shifted by +LOGIT_BIAS; the backend subtracts it back
    comp = _fake_completion([("a", -1.0 + b.LOGIT_BIAS), ("b", -2.0 + b.LOGIT_BIAS)])

    class FakeChat:
        def create(self, **kw):
            self.kw = kw
            return comp

    b.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeChat()))
    probs = b.probs([{"role": "user", "content": "pick"}])
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[0] > probs[1]                                   # 'a' had the higher logprob
    assert b.client.chat.completions.kw["top_logprobs"] == 20
    assert b.client.chat.completions.kw["logit_bias"] is not None
