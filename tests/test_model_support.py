import json
import os

import pytest

from serving.core.model_support import unsupported_features, check_frontend_support

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(rel):
    with open(os.path.join(ROOT, "configs", "model", rel)) as f:
        return json.load(f)


def test_supported_models_pass():
    for rel in ("Qwen/Qwen3-235B-A22B.json", "meta-llama/Llama-3.1-8B.json", "mistralai/Mixtral-8x7B-v0.1.json"):
        assert unsupported_features(_cfg(rel)) == [], rel


def test_mla_and_partly_dense_moe_are_refused():
    ds = unsupported_features(_cfg("deepseek-ai/DeepSeek-V3.json"))
    assert any("MLA" in p for p in ds)
    assert any("n_routed_experts" in p for p in ds)
    assert any("first_k_dense_replace=3" in p for p in ds)
    assert any("shared experts" in p for p in ds)
    km = unsupported_features(_cfg("moonshotai/Kimi-K2-Thinking.json"))
    assert any("first_k_dense_replace=1" in p for p in km)


def test_llama4_interleaved_moe_is_refused():
    l4 = unsupported_features(_cfg("meta-llama/Llama-4-Maverick-17B-128E-Instruct.json"))
    assert any("text_config" in p for p in l4)
    assert any("subset of layers" in p for p in l4)
    assert not any("MLA" in p for p in l4)


def test_check_raises_with_every_reason():
    with pytest.raises(ValueError) as e:
        check_frontend_support("deepseek-ai/DeepSeek-V3", _cfg("deepseek-ai/DeepSeek-V3.json"))
    msg = str(e.value)
    assert "MLA" in msg and "first_k_dense_replace" in msg and "trace_generator.py" in msg
