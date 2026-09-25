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


def test_mla_is_the_remaining_refusal():
    ds = unsupported_features(_cfg("deepseek-ai/DeepSeek-V3.json"))
    assert len(ds) == 1 and "MLA" in ds[0]
    km = unsupported_features(_cfg("moonshotai/Kimi-K2-Thinking.json"))
    assert len(km) == 1 and "MLA" in km[0]


def test_llama4_interleaved_moe_is_admitted():
    # text_config wrapper, odd-layer MoE and the shared expert are modelled
    # (serving/core/model_arch.py); nothing left to refuse.
    assert unsupported_features(_cfg("meta-llama/Llama-4-Maverick-17B-128E-Instruct.json")) == []


def test_check_raises_with_every_reason():
    with pytest.raises(ValueError) as e:
        check_frontend_support("deepseek-ai/DeepSeek-V3", _cfg("deepseek-ai/DeepSeek-V3.json"))
    msg = str(e.value)
    assert "MLA" in msg and "trace_generator.py" in msg
