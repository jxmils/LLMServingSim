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


def test_deepseek_and_kimi_are_admitted():
    for rel in ("deepseek-ai/DeepSeek-V3.json", "moonshotai/Kimi-K2-Thinking.json"):
        assert unsupported_features(_cfg(rel)) == [], rel


def test_llama4_interleaved_moe_is_admitted():
    assert unsupported_features(_cfg("meta-llama/Llama-4-Maverick-17B-128E-Instruct.json")) == []


def test_check_raises_for_a_direct_q_mla_config():
    cfg = dict(_cfg("deepseek-ai/DeepSeek-V3.json"))
    cfg.pop("q_lora_rank")
    with pytest.raises(ValueError) as e:
        check_frontend_support("deepseek-lite-like", cfg)
    assert "q_lora_rank" in str(e.value) and "trace_generator.py" in str(e.value)
