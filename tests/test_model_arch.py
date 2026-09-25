"""G4: mixed MoE layer schedules, shared experts and the HF spellings of the
expert count, derived once in model_arch and used by sizes, weights and the
trace generator."""
import json
import os

from serving.core import model_arch
from serving.core.memory_model import MemoryModel, calculate_sizes
from serving.core.model_support import unsupported_features
from serving.core.utils import get_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _raw(rel):
    with open(os.path.join(ROOT, "configs", "model", rel)) as f:
        return json.load(f)


def test_llama4_layout():
    lay = model_arch.moe_layout(_raw("meta-llama/Llama-4-Maverick-17B-128E-Instruct.json"))
    assert lay.num_experts == 128 and lay.top_k == 1
    assert lay.moe_layers == frozenset(range(1, 48, 2)) and len(lay.dense_layers) == 24
    assert lay.moe_intermediate == 8192 and lay.dense_intermediate == 16384
    assert lay.shared_experts == 1 and lay.shared_intermediate == 8192
    assert not lay.all_moe
    assert model_arch.architecture_name(_raw("meta-llama/Llama-4-Maverick-17B-128E-Instruct.json")) == "llama4"


def test_deepseek_kimi_qwen_layouts():
    ds = model_arch.moe_layout(_raw("deepseek-ai/DeepSeek-V3.json"))
    assert ds.num_experts == 256 and ds.top_k == 8 and ds.moe_layers == frozenset(range(3, 61))
    assert ds.shared_experts == 1 and ds.shared_intermediate == 2048 and ds.dense_intermediate == 18432
    km = model_arch.moe_layout(_raw("moonshotai/Kimi-K2-Thinking.json"))
    assert km.num_experts == 384 and km.moe_layers == frozenset(range(1, 61))
    qw = model_arch.moe_layout(_raw("Qwen/Qwen3-235B-A22B.json"))
    assert qw.all_moe and qw.shared_experts == 0 and qw.moe_intermediate == 1536
    assert model_arch.moe_layout(_raw("meta-llama/Llama-3.1-8B.json")) is None


def test_wrapper_config_is_flattened():
    cfg = get_config("meta-llama/Llama-4-Maverick-17B-128E-Instruct")
    assert "text_config" not in cfg and cfg["hidden_size"] == 5120 and cfg["model_type"] == "llama4_text"


def test_sizes_follow_the_layout():
    m = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
    _, dense_w, _ = calculate_sizes(m, "gate_up_proj", 1, parallel=1, fp=2)
    assert dense_w == 5120 * 2 * 16384 * 2                    # dense layers: intermediate_size_mlp
    _, shared_w, _ = calculate_sizes(m, "shared_gate_up_proj", 1, parallel=1, fp=2)
    assert shared_w == 5120 * 2 * 8192 * 2                    # shared expert: intermediate_size
    _, moe_w, _ = calculate_sizes(m, "moe", 1, parallel=1, fp=2)
    assert moe_w == (5120 * 128 * 2) + 128 * 3 * 5120 * 8192 * 2
    _, s1, _ = calculate_sizes(m, "shared_gate_up_proj", 1, parallel=8, fp=2)
    assert s1 == 5120 * 2 * (8192 // 8) * 2                   # shared experts are TP-sharded


def test_llama4_weight_matches_the_port_check():
    """The memory model's per-rank weight at tp=1 must equal the ModelSpec
    parameter count the port check reports (400.712 B params, bf16)."""
    mm = MemoryModel("meta-llama/Llama-4-Maverick-17B-128E-Instruct", 0, 0, 1, 1, 2000, 1, 16, 16,
                     False, False, None, None, ep_size=1, pp_size=1)
    params = mm.weight / 2
    assert abs(params - 400.712e9) / 400.712e9 < 0.002
    # heaviest pipeline stage: 2 stages of 24 layers each carry 12 MoE + 12 dense
    mm2 = MemoryModel("meta-llama/Llama-4-Maverick-17B-128E-Instruct", 0, 0, 2, 1, 2000, 1, 16, 16,
                      False, False, None, None, ep_size=1, pp_size=2)
    assert mm2.weight < mm.weight and mm2.weight > 0.49 * mm.weight


def test_support_check_now_admits_mixed_moe():
    assert unsupported_features(_raw("meta-llama/Llama-4-Maverick-17B-128E-Instruct.json")) == []
    ds = unsupported_features(_raw("deepseek-ai/DeepSeek-V3.json"))
    assert len(ds) == 1 and "MLA" in ds[0]
