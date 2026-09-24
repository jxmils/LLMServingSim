import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from serving.core.kv_layout import KVLayout  # noqa: E402


def _old(config, tokens, num_npus, kv_bytes=2):
    n_head = config["num_attention_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // n_head)
    kv_dim = config.get("num_key_value_heads", n_head) * head_dim
    return 2 * kv_dim * tokens * config["num_hidden_layers"] * kv_bytes // num_npus


def test_qwen3_235b_tp64_is_the_replicated_head_not_a_64th():
    cfg = json.load(open(REPO_ROOT / "configs/model/Qwen/Qwen3-235B-A22B.json"))
    lay = KVLayout.from_config(cfg, 2)
    assert lay.kind == "gqa" and lay.kv_heads == 4 and lay.head_dim == 128
    assert lay.kv_heads_per_rank(64) == 1 and lay.replication(64) == 16
    assert lay.bytes_per_token(1, tp=64) == 48128          # 2 * 1 * 128 * 2 * 94
    assert _old(cfg, 1, 64) == 3008                          # the model-wide / 64 error
    assert lay.model_bytes_per_token() == 192512


def test_shipped_configs_unchanged_where_tp_divides_kv_heads():
    llama = json.load(open(REPO_ROOT / "configs/model/meta-llama/Llama-3.1-8B.json"))
    lay = KVLayout.from_config(llama, 2)
    for tp in (1, 2, 4, 8):
        for tokens in (1, 16, 4097):
            assert lay.bytes_per_token(tokens, tp) == _old(llama, tokens, tp)
    assert lay.bytes_per_token(1, tp=2) == 65536
    # beyond the KV-head count the old formula halves what a rank really stores
    assert lay.bytes_per_token(1, tp=16) == 16384 and _old(llama, 1, 16) == 8192
    qwen30 = json.load(open(REPO_ROOT / "configs/model/Qwen/Qwen3-30B-A3B-Instruct-2507.json"))
    lay = KVLayout.from_config(qwen30, 2)
    for tp in (1, 2, 4):
        assert lay.bytes_per_token(7, tp) == _old(qwen30, 7, tp)


def test_pipeline_stages_and_fp8():
    cfg = json.load(open(REPO_ROOT / "configs/model/meta-llama/Llama-3.1-8B.json"))
    lay = KVLayout.from_config(cfg, 2)
    assert lay.bytes_per_token(3, tp=2, pp=2) == _old(cfg, 3, 4)
    assert KVLayout.from_config(cfg, 1).bytes_per_token(1, tp=2) == 32768


def test_mla_latent_is_not_sharded_by_heads():
    deepseek = {"hidden_size": 7168, "num_attention_heads": 128, "num_hidden_layers": 61,
                "num_key_value_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    lay = KVLayout.from_config(deepseek, 2)
    assert lay.kind == "mla" and lay.latent_dim == 576
    assert lay.bytes_per_token(1, tp=1) == 576 * 2 * 61 == 70272
    assert lay.bytes_per_token(1, tp=64) == 70272 and lay.replication(64) == 64
