"""What the frontend's trace generator actually models, checked against a
model config before a run is allowed to execute on the panel backend.

The trace generator (``trace_generator.py``) emits standard Q/K/V projections
sized from ``num_attention_heads``/``num_key_value_heads``/``head_dim`` and,
for an MoE model, a dispatch/expert/combine block on *every* layer with the
expert count read from ``num_local_experts``/``num_experts``. A config whose
architecture differs from that is not refused by the frontend today: it is
executed as something else (an MLA model as GQA, a partly-dense MoE as
all-MoE, a ``n_routed_experts`` model as dense). On the panel path that is a
silently wrong workload, so it is refused here with the reason.
"""
from typing import Dict, List


def unsupported_features(config: Dict) -> List[str]:
    """Return the architecture features of ``config`` the trace generator
    does not model. Empty means the model can be executed faithfully."""
    problems = []
    if "text_config" in config:
        problems.append("multimodal wrapper config (text_config): the frontend reads decoder "
                        "fields from the top level only")
        config = config["text_config"]
    if "kv_lora_rank" in config or "q_lora_rank" in config:
        problems.append("MLA attention (kv_lora_rank/q_lora_rank): the trace generator emits "
                        "GQA-shaped q/k/v projections and KV reads")
    layers = int(config.get("num_hidden_layers", 0))
    if "n_routed_experts" in config and not ("num_experts" in config or "num_local_experts" in config):
        problems.append("MoE declared with n_routed_experts only: the frontend keys MoE on "
                        "num_experts/num_local_experts and would run the model as dense")
    is_moe = any(k in config for k in ("num_experts", "num_local_experts", "n_routed_experts"))
    if is_moe:
        first_dense = int(config.get("first_k_dense_replace", 0))
        if first_dense > 0:
            problems.append(f"first_k_dense_replace={first_dense}: the trace generator emits an "
                            "MoE block on every layer")
        step = int(config.get("interleave_moe_layer_step", 1))
        moe_layers = config.get("moe_layers")
        if step > 1 or (isinstance(moe_layers, list) and layers and len(moe_layers) != layers):
            problems.append("MoE on a subset of layers (interleave_moe_layer_step/moe_layers): the "
                            "trace generator emits an MoE block on every layer")
        if int(config.get("n_shared_experts", 0)) > 0:
            problems.append("shared experts (n_shared_experts): always-active expert compute is "
                            "not modelled")
    return problems


def check_frontend_support(model_name: str, config: Dict) -> None:
    """Raise with every unsupported feature listed, or return."""
    problems = unsupported_features(config)
    if problems:
        joined = "\n  - ".join(problems)
        raise ValueError(f"model {model_name!r} cannot be executed faithfully by the frontend's "
                         f"trace generator:\n  - {joined}\nAdd support in trace_generator.py "
                         "(and the profile bundle) before running it on the panel backend.")
