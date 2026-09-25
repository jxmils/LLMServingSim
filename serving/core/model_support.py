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

from .model_arch import decoder_config, moe_layout


def unsupported_features(config: Dict) -> List[str]:
    """Return the architecture features of ``config`` the trace generator
    does not model. Empty means the model can be executed faithfully.

    Modelled since 2026-09-25 (serving/core/model_arch.py): the multimodal
    ``text_config`` wrapper, ``n_routed_experts``, a mixed layer schedule
    (``first_k_dense_replace``, ``moe_layers`` / ``interleave_moe_layer_step``)
    and shared experts (``n_shared_experts``, Llama 4's implicit shared
    expert). Still refused: MLA attention, whose projections and KV reads
    differ from the GQA shapes the trace generator emits."""
    problems = []
    config = decoder_config(config)
    if "kv_lora_rank" in config or "q_lora_rank" in config:
        problems.append("MLA attention (kv_lora_rank/q_lora_rank): the trace generator emits "
                        "GQA-shaped q/k/v projections and KV reads")
    layout = moe_layout(config)
    if layout is not None and not layout.moe_layers:
        problems.append("MoE model whose layer schedule resolves to no MoE layer "
                        "(moe_layers/interleave_moe_layer_step/first_k_dense_replace)")
    return problems


def check_frontend_support(model_name: str, config: Dict) -> None:
    """Raise with every unsupported feature listed, or return."""
    problems = unsupported_features(config)
    if problems:
        joined = "\n  - ".join(problems)
        raise ValueError(f"model {model_name!r} cannot be executed faithfully by the frontend's "
                         f"trace generator:\n  - {joined}\nAdd support in trace_generator.py "
                         "(and the profile bundle) before running it on the panel backend.")
