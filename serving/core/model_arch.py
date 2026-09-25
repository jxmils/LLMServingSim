"""Model-architecture facts the frontend derives from an HF config.

One place answers "is this an MoE model, on which layers, with how many
routed and shared experts, and what widths do the dense and expert FFNs
have" so the memory model, the config builder, the trace generator and the
support check agree. The HF families spell these differently:

* expert count: ``num_local_experts`` (Mixtral, Llama 4), ``num_experts``
  (Qwen3), ``n_routed_experts`` (DeepSeek-V3, Kimi-K2);
* MoE layer set: every layer (Mixtral, Qwen3), ``moe_layers`` /
  ``interleave_moe_layer_step`` (Llama 4: odd layers), ``first_k_dense_replace``
  (DeepSeek: the first k layers are dense);
* expert width: ``moe_intermediate_size`` (Qwen3, DeepSeek) or plain
  ``intermediate_size`` (Llama 4, whose dense layers use
  ``intermediate_size_mlp``);
* shared experts: ``n_shared_experts`` of width ``moe_intermediate_size``
  each (DeepSeek, Kimi), or the one implicit ``shared_expert`` of a Llama 4
  MoE layer (``Llama4TextMoe.shared_expert``, width ``intermediate_size``).

``decoder_config`` unwraps the multimodal ``text_config`` wrapper (Llama 4)
so every consumer reads decoder fields from one flat dict.
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional

_EXPERT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts")


def decoder_config(config: Dict) -> Dict:
    """Flat decoder config: ``text_config`` fields over the wrapper's own."""
    if "text_config" in config and isinstance(config["text_config"], dict):
        flat = dict(config)
        flat.update(config["text_config"])
        flat.pop("text_config", None)
        flat.pop("vision_config", None)
        return flat
    return config


def num_experts(config: Dict) -> Optional[int]:
    config = decoder_config(config)
    for key in _EXPERT_KEYS:
        if key in config:
            return int(config[key])
    return None


@dataclass(frozen=True)
class MoELayout:
    num_experts: int
    top_k: int
    moe_intermediate: int      # width of one routed expert
    shared_experts: int        # always-active experts per MoE layer
    shared_intermediate: int   # total width of the shared FFN (n_shared x expert width)
    dense_intermediate: int    # width of the dense FFN on non-MoE layers
    moe_layers: FrozenSet[int]
    num_hidden_layers: int

    @property
    def dense_layers(self) -> FrozenSet[int]:
        return frozenset(range(self.num_hidden_layers)) - self.moe_layers

    @property
    def all_moe(self) -> bool:
        return len(self.moe_layers) == self.num_hidden_layers

    def is_moe_layer(self, layer: int) -> bool:
        return layer in self.moe_layers


def moe_layout(config: Dict) -> Optional[MoELayout]:
    """None for a dense model."""
    c = decoder_config(config)
    n = num_experts(c)
    if not n:
        return None
    layers = int(c["num_hidden_layers"])
    model_type = str(c.get("model_type", ""))
    dense_ffn = int(c.get("intermediate_size_mlp", c.get("intermediate_size", c.get("ffn_dim", 0))))
    expert_ffn = int(c.get("moe_intermediate_size", c.get("intermediate_size", c.get("ffn_dim", 0))))
    if "moe_layers" in c and isinstance(c["moe_layers"], list):
        moe = frozenset(int(i) for i in c["moe_layers"])
    else:
        step = int(c.get("interleave_moe_layer_step", 1) or 1)
        first_dense = int(c.get("first_k_dense_replace", 0) or 0)
        freq = int(c.get("moe_layer_freq", 1) or 1)
        moe = frozenset(i for i in range(layers)
                        if i >= first_dense and (i + 1) % step == 0 and (i - first_dense) % freq == 0)
    if "n_shared_experts" in c:
        shared = int(c["n_shared_experts"] or 0)
    elif model_type.startswith("llama4"):
        shared = 1                       # Llama4TextMoe.shared_expert, not a config field
    else:
        shared = 0
    return MoELayout(
        num_experts=n,
        top_k=max(1, min(int(c.get("num_experts_per_tok", 1) or 1), n)),
        moe_intermediate=expert_ffn,
        shared_experts=shared,
        shared_intermediate=shared * expert_ffn,
        dense_intermediate=dense_ffn,
        moe_layers=moe,
        num_hidden_layers=layers,
    )


def is_moe(config: Dict) -> bool:
    return moe_layout(config) is not None


def architecture_name(config: Dict) -> str:
    """The ``profiler/models/<name>.yaml`` catalog for this config."""
    c = decoder_config(config)
    mt = str(c.get("model_type", ""))
    if mt == "llama4_text":
        return "llama4"
    return mt
