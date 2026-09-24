"""KV-cache layout of a model on a rank (plan sec. 4C).

Answers, from the model config and the parallelism, what a rank actually
stores per token: which KV shards it holds, how many bytes they take, and how
many ranks hold the same bytes. The one rule that matters:

  a rank's KV per token is computed from the KV heads *resident on that rank*,
  never by dividing a model-wide KV size by the accelerator count.

With grouped-query attention vLLM shards KV heads over TP and, when TP exceeds
the number of KV heads, replicates each head on tp / kv_heads ranks. The old
`2 * kv_dim * layers * dtype // num_npus` is exact only while tp divides
kv_heads; at TP64 on a 4-KV-head model it understates per-rank KV 16x (and so
overstates KV capacity 16x). Multi-head latent attention (DeepSeek-V3,
Kimi-K2) stores one compressed latent per token per layer that is not
sharded by heads at all.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVLayout:
    kind: str                 # "gqa" or "mla"
    num_layers: int
    kv_heads: int             # gqa
    head_dim: int             # gqa
    latent_dim: int           # mla: kv_lora_rank + qk_rope_head_dim
    kv_bytes: int             # bytes per KV element

    @staticmethod
    def from_config(config: dict, kv_bytes: int) -> "KVLayout":
        n_head = config["num_attention_heads"]
        head_dim = config.get("head_dim", config["hidden_size"] // n_head)
        if "kv_lora_rank" in config:
            latent = int(config["kv_lora_rank"]) + int(config.get("qk_rope_head_dim", 0))
            return KVLayout("mla", config["num_hidden_layers"], 0, head_dim, latent, kv_bytes)
        return KVLayout("gqa", config["num_hidden_layers"],
                        config.get("num_key_value_heads", n_head), head_dim, 0, kv_bytes)

    def kv_heads_per_rank(self, tp: int) -> int:
        """KV heads resident on one TP rank (vLLM replicates when tp > kv_heads)."""
        return max(self.kv_heads // max(tp, 1), 1)

    def replication(self, tp: int) -> int:
        """Ranks holding the same KV bytes (1 when tp <= kv_heads)."""
        if self.kind == "mla":
            return max(tp, 1)
        return max(max(tp, 1) // max(self.kv_heads, 1), 1)

    def bytes_per_token_per_layer(self, tp: int) -> int:
        if self.kind == "mla":
            return self.latent_dim * self.kv_bytes
        return 2 * self.kv_heads_per_rank(tp) * self.head_dim * self.kv_bytes

    def bytes_per_token(self, tokens: int, tp: int, pp: int = 1) -> int:
        """KV bytes `tokens` occupy on one rank, its pipeline stage's layers only.

        Integer arithmetic ordered so that, when tp divides kv_heads, the
        result equals the historical `2 * kv_dim * tokens * layers * bytes //
        (tp * pp)` to the byte.
        """
        return self.bytes_per_token_per_layer(tp) * tokens * self.num_layers // max(pp, 1)

    def model_bytes_per_token(self) -> int:
        """Whole-model KV per token (all layers, all shards, no replication)."""
        if self.kind == "mla":
            return self.latent_dim * self.kv_bytes * self.num_layers
        return 2 * self.kv_heads * self.head_dim * self.kv_bytes * self.num_layers
