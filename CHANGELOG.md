# Changelog

All notable changes to this project are documented in this file.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

## [Unreleased]

### Added
- Public Docusaurus 3 documentation site at
  [llmservingsim.ai](https://llmservingsim.ai), built from `docs/` and
  deployed via GitHub Actions Pages. Replaces the old `docs/index.html`
  placeholder and shifts long-form content (CLI flag tables, dataset
  schema, profiler walkthroughs, validation plots, etc.) off the README.
  The repo's `README.md` is now a minimal front door (About / Getting
  Started / Publications / Citation) that links to the website. The
  `README and docs split` policy is documented in `AGENTS.md` /
  `CLAUDE.md`.
- Local search on the docs site via
  `@easyops-cn/docusaurus-search-local`. Indexes all `/docs/*` and
  top-level page routes (Contact, Changelog) at build time. Access via
  the navbar input or Ctrl/Cmd-K once the production build runs (dev
  mode does not generate the index — `pnpm build && pnpm serve` to test
  locally).
- Module helper `full_cluster_kv_bytes_per_token(model, fp, kv_cache_dtype)`
  in `serving/core/memory_model.py`. Computes full-cluster KV bytes per
  token directly from a HuggingFace-style config, avoiding the per-rank
  floor-division roundoff in `MemoryModel.get_kv(1) * num_npus`. Used by
  `__main__.py` to size shared prefix pools at startup, before any
  `MemoryModel` exists.

### Changed
- Trace-level PP modeling write-up overhauled in
  `docs/docs/simulator/parallelism-mechanics.md` — explicitly describes
  the Chakra layer split + `COMM_SEND` / `COMM_RECV` between stages,
  with a stage-split figure. Replaces the previous
  "scheduling-only / lower bound" framing which underdescribed what
  the simulator actually models.
- `--expert-routing-policy` default documented as `BALANCED`
  everywhere (expert-parallel example, troubleshooting,
  trace-generation, `AGENTS.md`) — the earlier docs referenced a
  non-existent `COPY` default. `CUSTOM` listed under both request- and
  expert-routing options; `--enable-block-copy` decoupled from routing
  policy in the docs.
- `LOAD` request-routing scoring (`waiting * 4 + running`) documented
  in the multi-instance example. Policy lists reformatted into bullets
  across affected pages.
- `MemoryModel.get_weight` now divides the transformer-block weight by
  `pp_size` (heaviest-rank conservative bound:
  `embedding + n_layer//pp × per_block + final_layernorm + lm_head`).
  Required adding a `pp_size` parameter to `MemoryModel.__init__`
  (threaded through from `Scheduler`). PP=1 behavior unchanged — fix
  only affects future PP > 1 runs (no current cluster config exercises
  PP > 1).
- `MemoryModel.apply_kv_cache_events` now drains the second-tier event
  queue for CXL prefix storage and CPU + prefix-sharing modes (in
  addition to the previously-handled CPU non-sharing case). The CPU
  non-sharing branch keeps bridging events into `cpu_used`; the other
  paths just drain the queue (no accounting impact — pool memory usage
  is already tracked via `total_size * kv_size` in `total_memory_usage`).
  Prevents unbounded growth of the event queue over the simulation
  lifetime.

### Fixed
- Chunked prefill double-counted prefix-cache hits. In
  `schedule_with_prefix`, `chunk_size = original_input - num_computed_tokens`
  already excludes prefix-cached tokens (because `num_computed_tokens`
  is bumped to `prefix_cache_hit` on the first `prefix_match`). The
  scheduler then accumulated `hit_len += prefix_hit` on top of that,
  and `_build_batch_ctx` (trace_generator.py) subtracted the prefix
  hit a second time — collapsing `total_len` to 1 for any prefill
  chunk with prefix caching on. Dense-layer latency and TP collective
  sizing were both being looked up at 1 token instead of `chunk_size`.
  Fix: drop the second subtraction; sub-batch interleaving and the
  `Batch.hit_len` field were removed as part of the cleanup.
- `_make_sub_batch` (sub-batch interleaving) was not chunked-prefill
  aware: it used `req.is_init` (later chunks have `is_init=False` and
  would be misclassified as decode), `req.input` (full prompt length
  instead of this step's chunk), and `prefill_k_list=0` (ignoring KV
  already produced by prior chunks). It also failed to reset
  `prefill_q_list` / `prefill_k_list` / `decode_k_list` between the
  two sub-batches, leaking batch1 state into batch2. Now reads
  `batch.scheduled_tokens` (set by the scheduler), keys off
  `req.is_prefill()`, and uses `req.num_computed_tokens` for KV
  already in cache.
- `MemoryModel.evict_prefix_cache` over-evicted the second-tier
  (CPU/CXL) cache by `num_npus`× because `space_needed` was computed
  with the per-rank `self._bytes_per_token` while each second-tier
  token represents full-cluster bytes (`per-rank × num_npus`). Now
  uses the cache's own `kv_size` for the per-token bytes (per-rank for
  NPU, full-cluster for second-tier). TP=1 unaffected; TP>1 prefix
  hit rates were collapsing as the storage tier was over-evicted on
  every spill.
- `MemoryModel.evict_prefix_cache` early-return guard required *both*
  `not enable_prefix_caching` AND `bytes <= 0`. Changed to `or` — the
  intent is to return early if either condition holds.
- NPU→CPU offload alloc/free in `scheduler.py` used per-rank bytes
  while prefix-cache events tracked full-cluster bytes
  (`get_kv(tlen) * num_npus`). At TP>1 `cpu_used` drifted between
  the two paths. Offload paths now scale by `num_npus` to match the
  existing CPU accounting convention so `cpu_used` is consistently
  full-cluster bytes per instance.
- `MemoryModel.storage_cache_evicted_req` called
  `npu_prefix_cache.inc_lock_ref(new_last_node)` where
  `new_last_node` belongs to the **second-tier** prefix tree.
  Walking up parents from a foreign-tree node never reaches
  `npu_prefix_cache.root_node` and ultimately dereferences `None`,
  crashing the simulator when evicting from NPU to CPU/CXL storage
  with prefix caching on. Now uses the correct tree (PR #25).
- `MemoryModel.avail_size` returned `RadixCache.avail_size() *
  self._bytes_per_token`, but `RadixCache.avail_size()` already
  returns bytes (`capacity - total_memory_usage()`). The extra
  multiplication produced a meaninglessly large value, making
  scheduler decisions based on it (e.g.
  `avail_size + evictable_size`) under-conservative even at TP=1.
  Now passes the byte value through unchanged (PR #25).
- Hardcoded `131072` bytes-per-token (Llama-3.1-8B bf16-specific)
  in five sites in `serving/__main__.py` (prefix-pool creation +
  CPU/CXL usage display) replaced with model-aware values: pools
  now build via `full_cluster_kv_bytes_per_token` at startup, and
  display lines use each `RadixCache`'s own `kv_size`. Fixes
  utilization readout for non-Llama-3.1-8B models (Qwen3 family,
  etc.).
- Tuple-unpacking crash in the CXL + prefix-sharing display path:
  `for i, cxl_id, cxl_pool in enumerate(prefix_pools):` would
  raise `ValueError: not enough values to unpack` because
  `enumerate()` yields 2-tuples. Replaced with proper 2-element
  unpacking.
- Refreshed validation baselines + website plots after the
  chunked-prefill + prefix-cache fix. Means / P99s now slightly
  over-predict vLLM instead of slightly under-predicting (the
  prior under-prediction came from dense layers being looked up
  at 1 token whenever a prefill chunk had any prefix-cache hit).
  All three bundled configurations still land within ~2.5% on
  TTFT / TPOT / latency means.

### Security
- Bump `fast-uri` to ≥3.1.2 (CVE-2026-6321 path traversal via
  percent-encoded dot segments + CVE-2026-6322 host confusion via
  percent-encoded authority delimiters, both rated High). Pinned in
  `pnpm.overrides` since the package ships as a transitive
  Docusaurus dependency.
- Bump `@babel/plugin-transform-modules-systemjs` to ≥7.29.4
  (GHSA-fv7c-fp4j-7gwp, CVE-2026-44728, High). Arbitrary code
  generation when compiling malicious input; affects 7.12.0–7.29.3.
  We shipped 7.29.0 via `@docusaurus/preset-classic`. Pinned in
  `pnpm.overrides`.
- Bump `serialize-javascript` to ≥7.0.5 (Dependabot, XSS via
  deferred function / regexp serialization). Pulled in transitively
  by `copy-webpack-plugin` and `css-minimizer-webpack-plugin` in
  Docusaurus 3.10.
- Bump `uuid` to ≥14.0.0 (Dependabot, missing buffer bounds check
  in v3/v5/v6 when `buf` is provided). Replaces both transitive
  8.3.2 (via `sockjs`) and 11.1.1.

## [v1.1.0] - 2026-04-26

### Added
- New vLLM-based layerwise profiler (`profiler/`) replacing the old `llm_profile/`
  module. Uses vLLM's built-in `layerwise_profile()` via a worker extension class to
  capture per-layer CUDA kernel timings from real vLLM execution paths. Architecture
  is dispatched by the HF config's `model_type` against YAML catalogs under
  `profiler/models/`, and each run emits a per-category CSV bundle
  (`dense.csv`, `per_sequence.csv`, `attention.csv`, and `moe.csv` for MoE) under
  `perf/<hw>/<model>/<variant>/tp<N>/`, with latencies in microseconds.
  The base layerwise-profile methodology — driving a real vLLM engine via a worker
  extension class and emulating TP=N on a single GPU by sharding `hf_overrides` — is
  adapted from [@waneon](https://github.com/waneon).
- Unified 4D attention profiling (`attention.csv`) replacing the earlier
  prefill/decode-separated scheme with a single table over
  `prefill_chunk × kv_prefill × n_decode × kv_decode` that matches what
  vLLM's chunked-prefill scheduler actually produces each step.
  Geometric axes with `ATTENTION_CHUNK_FACTOR` / `ATTENTION_KV_FACTOR`
  (default 2.0 = doubling) tune density against profile time
- Skew profiling + 5-axis alpha fit for heterogeneous-decode attention
  (`profiler/core/skew.py`, `fit_alpha.py`). The sweep fires bimodal
  decode batches and measures `(t_mean, t_max, t_skew)` per case; `fit_alpha`
  then groups rows by a 5-axis key `pc | n_label | skew_rate_label |
  kv_big_label | kp_label` and runs weighted least-squares per cell.
  At query time the simulator blends two uniform-attention lookups via the
  fitted alpha to recover the FlashAttention tile-padding / SM-imbalance
  penalty the uniform grid can't see (`serving/core/trace_generator.py`
  `_lookup_attention_with_skew` / `_skew_alpha`). Axis ablation on the
  widened ~13k-sample dataset picked the 5-axis scheme over the earlier
  3-axis fit (test p50/p90 ≈ 2.7% / 14.8% vs 3.5% / 16.4% on TP=1)
- Data-derived bucket axes for the skew fit. `n` and `kp` buckets are one
  per unique profiled value (+ `kp=0` sentinel + overflow); `kv_big` uses
  log-4x bins adapted to the observed max; `skew_rate` is a fixed
  normalised [0, 1] scheme; `pc` is keyed raw. Derived axes are written
  to `meta.yaml::skew_fit.bucket_axes` and the simulator reads them from
  there, so widening `MAX_NUM_SEQS` or `ATTENTION_MAX_KV` lights up finer
  resolution without any simulator code change
- Per-axis skew density knobs: `SKEW_N_FACTOR` / `SKEW_PC_FACTOR` /
  `SKEW_KP_FACTOR` / `SKEW_KVS_FACTOR` (CLI: `--skew-*-factor`, default
  2.0 = doubling). Crank higher to coarsen a given axis and cut profile
  time; effective values land in `meta.yaml::skew_profile.factors`
- Per-TP `skew_fit.csv` file spills the full per-bucket alpha table out
  of `meta.yaml` so the latter stays readable (~100 lines vs ~3100 lines
  for Qwen3-32B at 2 TPs). `meta.yaml::skew_fit.per_tp[tp].bucket_table`
  points at `tp<N>/skew_fit.csv`; the simulator hydrates it back into
  `alpha_by_bucket` on `_load_perf_db()`
- Compact `attention_grid` / `skew_profile` grid specs in `meta.yaml`
  (e.g. `"0, 16-2048 x2"` instead of the full value list)
- RTXPRO6000 (NVIDIA RTX PRO 6000 Blackwell) hardware support: 96 GB, 1597 GB/s,
  600W TDP
- DP+EP (Data Parallel + Expert Parallel) support with ASTRA-Sim ALLTOALL synchronization
  via `involved_dim` dimension scoping. Instances with the same `dp_group` share a single
  ASTRA-Sim process; the 2D topology `[tp_size, dp_group_size]` enables per-dimension
  collective routing (ALLREDUCE on TP dim, ALLTOALL on DP dim)
- Wave synchronization for DP groups: Python-side `dp_pending` barrier ensures all instances
  schedule before trace generation. ALLTOALL `comm_size` synchronized to `max(total_len)`
  across the group. Dummy batches keep idle instances participating in ALLTOALL sync
- `single_node_moe_dp_ep_instance.json` cluster config for MoE with DP+EP
  (2 instances, TP=1, EP=2, same DP group)
- Agentic session support for closed-loop workloads (e.g., SWE-bench). The new JSONL
  format uses `sub_requests` arrays with `tool_duration_ns` to model dependency chains
  where each LLM call waits for the previous one to complete plus tool execution time.
  The router dynamically releases sub-requests as their predecessors finish, enabling
  accurate simulation of multi-step agentic workflows
- `--num-reqs` CLI argument (replaces `--num-req`), default changed from 100 to 0
  (load all entries from dataset). For agentic datasets, counts sessions not sub-requests
- Example SWE-bench agentic dataset (`workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl`)
- Qwen3-32B and Qwen3-30B-A3B-Instruct-2507 model configs with explicit `head_dim`
  support for models where `head_dim != hidden_size // num_attention_heads`
- FP8 KV cache simulation support (`--kv-cache-dtype fp8`): selects `profile_fp8.csv`
  for compute latency lookup and halves KV cache memory usage in the memory model
- FP8 KV cache profiling support (`kv_cache_dtype: "fp8"` in receipts, outputs
  `profile_fp8.csv`)
- Chunked prefill support (enabled by default, matching vLLM v1) with
  `--long-prefill-token-threshold` for per-request token cap per step
  (chunked prefill core by [@HyunsuYEE](https://github.com/HyunsuYEE))
- Chunked prefill compatible with prefix caching (RadixAttention)
- Prefix cache lock tracking (`_prefix_locked`) to prevent incorrect eviction during
  multi-chunk prefill
- Non-Docker vLLM installer (`scripts/install-vllm.sh`) using `uv` with
  precompiled vLLM 0.19.0 wheels ([@junwha](https://github.com/junwha))
- End-to-end vLLM benchmark + simulator validation suite (`bench/`,
  invoked as `python -m bench {run,validate}`). `bench run` replays a
  workload through a real vLLM `AsyncLLM` engine with `output_toks`
  pinned via `SamplingParams(min_tokens=N, max_tokens=N, ignore_eos=True)`
  so results are bit-for-bit comparable to the simulator's view of the
  same dataset. A custom `vllm.v1.metrics.loggers.StatLoggerBase` writes
  per-tick scheduler / iteration stats; `RequestStateStats` from
  `vllm.v1.metrics.stats` lands in `requests.jsonl`. `bench validate`
  loads a finished run plus the simulator's `sim.csv` / `sim.log` and
  emits throughput, running/waiting, and TTFT/TPOT/latency-CDF plots
  plus a numeric diff% summary
- Workload generators (`workloads/generators/`, invoked as
  `python -m workloads.generators sharegpt …`). Multi-turn ShareGPT
  parser with running context accumulation; default source
  `shibing624/sharegpt_gpt4`. Runs in tokenizer-only mode by default
  (output IDs from the assistant turn) or with `--use-vllm` to drive an
  offline batched `vllm.LLM` for free-generated outputs at maximum
  throughput. Optional `--fix-len` (random fixed-length tokens) and
  `--pulse` (bursty arrivals) modes
- Per-model invocation templates under `workloads/examples/`
  (`gen-llama-3.1-8b.sh`, `gen-qwen3-30b-a3b.sh`, `gen-qwen3-32b.sh`)
- Module READMEs for `bench/`, `scripts/` (top-level wrappers for the
  vLLM and simulator container launchers, the bare-metal vLLM installer,
  and the ASTRA-Sim build)
- Rich-backed logger shared between simulator, profiler, and bench
  (`serving/core/logger.py`, `profiler/core/logger.py`,
  `bench/core/logger.py`).
  Keeps the original `[HH:MM:SS.mmm] [Component] [node=X,inst=Y] LEVEL msg`
  line shape via a custom ``_RichSimHandler`` (public API unchanged —
  ``configure_logger`` / ``get_logger`` / the ``ComponentLoggerAdapter``
  still work for every existing call site) and adds:
  - ``.success()`` (green ✓ at INFO) and ``.summary()`` (verbatim,
    no prefix) on the adapter, plus module-level ``print_banner()`` /
    ``print_input_config()`` / ``print_markup()`` / ``print_rule()``
    and ``stage(title)`` / ``progress(label, total)`` context managers
    mirroring the profiler's helpers.
  - Rich theme + ``soft_wrap=True`` so colour renders in interactive
    terminals, long lines stay on one logical row, and redirected
    files (``> out.log``, ``nohup`` …) get clean plain-text logs
    with no stray ANSI escape bytes. ``FORCE_COLOR=1`` still forces
    colour when an IDE terminal doesn't self-identify as a TTY.
  - Banner / logo / input-config / simulation-results blocks in
    `serving/__main__.py` migrated to the new helpers (with `bench/__main__.py`
    using the same banner / stage / progress conventions); heartbeat status tree
    (``├─`` / ``└─``) now builds each line as a string and emits
    via Rich markup for consistent colouring.
  - ``RadixCache.format_prefix_info()``,
    ``Scheduler.print_result()``, and
    ``PowerModel.print_power_summary()`` rewritten around the new
    helpers. ``serving/utils.py`` loses its ANSI colour
    wrappers (``cyan`` / ``bold`` / ``ANSI_*`` / …) and the logo /
    input-config renderers now live in ``logger.py``
- READMEs for `configs/model/`, `configs/pim/`, `workloads/`, `serving/`
- `.gitignore` entries for AI agent cache files (`.claude/`, `.cursor/`, `.copilot/`,
  `.codex/`, `.aider*`, `.continue/`)

### Fixed
- Skew sweep feasibility filter used strict `n_reqs >= max_num_seqs` and
  dropped every `n = MSQ` case (including the pure-decode corner the
  attention sweep was already allowing). Relaxed to `>` to match
  attention and unlock pure `n = MSQ` shots. Mixed-regime `n = MSQ`
  (requires MSQ+1 requests) still filtered; profile with `MAX_NUM_SEQS`
  one above runtime MSQ to cover that corner too
- Missing `prefix_match` call on non-chunked prefill path: prefix cache hits were not
  detected for full prefill requests, preventing prefix caching benefits when chunked
  prefill was disabled ([@junwha](https://github.com/junwha))
- Typo in timer reference in legacy Mixtral profiler model
  ([@junwha](https://github.com/junwha))
- Prompt throughput now includes prefix cache hit tokens. Previously only actually
  computed prefill tokens were counted, making throughput appear lower than vLLM's
  reported prompt throughput when prefix caching was active
- Prefix cache `is_init` never cleared for full prefix cache hits, causing
  `total_requested_tokens` to inflate on every decode step and `lock_ref` leaks
- Prefix cache `lock_prefix` not called for full prefix hits, causing memory leaks
  at simulation end
- MoE expert latency aggregated both EP ranks onto one GPU (2x overestimate);
  now each GPU uses only its own rank's tokens and activated experts
- MoE weight calculation in `memory_model.py` now uses `ep_size` (not `tp_size`)
  for expert weight sharding
- Status print timing: only prints on start NPU to avoid transient "0 running" states
- `system.json` collective implementations now match topology dimensions (2 entries
  for 2D topologies) — previously 1 entry caused ASTRA-Sim to create only 1 dimension
- DP group termination: instances wait for all DP members to finish before marking done
- `argparse` `allow_abbrev=False` to prevent silent prefix matching of wrong arguments
- Add missing `return parser.parse_args()` in legacy profiler layers/main.py
  (reported and fixed by [@junwha](https://github.com/junwha), [@gleb-kun](https://github.com/gleb-kun))

### Changed
- `--fp` flag replaced with `--dtype` (vLLM-style: `float16`, `bfloat16`, `float32`,
  `int8`)
- `--gen` flag replaced with `--skip-prefill` for clarity
- `--request-routing-policy` default changed from `RR` to `LOAD` (vLLM-style weighted
  least-loaded). Requests are now routed in real-time based on current system state
  instead of upfront assignment
- `--expert-routing-policy` `FAST` renamed to `COPY` for clarity (enables block copy)
- Cluster config: `npu_num`/`npu_group` replaced with `tp_size`/`pp_size`/`ep_size`/`dp_group`.
  Partial configs supported (e.g., `num_npus=4, tp_size=2` infers `pp_size=2`).
  TP and EP share the same GPU set; DP via multiple instances with same `dp_group`
- MoE modeling: per-EP-rank latency lookup (`key_0=local_tokens, key_1=activated_experts`),
  even expert-to-rank partitioning, ASTRA-Sim ALLTOALL with `involved_dim` for cross-DP sync
- MoE `calculate_sizes`: uses `moe_intermediate_size` (per-expert FFN dim) separate from
  `intermediate_size` (dense FFN dim)
- `calculate_sizes` parameter renamed: `tp` → `parallel` (generic for TP or EP)
- Trace `comm_type` now supports dimension scoping: `ALLREDUCE:1,0`, `ALLTOALL:0,1`
- Network topology for DP groups: `npus_count: [tp_size, dp_group_size]` with per-dimension
  collective implementations in `system.json`
- Removed analytical ALLTOALL workaround functions (`_inflate_comm_size`,
  `_ring_alltoall_time_ns`, `_bw_gb_to_bpns`) — replaced by native ASTRA-Sim ALLTOALL
- `link_bw`/`link_latency` removed from `TraceCtx` and `generate_trace` (no longer needed
  for analytical fallback)
- Latency lookup extrapolates beyond profiled range instead of clamping for improved
  accuracy on large batch sizes
- Profiler rewritten from PyTorch Profiler + scikit-learn predictor to direct vLLM
  `layerwise_profile()` approach. Architecture yamls live in `profiler/models/`
  keyed on the HF config's `model_type`; CLI flags match vLLM (`--dtype`,
  `--kv-cache-dtype`, `--max-num-batched-tokens`, `--max-num-seqs`, `--tp`,
  `--variant`). Docker pinned to vLLM v0.19.0 (`vllm/vllm-openai:v0.19.0` or
  `v0.19.0-cu130` for CUDA 13.x)
- Old profiler preserved under `profiler/v0/` for reference
- Layer names unified between profiler and simulator: `qkv_projection`, `o_projection`,
  `ffn1`, `ffn2`, `attention`, `layernorm` (old names removed)
- `memory_model.py` updated to use explicit `head_dim` and `q_dim`/`kv_dim` for correct
  tensor size computation on models like Qwen3
- `trace_generator.py` rewritten with composable helpers (`TraceCtx`, `BatchCtx`,
  `_emit_layer`, `_emit_pre_attn_layers`, `_emit_post_attn_layers`) and unified profile
  CSV lookup with 2D bilinear interpolation
- Sampler output location changed to `REMOTE` (was on `lm_head`) to match Chakra
  converter's MEM_STORE node placement
- Removed `--enable-attn-prediction` flag (scikit-learn predictor replaced by direct
  profiled latency lookup)
- Cluster configs updated to RTXPRO6000 hardware specs
- `AGENTS.md` expanded with full repo structure, simulation flow, trace format
  documentation, and additional pitfalls
- `--max-batch` renamed to `--max-num-seqs` (default: 128, matching vLLM);
  now limits total running requests across inflight batches
- `--enable-chunked-prefill` now enabled by default (matching vLLM v1);
  use `--no-enable-chunked-prefill` to disable
- `--enable-prefix-caching` now enabled by default (matching vLLM v1);
  use `--no-enable-prefix-caching` to disable
- Scheduler rewritten to use vLLM-style token-budget-based allocation for both
  chunked and non-chunked prefill paths (`schedule_base`, `schedule_with_prefix`)
- KV cache block allocation uses vLLM-style cumulative ceiling division
- Radix tree `cache_unfinished_req` now uses `num_computed_tokens` instead of
  `req.input`, enabling correct incremental caching across chunks
- Prefix cache memory accounting changed to free-before-allocate order
- Hash-to-length map in `memory_model.py` changed from `{hash: tlen}` to
  `{hash: [tlen, refcount]}` to handle duplicate block hashes
- All `Request` attributes now properly initialized in `__init__`; removed
  `getattr` fallbacks throughout scheduler and radix tree
- Directory restructuring:
  - `cluster_config/` → `configs/cluster/`
  - `model_config/` → `configs/model/`
  - `pim_config/` → `configs/pim/`
  - `dataset/` → `workloads/` (the directory holds ShareGPT-style
    request workloads consumed by the simulator and bench)
  - `output/` → `outputs/`
  - `script/` → `scripts/`
  - `llm_profile/` → `profiler/legacy_profiler/` (later moved to `profiler/v0/`)
- Top-level package layout finalized as Python-style sibling modules:
  - `inference_serving/` → `serving/` with internals under `serving/core/`
    (every `.py` previously at the package root now lives one directory
    deeper); entrypoint `main.py` becomes `serving/__main__.py` and is
    invoked as `python -m serving …`.
  - `llm_profiler/` → `profiler/` (collapses the duplicated
    `llm_profiler/profiler/` package layer) with internals under
    `profiler/core/` and `profiler/core/hooks/`.
  - `bench/` added with the same shape (`bench/core/`).
  - `workloads/` ships the ShareGPT generator under
    `workloads/generators/sharegpt.py` (invoked as
    `python -m workloads.generators sharegpt …`) with per-model
    invocation templates under `workloads/examples/`. The package
    deliberately avoids the name `datasets/` so the HuggingFace
    `datasets` library imports cleanly.
  - Module-specific shell scripts live at the module home (e.g.
    `profiler/profile.sh`, `bench/bench.sh`, `serving/run.sh`); only
    cross-cutting environment / build helpers stay in `scripts/`
    (`docker-vllm.sh`, `docker-sim.sh`, `install-vllm.sh`, `compile.sh`).
- Evaluation configs moved from `config/` to `configs/` subdirectories within each
  figure folder
- `run.sh` updated with reorganized examples and commented out unavailable MoE config

### Removed
- `internal/` directory (debug docs and scheduler tests moved or removed)
- `scripts/` batch experiment scripts (superseded by `run.sh` examples)
- `evaluation/` directory (preserved on `ispass26-artifact` branch)
- `--enable-attn-prediction` flag and scikit-learn attention predictor
- `--fp` flag (replaced by `--dtype`)
- `--gen` flag (replaced by `--skip-prefill`)
- `--expert-routing-policy FAST` (renamed to `COPY`)
- `serving/attn_utils.py` (stale scikit-learn attention feature helper)
- `npu_num`/`npu_group` config fields (replaced by `tp_size`/`pp_size`/`ep_size`)
- `--num-req` flag (replaced by `--num-reqs`)
- Analytical ALLTOALL workaround functions (`_inflate_comm_size`, `_ring_alltoall_time_ns`)
- `evaluation/` directory (preserved on `ispass26-artifact` branch)

---

## [v1.0.0] - 2026-02-25

### Added
- Multi-instance simulation with configurable request routing policies (Round Robin, Random, Custom)
- Prefill/Decode (P/D) disaggregation support across instances
- Mixture of Experts (MoE) support with expert parallelism, expert offloading, and configurable
  routing policies (Round Robin, Random, Fast, Custom)
- Prefix caching using RadixAttention (based on SGLang), with support for second-tier prefix cache
  pooling across CPU and CXL memory (`--enable-prefix-caching`, `--enable-prefix-sharing`)
- Sub-batch interleaving to overlap prefill and decode phases within an iteration
  (`--enable-sub-batch-interleaving`)
- Attention latency predictor using scikit-learn for real-time per-request estimation
  (`--enable-attn-prediction`)
- Power and energy modeling per node covering NPU, CPU, DRAM, interconnect, NIC, and storage
- CXL memory expansion support with configurable bandwidth and latency
- Enhanced PIM (Processing-In-Memory) model with per-device INI configuration (`configs/pim/`)
- Cluster-level configuration system (`configs/cluster/*.json`) that consolidates all hardware,
  topology, and placement parameters into a single file
- Per-layer weight, KV cache, and expert placement rules in cluster config
- Additional latency metrics: ITL (Inter-Token Latency) and p99 for TTFT, TPOT, ITL
- Hardware performance profiles for TPU-v6e-1
- Batch experiment scripts for systematic evaluation (`scripts/`)
- Artifact evaluation scripts and reference results (`evaluation/`)
- `llm_profile` integrated as a local module with support for MoE models and power profiling

### Changed
- All hardware and topology parameters are now specified via `cluster_config` JSON files;
  per-invocation hardware arguments (`--model_name`, `--hardware`, `--npu_num`, etc.) are removed
- Command-line argument style changed from underscore to hyphen (e.g., `--cluster-config`,
  `--num-req`, `--block-size`)
- Dataset format changed from `.tsv` to `.jsonl`
- Build process consolidated into `./compile.sh` and `./docker.sh`
- Performance model directory relocated from `perf_model/` to `llm_profile/perf_models/`
- `serving/` modules renamed for clarity:
  - `control.py` → `controller.py`
  - `generate_graph.py` → `graph_generator.py`
  - `generate_trace.py` → `trace_generator.py`
  - `config_generator.py` → `config_builder.py`
  - `pim.py` → `pim_model.py`
- Fix incorrect `evict_size` accumulation

### Removed
- `trace_test/` directory (superseded by `evaluation/` scripts)
- Direct per-invocation hardware arguments (`--model_name`, `--hardware`, `--npu_num`,
  `--npu_group`, `--npu_mem`, `--remote_bw`, `--link_bw`)

---

## [v0.2.1] - 2025-07-18

### Added
- `llm_profile` module with PyTorch Profiler for GPU layer and attention latency measurement
- Llama-3.1-8B-Instruct model support (replaces GPT-3 6.7B as the default model)
- Hugging Face model configuration support for easy addition of new models

### Changed
- Function names standardized to snake_case (e.g., `createNetworkConfig` → `create_network_config`,
  `calculateSizes` → `calculate_sizes`)
- Model configuration files updated to Llama-3.1-8B-Instruct format

### Fixed
- Collective operation stall caused by unresolved dependencies in the ASTRA-Sim workload graph
- Network dimension calculation for full pipeline parallelism (`npus_per_dim` formula corrected)

---

## [v0.2.0] - 2025-06-04

### Changed
- ASTRA-Sim submodule updated to latest version (branch `v0.2.0`)
- Chakra updated to latest version
- Network configuration format changed from JSON to YAML
- `local_bw` and `remote_bw` parameters replaced with `link_latency`
- Conda environment dependencies updated and simplified

---

## [v0.1.0] - 2025-01-03

### Added
- GPU performance model based on TensorRT-LLM profiling (replaces NPU simulator)
- Auto config generator for network and memory configurations
- New parameters: `--hardware`, `--local_bw`, `--remote_bw`, `--link_bw`, `--fp`
- Additional metrics: `queuing_delay`, TTFT, TPOT
- Verbose logging option for detailed execution output

### Changed
- ASTRA-Sim submodule branch updated from `artifact` to `v0.1.0`
- Output format changed from TSV to CSV

### Removed
- Polymath and codelets_src submodules (NPU simulator components replaced by performance model)

---

## [artifact] - 2024-06-23

### Added
- Initial project release as IISWC 2024 artifact: "LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale"
- NPU simulator-based co-simulation infrastructure (ASTRA-Sim + Polymath + codelets_src)
- Evaluation scripts and benchmark results
- Conda environment configuration (`environment.yml`)
