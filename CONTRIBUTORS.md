# Contributors

LLMServingSim is developed and maintained by the
[CASYS](https://casys.kaist.ac.kr) research group at KAIST. It would not be
what it is without the people who have given their time, insight, and code to
the project. This page is our way of saying **thank you**.

## Core Team — CASYS, KAIST

- **Jaehong Cho** ([@JaehongCS20](https://github.com/JaehongCS20))
- **Hyunmin Choi** ([@hyuenmin-choi](https://github.com/hyuenmin-choi))
- **Guseul Heo**
- **Minsu Kim**
- **Jongse Park** — faculty advisor

## Community Contributors

We are especially grateful to contributors from outside CASYS who have
volunteered their effort to make LLMServingSim better for everyone. 🙏

- **[@horser1](https://github.com/horser1)**
  - Per-dimension link settings + collective dimension sync ([#33](https://github.com/casys-kaist/LLMServingSim/pull/33))
  - Prefix-cache / radix-tree fixes ([#35](https://github.com/casys-kaist/LLMServingSim/pull/35))
  - Per-instance runtime config overrides ([#37](https://github.com/casys-kaist/LLMServingSim/pull/37))
  - Non-DP multi-instance collective scoping ([#39](https://github.com/casys-kaist/LLMServingSim/pull/39))
  - Run-isolated ASTRA-Sim input paths ([#43](https://github.com/casys-kaist/LLMServingSim/pull/43))
  - KV eviction/reload accounting ([#48](https://github.com/casys-kaist/LLMServingSim/pull/48))
- **[@Veilwalker](https://github.com/Veilwalker)**
  - Avoid duplicate prefix-cache hit accounting under chunked prefill ([#49](https://github.com/casys-kaist/LLMServingSim/pull/49))
- **[@zsxh1990](https://github.com/zsxh1990)**
  - Docs for per-instance runtime overrides ([#38](https://github.com/casys-kaist/LLMServingSim/pull/38))
  - Generalized PIM latency model for arbitrary architectures ([#45](https://github.com/casys-kaist/LLMServingSim/pull/45))
- **[@shermanjlim](https://github.com/shermanjlim)**
  - `avail_size()` overestimation and `storage_cache_evicted_req` fixes ([#29](https://github.com/casys-kaist/LLMServingSim/pull/29))
- **[@gleb-kun](https://github.com/gleb-kun)**
  - Fix missing return value in the profiler's argument parser ([#22](https://github.com/casys-kaist/LLMServingSim/pull/22))

If you have contributed and are not listed here, or you'd like your entry
updated, please open a pull request or
[reach out](https://llmservingsim.ai/contact) — we want everyone's work to be
recognized.

## Acknowledgments

The base layerwise-profile methodology in `profiler/` is adapted from
[@waneon](https://github.com/waneon). LLMServingSim builds on
[ASTRA-Sim](https://github.com/astra-sim/astra-sim) and
[Chakra](https://github.com/mlcommons/chakra), and was inspired in part by
[vLLM](https://github.com/vllm-project/vllm) and
[SGLang](https://github.com/sgl-project/sglang).

---

Interested in contributing? See the
[contributor guide](https://llmservingsim.ai/docs/contributor/welcome).
