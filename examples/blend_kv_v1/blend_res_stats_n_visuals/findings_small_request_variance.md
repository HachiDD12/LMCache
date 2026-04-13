# CacheBlend Speedup Variance Analysis: The Small-Request Fragmentation Effect

## 1. Observation

The 3D surface plot of **speedup vs. cache hit ratio vs. total prompt tokens** (with IQR-based outlier removal at 3.0x) reveals three distinct behaviors:

1. **Hit ratio drives speedup on average.** The fitted plane `speedup = 1.479 * hit_ratio + 1.09e-6 * total_tokens + 0.427` (R^2 = 0.610, n = 532 after outlier removal) shows that a 10pp increase in cache hit ratio yields ~0.15x additional speedup, while the total_tokens coefficient is effectively zero. The breakeven point (speedup = 1.0x) sits at approximately 39% cache hit ratio.

2. **Small requests exhibit high speedup variance.** At total_tokens < 5,000, the point cloud fans vertically: some requests achieve 4--5x speedup while others drop below 0.5x. The coefficient of variation (CV) of the profile-time ratio in the <2k bucket is 0.113 -- seemingly low, but the absolute spread (0.465 to 0.756) is wide relative to the mean and the cloud is visually dispersed in the 3D plot because TTFT-level speedup (which includes the saved compute component) amplifies these differences. By contrast, the 10--30k bucket shows a CV of 0.211 but its points cluster much more tightly around the fitted surface because the per-step overhead is a smaller fraction of the total work.

3. **Large requests converge onto the fitted surface.** Above 10,000 tokens, almost all points lie within a narrow band around the plane, indicating that blend mode is highly predictable for long-context workloads.

## 2. Root Cause: Scheduler-Induced Chunked-Prefill Fragmentation

### 2.1 The Two Populations

Analysis of the per-request profiler records reveals a **bimodal distribution** in the number of `retrieve_layer` operations per logical request:

| retrieve_layer ops | Requests | Interpretation |
|---|---|---|
| **4** | 322 | 1 prefill step x 4 TP workers |
| **36** | 99 | 9 prefill steps x 4 TP workers |

vLLM's chunked-prefill scheduler (`vllm/v1/core/scheduler.py`) decides how many tokens to process per step based on `max_num_batched_tokens` and the current batch composition (decode requests in flight). A request that arrives during a decode-heavy window gets interleaved across many steps, each handling only a fraction of its prompt.

### 2.2 Fragmentation Disproportionately Hits Small Requests

Cross-tabulating op count against token bucket reveals that **small requests are 3.4x more likely to land in the 36-op (fragmented) bucket**:

| Token bucket | 4-op (%) | 36-op (%) |
|---|---|---|
| < 2,000 | 55% | **45%** |
| 2--5,000 | 79% | 21% |
| 5--10,000 | 90% | 10% |
| 10--30,000 | 71% | 29% |
| > 30,000 | 87% | 13% |

Small prompts are more susceptible because they fill a smaller portion of `max_num_batched_tokens`, leaving room for the scheduler to pack decode steps alongside them and defer their remaining tokens to subsequent batches.

### 2.3 Per-Step Overhead is Fixed-Cost, Not Proportional to Tokens

Each call to `retrieve_layer` in `cache_engine.py:598` triggers a chain of fixed-cost operations:

1. **Token database lookup** (`cache_engine.py:616`, phase `retrieve_layer.token_db`): walks the token sequence to identify cached chunks. Cost: ~1.4 ms per call regardless of how many tokens are in the step.

2. **Per-layer storage retrieval** (`cache_engine.py:680`, phase `retrieve_layer.storage_get_L{i}`): fetches each layer's KV cache from the CPU storage backend. Cost: ~0.3 ms x 36 layers = ~10.8 ms.

3. **Per-layer GPU onload** (`cache_engine.py:694`, phase `retrieve_layer.gpu_onload_L{i}`): copies KV data into vLLM's paged KV buffer via the GPU connector (`gpu_connector.py:936--968`). Cost: ~0.4 ms x 36 layers = ~14.4 ms.

Measured totals per `retrieve_layer` call:

| Mode | Median | P10 | P90 | Mean |
|---|---|---|---|---|
| **Blend** | 27.2 ms | 22.3 ms | 42.2 ms | 30.8 ms |
| **No-blend** | 16.7 ms | 12.1 ms | 28.1 ms | 18.0 ms |
| **Ratio** | **1.63x** | 1.84x | 1.50x | **1.71x** |

The ~14 ms difference between blend and no-blend comes from the blender's gap-aware recomputation: `blender.py:59--139` (`process_qkv`) selects the top `recomp_ratio * total_len` (i.e., 15%) most-divergent token positions per layer and recomputes their KV state (lines 96--122), plus the additional GPU work of comparing fresh vs. cached keys (`diff_k` at line 91--93) and sorting gap positions (lines 101--122).

### 2.4 Why Fixed Cost Dominates Small Requests

For a 1,000-token request fragmented across 9 steps:
- Each step processes ~111 tokens.
- Each step incurs ~27 ms of blend retrieve overhead.
- Total overhead: 9 steps x 27 ms x 4 TP workers' worth of records = **~972 ms** of aggregate profile time.
- Baseline cost (no-blend, no fragmentation): ~114 ms TTFT for straight prefill of 1,000 tokens.

The retrieve overhead alone is **8.5x** the entire no-blend TTFT. Even though cache hits save the GPU compute on 90%+ of the prompt tokens, the savings (~100 ms of prefill compute) are dwarfed by the 972 ms of per-step LMCache overhead.

For a 50,000-token request fragmented across the same 9 steps:
- Each step processes ~5,556 tokens.
- Same 27 ms per-step overhead, so total overhead is unchanged at ~972 ms.
- Baseline cost: ~5,000 ms TTFT for straight prefill.
- The 972 ms overhead is **19%** of baseline, and cache hits save ~4,500 ms of compute.
- Net speedup: ~2x, closely matching the fitted surface.

**The core asymmetry**: the per-step overhead is invariant to the number of tokens in the step, but the compute savings scale linearly with token count. Small requests sit below the crossover where overhead > savings.

## 3. Why the Variance (Not Just the Mean) is High for Small Requests

Two identically sized, identically cached small requests can land on opposite sides of the speedup distribution based on a **single scheduler decision**: whether they get 1 prefill step (4 ops) or 9 steps (36 ops).

- **1 step (4 ops)**: total LMCache overhead ~4 x 27 ms = ~108 ms. If baseline is 114 ms and cache saves 100 ms of compute, TTFT ~ 108 + 14 = 122 ms. Speedup ~ 114/122 = 0.93x (slight regression, but close to parity).
- **9 steps (36 ops)**: total overhead ~36 x 27 ms = ~972 ms. TTFT ~ 972 + 14 = 986 ms. Speedup ~ 114/986 = 0.12x (severe regression).

Same request, same cache state, 8x difference in speedup -- driven entirely by the scheduler's batching decision at the time of arrival. This is why the 3D plot shows a **fan-shaped spread** at low total_tokens rather than a tight cluster.

For large requests, both 1-step and 9-step scenarios produce per-step token counts large enough that the fixed overhead is a small fraction of per-step compute, so the fan closes.

## 4. Implications and Potential Improvements

### 4.1 Approach A: Minimum Token-Per-Step Threshold for LMCache Retrieval

**Concept**: In `vllm_v1_adapter.py:start_load_kv()`, before entering the retrieve path, check the number of tokens allocated to this scheduler step for the current request. If below a threshold (e.g., 256 tokens), skip the LMCache retrieve and let vLLM do normal prefill for those tokens.

**Trade-off**: Loses cache benefit for micro-batched steps, but those steps are exactly where the overhead exceeds the benefit. The threshold can be tuned per-deployment based on the per-call overhead profile (27 ms blend, 17 ms no-blend).

**Implementation site**: The per-request loop in `vllm_v1_adapter.py` (currently line 821). Before calling `retrieve_layer()` or `retrieve()`, check `len(request.token_ids) < MIN_TOKENS_FOR_CACHE_LOOKUP` and skip if true.

### 4.2 Approach B: Prompt-Length-Aware Blend Bypass

**Concept**: For requests with `total_tokens < N` (e.g., N = 2,000), bypass the blend path entirely at the server level or the adapter level. Small prompts are cheap to recompute from scratch, and the data shows blend is net-negative for most sub-2k requests in the fragmented bucket.

**Trade-off**: Loses non-prefix cache reuse for small prompts. But sub-2k prompts represent only ~23% of the request population (96 / 421), and their cache savings are modest in absolute terms (saving ~100 ms of compute vs. paying ~100--1,000 ms of overhead depending on fragmentation).

**Implementation site**: `blend_server_ttft.py`, immediately before invoking the blend path. Check `len(prompt_tokens)` and route to the no-blend path (`messages_to_prompt_no_blend`) when below threshold.

### 4.3 Approach C: Coalesced Retrieval Across Prefill Steps

**Concept**: Instead of calling `retrieve_layer` once per scheduler step, accumulate the token ranges across steps and issue a single coalesced retrieval when the prompt is fully prefilled. This eliminates the 9x overhead multiplier from fragmentation.

**Trade-off**: Increases latency for the coalesced retrieval (larger single-shot fetch), and requires significant changes to the adapter's streaming model. The layerwise pipeline (`vllm_v1_adapter.py` `wait_for_layer_load`) currently interleaves layer loads with the model's forward pass; coalescing would break this pipelining.

**Implementation complexity**: High -- requires a new "deferred retrieval" mode in the adapter that batches token ranges across steps and triggers the retrieval at the last step. The `cache_engine.py` `retrieve_layer` generator would need to accept a list of (start, end) ranges rather than a contiguous token sequence.

### 4.4 Assessment: Necessary Evil vs. Fixable Overhead

The fragmentation effect is **not** a fundamental limitation of CacheBlend's cache reuse model -- it is a consequence of the interaction between vLLM's chunked-prefill scheduler and LMCache's per-step invocation pattern. The per-step overhead itself is not inherently excessive; it is the **multiplication** by the fragmentation factor that makes it problematic for small requests.

**Approach A** (minimum token-per-step threshold) is the lowest-risk, highest-impact fix: it directly addresses the pathological case (micro-batched steps) without changing the scheduler or the cache engine's retrieval model. It can be implemented in ~10 lines at the adapter level and tuned via an environment variable.

**Approach B** (prompt-length bypass) is the safest fallback: it avoids the problem entirely for the population that suffers most, at the cost of giving up cache reuse for small prompts.

**Approach C** (coalesced retrieval) is the most principled long-term solution but requires a non-trivial redesign of the adapter--cache-engine interface. It would eliminate the fragmentation effect for all request sizes, including future workloads where scheduler decisions may change.

The current behavior can be characterized as a **necessary evil of the streaming chunked-prefill architecture**: the scheduler's freedom to interleave prefill and decode steps is what makes vLLM's TTFT competitive for large requests, and LMCache's per-step overhead is the price of hooking into that streaming model. For production deployments where the request length distribution is known (as in this thesis's Django trace), Approach A or B provides an effective, low-cost mitigation.

## 5. Summary of Key Evidence

| Metric | Value | Source |
|---|---|---|
| Plane fit (no outliers) | `speedup = 1.479 * hit + 1.1e-6 * tokens + 0.427` | `compare_lmcache_runs.py` |
| R^2 with / without outliers | 0.193 / 0.610 | 3D plot stdout |
| Breakeven hit ratio | ~39% | Plane intercept at speedup = 1.0 |
| Blend per-call overhead | 27.2 ms median (1.71x vs. no-blend) | Profiler JSONL |
| Fragmented (36-op) fraction | 99 / 421 = 23.5% of requests | Profiler JSONL |
| Small-request fragmentation rate | 45% of <2k tokens (vs. 13% of >30k) | Profiler JSONL |
| Worst-case overhead multiplier | 9 steps x 4 TP x 27 ms = 972 ms | Profiler JSONL |
| Baseline TTFT for 1k-token request | ~114 ms | No-blend run .out |
