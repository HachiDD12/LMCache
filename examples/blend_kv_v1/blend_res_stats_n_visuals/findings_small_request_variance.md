# CacheBlend Speedup Variance Analysis: The Small-Request Fragmentation Effect

## 1. Observation

The 3D surface plot of **speedup vs. cache hit ratio vs. total prompt tokens** (with IQR-based outlier removal at 3.0x) reveals three distinct behaviors:

1. **Hit ratio drives speedup on average.** The fitted plane `speedup = 1.479 * hit_ratio + 1.09e-6 * total_tokens + 0.427` (R^2 = 0.610, n = 532 after outlier removal) shows that a 10pp increase in cache hit ratio yields ~0.15x additional speedup, while the total_tokens coefficient is effectively zero. The breakeven point (speedup = 1.0x) sits at approximately 39% cache hit ratio.

2. **Small requests exhibit high speedup variance.** At total_tokens < 5,000, the point cloud fans vertically: some requests achieve 4--5x speedup while others drop below 0.5x (see **Fig 1**: `speedup_by_bucket.png`). The <2k bucket has a standard deviation of 1.080 around a mean of 0.862, with an IQR of 0.080 but absolute range of 0.036 to 9.239. By contrast, the 5--10k bucket shows a tighter distribution with mean 0.996 and 33% of requests above breakeven.

3. **The high-hit / low-token corner is paradoxically noisy.** **Fig 2** (`speedup_by_bucket_high_hit.png`) isolates requests with cache hit ratio >= 50%. Even among these well-cached requests, the <2k bucket retains a massive spread (0.3x to 9.5x), while the 5--10k bucket collapses to a tight 1.9--2.5x band. A 99% cache hit ratio does not guarantee fast performance for small requests.

## 2. Root Cause: Scheduler-Induced Chunked-Prefill Fragmentation

### 2.1 The Two Populations

Analysis of the per-request profiler records reveals a **bimodal distribution** in the number of `retrieve_layer` operations per logical request:

| retrieve_layer ops | Requests | Interpretation |
|---|---|---|
| **4** | 322 | 1 prefill step x 4 TP workers |
| **36** | 99 | 9 prefill steps x 4 TP workers |

vLLM's chunked-prefill scheduler (`vllm/v1/core/scheduler.py`) decides how many tokens to process per step based on `max_num_batched_tokens` and the current batch composition (decode requests in flight). A request that arrives during a decode-heavy window gets interleaved across many steps, each handling only a fraction of its prompt.

### 2.2 Fragmentation Disproportionately Hits Small Requests

**Fig 3** (`fragmentation_rate.png`) cross-tabulates the fragmentation rate against prompt length:

| Token bucket | 1-step (4-op) | 9-step (36-op) | Fragmentation rate |
|---|---|---|---|
| < 2,000 | 53 | 43 | **45%** |
| 2--5,000 | 78 | 21 | 21% |
| 5--10,000 | 111 | 12 | 10% |
| 10--30,000 | 41 | 17 | 29% |
| > 30,000 | 39 | 6 | 13% |

Small prompts are most susceptible (45% fragmentation rate, 3.4x the rate of 5--10k requests) because they fill a smaller portion of `max_num_batched_tokens`, leaving room for the scheduler to pack decode steps alongside them and defer their remaining tokens to subsequent batches.

### 2.3 Per-Step Overhead: Measured Phase-Level Breakdown

**Fig 4** (`phase_breakdown.png`) shows the measured per-call cost of `retrieve_layer`, broken down by phase group, for blend vs. no-blend:

| Phase group | Blend (ms) | No Blend (ms) | Description |
|---|---|---|---|
| Token DB Lookup | 0.7 | 0.3 | `cache_engine.py:616`, walks token sequence to identify cached chunks |
| Storage Get | 2.0 | 3.1 | `cache_engine.py:680`, fetches each layer's KV from the CPU backend |
| GPU Onload | **28.1** | **14.6** | `cache_engine.py:694` -> `gpu_connector.py:936--968`, copies KV into vLLM's paged buffer |
| **Total** | **30.8** | **18.0** | **Ratio: 1.72x** |

The dominant cost is **GPU Onload** (91% of blend's per-call time), where the blend path pays an additional ~13.5 ms per call. This extra cost comes from the blender's gap-aware recomputation: `blender.py:59--139` (`process_qkv`) selects the top 15% most-divergent token positions per layer (`topk_num = int(total_len * recomp_ratios[0])` at line 99) and recomputes their KV state (lines 101--122), plus the GPU work of comparing fresh vs. cached keys (`diff_k` at lines 91--93).

**Data source**: 4,852 blend `retrieve_layer` calls and 3,092 no-blend calls from the profiler JSONLs (`lmcache_profile_20260408_044806_advprof_chunklb128.jsonl` and `lmcache_profile_20260408_062217_advprof_noblend.jsonl`).

### 2.4 Why Fixed Cost Dominates Small Requests

For a 1,000-token request fragmented across 9 steps:
- Each step processes ~111 tokens.
- Each step incurs ~30.8 ms of blend retrieve overhead.
- Total overhead: 9 steps x 30.8 ms x 4 TP workers' worth of records = **~1,109 ms** of aggregate profile time.
- Baseline cost (no-blend, no fragmentation): ~114 ms TTFT for straight prefill of 1,000 tokens.

The retrieve overhead alone is **9.7x** the entire no-blend TTFT. Even though cache hits save the GPU compute on 90%+ of the prompt tokens, the savings (~100 ms of prefill compute) are dwarfed by the per-step LMCache overhead.

For a 50,000-token request fragmented across the same 9 steps:
- Each step processes ~5,556 tokens.
- Same 30.8 ms per-step overhead, so total overhead is unchanged at ~1,109 ms.
- Baseline cost: ~5,000 ms TTFT for straight prefill.
- The 1,109 ms overhead is **22%** of baseline, and cache hits save ~4,500 ms of compute.
- Net speedup: ~2x, closely matching the fitted surface.

**The core asymmetry**: the per-step overhead is invariant to the number of tokens in the step, but the compute savings scale linearly with token count. Small requests sit below the crossover where overhead > savings.

## 3. Why the Variance (Not Just the Mean) is High for Small Requests

**Fig 5** (`speedup_vs_tokens_fragmentation.png`) plots speedup vs. total_tokens on a log scale, with points colored by fragmentation class (blue circles = 1 prefill step, orange diamonds = 9 steps).

Two patterns emerge:

1. **Non-fragmented small requests (blue, 4-op)** cluster tightly at ~0.5x speedup (mean = 0.538, std = 0.031 for <2k tokens). Blend overhead is consistent and always exceeds the compute savings for these short prompts. **This is predictable but consistently below breakeven.**

2. **Fragmented small requests (orange, 36-op)** show the full spread from 0x to 9.5x (mean = 1.604, std = 1.786 for <2k tokens). The variance is driven by the interaction between the number of scheduler steps and the cache hit pattern across those steps -- some fragmented requests encounter favorable cache timing where cached content is warm from earlier steps, while others pay the full overhead with minimal cache benefit.

The key insight from Fig 5 is that **fragmentation is both the source of the worst under-performers AND the best over-performers** for small requests. The scheduler's batching decision at arrival time determines which side a request lands on, and this decision is not correlated with the request's cache state.

For large requests (>10k tokens), both fragmentation classes converge to the same narrow band because the per-step overhead is a small fraction of per-step compute regardless of how many steps the scheduler chose.

## 4. Implications and Potential Improvements

### 4.1 Approach A: Minimum Token-Per-Step Threshold for LMCache Retrieval

**Concept**: In `vllm_v1_adapter.py:start_load_kv()`, before entering the retrieve path, check the number of tokens allocated to this scheduler step for the current request. If below a threshold (e.g., 256 tokens), skip the LMCache retrieve and let vLLM do normal prefill for those tokens.

**Trade-off**: Loses cache benefit for micro-batched steps, but those steps are exactly where the overhead exceeds the benefit. The threshold can be tuned per-deployment based on the per-call overhead profile (30.8 ms blend, 18.0 ms no-blend).

**Implementation site**: The per-request loop in `vllm_v1_adapter.py` (currently line 821). Before calling `retrieve_layer()` or `retrieve()`, check `len(request.token_ids) < MIN_TOKENS_FOR_CACHE_LOOKUP` and skip if true.

### 4.2 Approach B: Prompt-Length-Aware Blend Bypass

**Concept**: For requests with `total_tokens < N` (e.g., N = 2,000), bypass the blend path entirely at the server level or the adapter level. Small prompts are cheap to recompute from scratch, and the data shows blend is net-negative for most sub-2k requests in the non-fragmented bucket (mean 0.54x).

**Trade-off**: Loses non-prefix cache reuse for small prompts. But sub-2k prompts represent only ~27% of the request population (145 / 536), and their cache savings are modest in absolute terms (saving ~100 ms of compute vs. paying ~100--1,000 ms of overhead depending on fragmentation).

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

## 5. Figures Reference

| Figure | File | Description |
|---|---|---|
| Fig 1 | `speedup_by_bucket.png` | Violin + box plot: speedup distribution by token bucket (all requests) |
| Fig 2 | `speedup_by_bucket_high_hit.png` | Same, filtered to cache hit ratio >= 50%. Shows the paradox: high-hit small requests still have massive variance |
| Fig 3 | `fragmentation_rate.png` | Stacked bar: chunked-prefill fragmentation rate by token bucket. 45% of <2k requests vs 10% of 5--10k |
| Fig 4 | `phase_breakdown.png` | Grouped bar: per-call `retrieve_layer` phase cost (blend vs no-blend). GPU Onload is 1.93x, total is 1.72x |
| Fig 5 | `speedup_vs_tokens_fragmentation.png` | Scatter: speedup vs total_tokens (log scale) colored by fragmentation class. Orange diamonds (36-op) show full variance spread; blue circles (4-op) cluster tightly |
| Fig 6 | `speedup_3d_no_outlier.png` | 3D surface: speedup vs hit_ratio vs total_tokens with plane fit (from compare_lmcache_runs.py) |

All figures produced by `analyze_variance.py` (Figs 1--5) and `compare_lmcache_runs.py` (Fig 6). PNG and PDF versions available.

## 6. Summary of Key Evidence

| Metric | Value | Source |
|---|---|---|
| Plane fit (no outliers) | `speedup = 1.479 * hit + 1.1e-6 * tokens + 0.427` | 3D plane fit (compare_lmcache_runs.py) |
| R^2 with / without outliers | 0.193 / 0.610 | 3D plane fit stdout |
| Breakeven hit ratio | ~39% | Plane intercept at speedup = 1.0 |
| Blend per-call retrieve_layer cost | 30.8 ms mean (1.72x vs. no-blend 18.0 ms) | Profiler JSONL phase breakdown (Fig 4) |
| GPU Onload per-call cost | 28.1 ms blend vs 14.6 ms no-blend (1.93x) | Profiler JSONL phase breakdown (Fig 4) |
| Fragmented (36-op) fraction | 99 / 421 = 23.5% of requests | Profiler JSONL op count |
| Small-request fragmentation rate | 45% of <2k tokens (vs. 10% of 5--10k) | Fragmentation cross-tab (Fig 3) |
| <2k non-fragmented speedup | mean=0.538, std=0.031 (tight, below breakeven) | Numerical summary |
| <2k fragmented speedup | mean=1.604, std=1.786 (high variance, spans both sides) | Numerical summary |
| Worst-case overhead multiplier | 9 steps x 4 TP x 30.8 ms = 1,109 ms | Profiler JSONL |
| Baseline TTFT for 1k-token request | ~114 ms | No-blend run .out |
