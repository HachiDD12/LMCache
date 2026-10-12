# Deterministic admission of parallel-sampling children (the sibling gate)

Status: implemented and tested on CPU (real vLLM Scheduler, 0.10.1.1 and 0.10.2 sources) and on one local GPU (real vLLM 0.10.1.1 engine, `facebook/opt-125m`). Not yet run on the cluster: TP=4, LMCache, Devstral, compiled vLLM 0.10.2.

Code: `sibling_gate.py` (scheduler subclass and helpers), wired into `blend_server_ttft.py` and `blend_server_ttft_no_lmcache.py` behind `--sibling-gate`. Tests: `tests/test_sibling_gate.py` (CPU), `tests/e2e_sibling_gate_smoke.py` (one GPU, about 1 min).

---

## 1. Problem

Agentless's repair stage sends a greedy n=1 request and then an n=9 request. In 98 of the 100 cases the n=9 prompt is byte-identical to the greedy one.

**How vLLM handles n=9**
- `AsyncLLM.add_request` (`vllm/v1/engine/async_llm.py`) fans the request out into 9 children `"<idx>_<parent>"` (`ParentRequest.get_child_info`).
- Each child goes to the engine process as its own ZMQ `ADD` message (`engine_core.add_request_async`).
- `EngineCoreProc._process_input_queue` (`vllm/v1/engine/core.py`) blocks until child 0 arrives, drains only what is already queued, then steps: `schedule()`, then `execute_model` (blocking).
- So **k**, the number of children admitted in the first step, is a thread-timing race. The first step loads, blends or prefills every co-admitted child one after another, and TTFT (first token of any child) is the duration of that step.

**What the race did in run 930run1**

| | k = 1 | 2 | 3 | 4 | 5 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| new blend run | 77 | 6 | 6 | 2 | | 1 | 1 | 7 |
| April no-blend baseline | 47 | 20 | 10 | 4 | 5 | 1 | 3 | 10 |

- 52 of the 100 n=9 requests got a different k in the two runs.
- Old code: baseline TTFT relative to one child is 1.00, 1.64, 2.34, 3.05, 3.54, … 6.03 for k = 1…9.
- New code (sibling dedup): co-admitted *hits* cost about 1 ms each, but co-admitted *misses* each do a full prefill. Example: `778002650f1a80fe` (1,825 tokens, cold) had 8 cold children in step 1, giving 4.63 s against 0.27 s for one child.
- The speedup (baseline TTFT / blend TTFT) inherits whichever run drew the worse k. The median n=9 speedup was 0.69× when the baseline had k=1 (47 requests) and 4.21× when it had k ∈ {8, 9} (13 requests).

The goal is a k that is fixed by the request, not by thread timing, in every configuration compared:
- (a) vanilla vLLM, prefix caching off and on;
- (b) LMCache whole prompt (`--blend-granularity none`);
- (c) blend (`--blend-granularity chunk`).

## 2. Policy: leader-first (default `--sibling-gate leader`)

**Rule.** Admit exactly one child, the leader (index 0). Hold its siblings in the waiting queue until the leader's prefill has executed (it has produced an output token), or the leader has finished or been aborted, or a safety valve fires. Then release them; vLLM schedules them normally.

**Consequences.**
- k = 1 in every configuration and every run.
- TTFT is one child's cost: one prefill, one LMCache load or one blend. That is the quantity the comparison is about.
- The siblings reuse the leader's work:
  - LMCache configs: the leader has stored the prompt, so siblings hit and sibling dedup copies the KV (§4.4);
  - vLLM with prefix caching: siblings hit the leader's cached blocks;
  - vanilla with prefix caching off: siblings prefill in full, after the first token.
- This is what an optimized server does anyway: prefill a shared prompt once and fork the samples from it.

### 2.1 The meeting's suggestion ("lock add_request so all 9 arrive together"), evaluated

**A lock in the API process changes nothing.**
- The fan-out loop already sends the 9 messages back to back from one coroutine, and the sequential replay has no competing `add_request`.
- The race is between the *engine's* input thread and its main loop, in another process. A lock cannot reach there: the main loop wakes on child 0 and steps with whatever is queued.
- In the local GPU runs the 9 children reached the engine within 2–12 ms (median 2–6 ms). That spread decides k.

**One message for all children is not possible in vLLM 0.10.2.**
- `EngineCoreRequestType.ADD` carries a single `EngineCoreRequest`; the input thread decodes it with `MsgpackDecoder(EngineCoreRequest)`.
- vLLM's own TODO above `AsyncLLM.generate` notes that several requests cannot yet go to the core process in one message.
- Atomic arrival therefore needs either a vLLM patch (a batch `ADD`) or the engine holding children until all have arrived.

**The engine-side version is implemented as `--sibling-gate all`, so it can be measured.** It is a barrier: hold every child until all n have arrived (timeout 5 s), then release them together.

**`all` is deterministic, but it fixes k at the worst value for misses.**
- Hits are cheap per child: an LMCache hit costs 1 token of budget, and a prefix-cache hit costs the last partial block. All 9 fit.
- Misses each need a full prefill and compete for the 16,384-token budget. k becomes min(9, budget-limited count), for example 8 or 9 for a 1.8k prompt. TTFT ≈ k × prefill, which makes the `778002650f1a80fe` case permanent.
- On the local GPU, `all` gave an n=9 TTFT of 1.3–2.9× the greedy TTFT of the same prompt (prefix caching off). Leader-first gave 1.0–1.1× (§5.2).

**Rejected alternatives**
- **Gate inside LMCache** (`get_num_new_matched_tokens` returning `None`).
  - It covers only configs (b) and (c), not vanilla vLLM.
  - 0.10.1.1 has no `None` check (`num_new_local_computed_tokens + None` would raise).
  - 0.10.2 moves a `None` request to `skipped_waiting_requests`, which reverses the order of skipped requests when they are put back.
- **Leader-first in the server** (send child 0 as n=1, then an n=8 request after its first token).
  - It works with the stock scheduler.
  - But it changes the workload the engine sees: two requests and two id families instead of `N_<id>` children of one parent, which the profiler join keys and analysis scripts rely on.
  - It also ties the timing to an API round trip.

## 3. Mechanism (`sibling_gate.py`)

**Client side (API process, both servers).**
- `make_extra_args(parent_id, n)` returns `{"sibling_gate": {"parent": <request_id>, "n": n, "leader": 0}}` for n > 1, and `None` for n = 1. It is passed as `SamplingParams(extra_args=...)`.
  - Every child inherits it: `ParentRequest._get_child_sampling_params` shallow-copies the params.
  - msgspec carries it to the engine (`SamplingParams` is a msgspec struct with `extra_args: Optional[dict]`).
  - vLLM itself reads only `extra_args["kv_transfer_params"]` (`Request.__init__`), and so does LMCache (`vllm_v1_adapter.py`). No collision.
- `engine_kwargs(mode)` returns `{"scheduler_cls": "sibling_gate.SiblingGateScheduler"}` and sets `SIBLING_GATE_MODE`. It also puts this directory on `sys.path` and `PYTHONPATH`, so the engine process can import the class.
  - vLLM keeps a user-supplied class: `EngineArgs._set_default_args_v1` replaces only the V0 default, and `EngineCore.__init__` resolves it with `resolve_obj_by_qualname`.
  - `async_scheduling` is the only exception: there vLLM forces `AsyncScheduler`.
  - `--sibling-gate off` passes nothing, which is the stock scheduler, exactly as before.
- `emit_child_lines(...)` prints the server-side `[CHILD]`, `[CHILD_SUMMARY]` and `[CHILD_OUTPUT]` lines (§7).

**Engine side: `SiblingGateScheduler(vllm.v1.core.sched.scheduler.Scheduler)`.** It overrides two methods.

`add_request(request)`:
- calls `super().add_request` first;
- then `_sg_parse`: reads `sampling_params.extra_args["sibling_gate"]` and requires `request_id == f"{idx}_{parent}"` with 0 ≤ idx < n;
- registers the child in a `_Group` keyed by the parent id.

Requests without the key, or with an id that doesn't match, are never gated. A mismatch logs one `[SIBLING_GATE] WARNING`.

`schedule()`:
1. Increments the step counter. If no group is live (every n = 1 request), it returns `super().schedule()` directly.
2. `_sg_hold_ids`: for each group still holding, `_sg_release_reason` checks, in order:
   - the leader has finished or left `self.requests` → `leader_finished`;
   - `leader.num_output_tokens > 0` → `leader_prefill_done`. The `update_from_output` of the step that computed the last prompt token has run, so the leader's KV is in its blocks and, with LMCache, stored;
   - `stall_steps ≥ SIBLING_GATE_MAX_STALL_STEPS` (64): consecutive steps in which the leader got no tokens while siblings waited → `valve_stall`;
   - `hold_steps ≥ SIBLING_GATE_MAX_HOLD_STEPS` (4096) → `valve_hold`.

   Otherwise every sibling that has arrived but not been admitted is held. In `all` mode the leader is held as well, until all n have arrived (`all_arrived`), a member has finished (`member_finished`), or `SIBLING_GATE_BARRIER_TIMEOUT_S` has passed (`valve_barrier_timeout`).
3. Removes the held requests from `self.waiting` (`RequestQueue.remove_requests`), calls `super().schedule()`, and in a `finally` puts them back with `_sg_reinsert`:
   - FCFS deque: each held request goes back right after its nearest surviving predecessor from the pre-step snapshot, and before requests preempted in this step. The order of everything else is exactly what vLLM produced.
   - Priority queue: `add_request`.
4. `_sg_account`: records which children were admitted (`scheduled_new_reqs`) or resumed (`resumed_from_preemption`), updates the stall and hold counters, logs, and garbage-collects groups with no live member.

Held requests are back in `self.waiting` between steps, so `finish_requests` (abort), `get_request_counts` and stats see them as in stock vLLM. While siblings are held the leader is running, so steps are never empty because of the gate, except in valve cases. Empty steps are harmless: `execute_model` with zero tokens, and the busy loop keeps stepping because `has_requests()` is true.

**Modes** (`--sibling-gate`): `leader` (default), `all` (barrier), `observe` (no gating, same log lines; scheduling identical to stock, tested), `off` (stock class, no extra_args).

## 4. Interactions

1. **Token budget** (`max_num_batched_tokens`, 16,384).
   - The leader gets the whole budget in its first step.
   - Released siblings compete for the budget in the step after the first token: hits cost about 1 token, misses a full prefill. Misses are spread over steps deterministically by prompt length (patterns in §5.1).
   - None of this is in TTFT.
   - *Side effect:* the leader's 2nd token waits for that sibling step, and n=9 generation time grows by about one step.
2. **Chunked prefill.**
   - A leader longer than the budget prefills over several steps with every sibling held (`event=hold` lines with `leader=<computed>/<prompt>`).
   - Siblings are released in the step after the last chunk.
   - Without the gate, a sibling could even join the leader's *last* chunk step and delay its first token. The stock pattern `0:[0]|3:[1]|…` for a 1,900-token prompt with budget 512 is such a case.
3. **Preemption.**
   - A leader preempted mid-prefill goes back to `waiting` (`num_computed_tokens = 0`, no output token), so it stays unreleased and its siblings stay held. It resumes as a normal waiting request.
   - The stall valve counts only steps in which the leader got no tokens, and a re-admission resets it.
   - A preempt/resume livelock is bounded by `MAX_HOLD_STEPS`.
   - Tested with real KV-block pressure (`test_leader_preempted_during_prefill`).
4. **Sibling dedup (LMCache, `LMCACHE_BLEND_SIBLING_DEDUP`).**
   - *Hot prompt:* the leader hits and becomes the donor (`is_kv_donor`). At release it is RUNNING (decoding), so all 8 siblings copy its KV with `_copy_paged_kv`: 1 blend + 8 copies, as with k=9 before, but only the blend is in TTFT.
   - *Cold prompt:* the leader prefills and stores. Siblings hit; the first becomes the donor (1 blend) and 7 copy.
   - Donor eviction (finished, or not RUNNING) is unchanged.
5. **Prefix caching (vanilla, prefix caching on).**
   - vLLM V1 caches full blocks *at allocation* (`KVCacheManager.allocate_slots` → `cache_blocks(request, num_computed + num_new)`). A sibling scheduled after child 0 *in the same step* therefore already hits child 0's blocks, even for a cold prompt.
   - So the race costs least here (k=9 ≈ 1 prefill + 8 partial blocks); the gate removes the remaining variation.
6. **TP.** The scheduler runs once, in the engine process. All 4 workers receive the same `SchedulerOutput`, so nothing changes per rank.
7. **LMCache lookups (`Reqid:` lines).** Held siblings never reach `get_num_new_matched_tokens`, so their `Reqid:` lines appear only at release, after the leader's first token. `build_master.first_step_size()` should then return k=1.
8. **Unsupported setups.**
   - `async_scheduling`: refused at construction (vLLM would swap the class anyway).
   - Pipeline-parallel batch queue: still correct, but the release lags.
   - Stock vLLM 0.10.x cannot queue n>1 children under `policy=priority` at all: equal heap keys make `heapq` compare `Request` objects (`TypeError`, reproduced). The servers use FCFS.

## 5. Verification

### 5.1 CPU, real vLLM Scheduler (`tests/test_sibling_gate.py`, 17 tests)

**What is real:**
- `vllm.v1.core.sched.scheduler.Scheduler`: budget, chunked prefill, KV-block allocation, preemption, prefix caching, FCFS and priority queues;
- the client-to-engine path of a child: `SamplingParams(extra_args)` → `ParentRequest.get_child_info` → `EngineCoreRequest` → `MsgpackEncoder`/`MsgpackDecoder` → `Request.from_engine_core_request`.

**What is simulated:**
- the busy loop: arrivals happen between steps, and a step runs while `has_requests()` is true;
- the model: a request whose prefill completes samples one token;
- LMCache: a scheduler-side fake connector. A prompt is cached once some request has finished prefilling it, and a cached prompt is a full hit (external tokens not charged to the budget). Lookups are recorded.

**Results: 17/17 pass on vLLM 0.10.1.1 (installed) and on the vLLM 0.10.2 sources** (tag `v0.10.2`).
- The 0.10.2 run imported `vllm0102/vllm/v1/core/sched/scheduler.py` from the tag.
- Test-only shims: a `sitecustomize` that blocks the local sklearn/numpy ABI mismatch and stubs the compiled extensions, plus `SIBLING_GATE_TEST_CANNED_MODELINFO=1` to skip the model-architecture probe subprocess.
- All 63 rows of the `--report` pattern table are identical across the two versions.

| test | what it shows |
|---|---|
| `test_extra_args_survive_fanout_and_msgspec`, `test_bad_ids_are_not_gated` | transport; n=1 carries nothing; only `<idx>_<parent>` ids are gated |
| `test_n1_scheduling_unchanged` | 16 random n=1 workloads (prefix caching off/on, memory pressure with preemption, >500 steps): per-step output identical to the stock scheduler |
| `test_observe_mode_equals_stock_with_groups` | `observe` = stock, with n=9 groups and n=1 traffic, all arrival orders |
| `test_stock_scheduler_shows_the_race` | without the gate, k=9 (all queued) vs k=1 (child 0 alone) for the same request |
| `test_leader_mode_deterministic_everywhere` | configs a0/a1/lmc × hot/cold/long × arrival orders: k=1, identical pattern, siblings admitted exactly one step after the leader's first token, no sibling lookup before release, the gate's own log equals the observed pattern |
| `test_leader_mode_expected_patterns` | concrete patterns (table below) |
| `test_late_siblings_still_k1` | siblings arriving after the leader decodes: k still 1 |
| `test_all_mode_barrier_deterministic` | `all`: k=9 for hits, budget-limited for misses, identical across arrivals; barrier spin logged once |
| `test_leader_preempted_during_prefill` | leader preempted mid-prefill by memory pressure; siblings held until its first token |
| `test_abort_leader_releases_siblings`, `test_abort_whole_group_cleans_up`, `test_leader_finishes_before_release` | abort and finish paths; group cleanup |
| `test_stall_valve_and_empty_steps` | no leader: 3 harmless empty steps, then `valve_stall` |
| `test_waiting_order_restored`, `test_priority_policy_supported`, `test_bad_mode_fails_loudly` | queue order, priority path, loud failure |

**Patterns** (n=9). Rows: a0 = vanilla with prefix caching off; a1 = prefix caching on; lmc = LMCache. Kinds: hot/cold = 900-token prompt, budget 1024; long = 1,900 tokens, budget 512. Arrivals: A = all queued before the first step, B = child 0 alone, then the rest. A pattern reads `<step offset>:[admitted child indices]`.

| config | kind | stock A | stock B | **leader (A, B, C)** | all (A, B) |
|---|---|---|---|---|---|
| a0 | hot | k=2 `0:[0,1]\|1:[2]\|…` | k=1 `0:[0]\|1:[1,2]\|…` | **k=1 `0:[0]\|1:[1,2]\|2:[3]\|…\|7:[8]`** | k=2 |
| a1 | hot | k=9 `0:[0..8]` | k=1 | **k=1 `0:[0]\|1:[1..8]`** | k=9 |
| a1 | cold | k=9 | k=1 | **k=1 `0:[0]\|1:[1..8]`** | k=9 |
| lmc | hot | k=9 | k=1 | **k=1 `0:[0]\|1:[1..8]`** | k=9 |
| lmc | cold | k=2 `0:[0,1]\|1:[2..8]` | k=1 | **k=1 `0:[0]\|1:[1..8]`** | k=2 |
| lmc | long | k=1 `0:[0]\|3:[1]\|7:[2..8]` | same | **k=1 `0:[0]\|4:[1..8]`** | k=1 |

### 5.2 One GPU, real engine (`tests/e2e_sibling_gate_smoke.py`)

**Setup.** Local RTX 3070 Laptop, vLLM 0.10.1.1, `facebook/opt-125m`, budget 2048. 20 pairs of (greedy n=1, n=9 with the same prompt), replayed sequentially, with prompt lengths 150/600/1900/200/1200.

**What is real here:** `AsyncLLM` fan-out, ZMQ, `EngineCoreProc` busy loop, the class loaded by name in the engine process, and the forward pass.

| mode | prefix caching | k_first over 20 n=9 requests | distinct patterns per prompt length | n=9 TTFT / greedy TTFT (median per length, caching off) |
|---|---|---|---|---|
| observe (= stock) | off | 1:9, 2:6, 3:1, 4:2, 9:2 | 2–4 | 1.14–2.00 |
| **leader** | off | **1:20** | **1** | **1.02–1.14** |
| all | off | 2:8, 4:4, 9:8 (one k per length) | 1 | 1.33–2.85 |
| observe | on | 1:12, 2:2, 4:2, 5:1, 6:1, 8:1, 9:1 | 1–4 | – |
| **leader** | on | **1:20** | **1** | – |
| all | on | 9:20 | 1 | – |
| **leader**, budget 512 | off | **1:10** | **1** | leader prefill spans 4 steps with siblings held |

- Every request produced its first token from child 0 (`first_idx=0`), with no `[SIBLING_GATE] WARNING`.
- On a 125M model, TTFTs are 5–30 ms and dominated by overheads, so the ratios are qualitative.

### 5.3 Server wiring (fake engine)

Both servers in all 4 modes, with the engine replaced by a recorder:
- `scheduler_cls` is passed (or not, for `off`);
- `extra_args` is set for n=9 only, and never when off;
- the vanilla `--max-num-batched-tokens` reaches the engine args;
- the `[CHILD*]` lines are correct for interleaved DELTA outputs.

This check is not in the repo (it stubs `lmcache`).

### 5.4 Not verified

- LMCache connector + gate on GPU;
- TP=4;
- the compiled vLLM 0.10.2 engine;
- Devstral.

Run `tests/e2e_sibling_gate_smoke.py` on a cluster node first (§6.1) to cover the 0.10.2 engine path.

## 6. Run instructions (cluster)

The gate is **on by default** (`--sibling-gate leader`) in both servers. Run from `examples/blend_kv_v1/`, as before: the scripts already import `prefix_cache_stats` from there, and `engine_kwargs()` exports that directory to the engine process. Use the same `MOCK` trace and LMCache environment as the previous runs (`LMCACHE_CHUNK_SIZE=1024`, check layers, profiler output …). The only new flags are below.

```bash
cd <LMCache checkout>/examples/blend_kv_v1
MOCK=/home/zhuofanc/chat_requests_v037_0102_20251111_055647_django_blend_chunk.json  # as in 930run1

# (c) blend
python blend_server_ttft.py --mock --mock-file "$MOCK" --blend-granularity chunk \
    --min-chunk-size 256 --sibling-gate leader  > blend.out 2> blend.err

# (b) LMCache whole prompt ("no-blend")
python blend_server_ttft.py --mock --mock-file "$MOCK" --blend-granularity none \
    --sibling-gate leader  > noblend.out 2> noblend.err

# (a) vanilla vLLM, prefix caching off / on (same budget as the LMCache server)
python blend_server_ttft_no_lmcache.py --mock --mock-file "$MOCK" \
    --max-num-batched-tokens 16384 --sibling-gate leader  > vllm_apc0.out 2> vllm_apc0.err
python blend_server_ttft_no_lmcache.py --mock --mock-file "$MOCK" \
    --max-num-batched-tokens 16384 --enable-prefix-caching --sibling-gate leader \
    > vllm_apc1.out 2> vllm_apc1.err
```

**Optional controls** (same commands with a different flag):
- `--sibling-gate observe`: old scheduling, with exact k logged.
- `--sibling-gate all`: the barrier (meeting suggestion).
- `--sibling-gate off`: the previous setup (stock scheduler class, no `extra_args`). Only the new `[CHILD*]` lines in `.out` differ.

**Knobs** (env; the defaults are fine): `SIBLING_GATE_MAX_STALL_STEPS=64`, `SIBLING_GATE_MAX_HOLD_STEPS=4096`, `SIBLING_GATE_BARRIER_TIMEOUT_S=5`, `SIBLING_GATE_LOG=1`.

**Notes**
- The vanilla server has no `[MANIFEST]`, and its request ids are the mock index (`str(rid)`), not the content hash. Join on `mock_idx`.
- Without `--max-num-batched-tokens` it keeps vLLM's default budget of 2048.

### 6.1 Before the long runs (1 GPU, about 1 min)

```bash
python tests/e2e_sibling_gate_smoke.py leader 10 > smoke.out 2> smoke.err
grep 'event=first_admit' smoke.err | grep -o ' k=[0-9]*' | sort | uniq -c   # ->  10  k=1
grep -m1 '\[SIBLING_GATE\] init' smoke.err                                 # -> ... vllm=0.10.2
python -m pytest tests/test_sibling_gate.py -q   # CPU; needs facebook/opt-125m's HF config
```

## 7. Log lines that prove the fix (grep after the run)

The engine process writes `[SIBLING_GATE]` lines to **stderr (`.err`)**. Each is one line per sibling group per step with activity, prefixed with a millisecond timestamp (and by vLLM with `(EngineCore_0 pid=…)`). The server writes `[CHILD*]` lines to **stdout (`.out`)**. Below, `E=blend.err O=blend.out`.

```bash
# 1. The gate is loaded in the engine, in the intended mode
grep -m1 '\[SIBLING_GATE\] init' $E
#  [..] [SIBLING_GATE] init scheduler=SiblingGateScheduler mode=leader max_stall_steps=64 max_hold_steps=4096 barrier_timeout_s=5.000 queue=FCFSRequestQueue vllm=0.10.2
grep -o '"sibling_gate": "[a-z]*"' $O          # [MANIFEST] (LMCache server):  "sibling_gate": "leader"
grep -c 'SIBLING_GATE\] WARNING' $E            # 0  (else ids or extra_args were not recognised -> not gated)

# 2. k is 1 for every n=9 request, and the admitted child is the leader
grep 'event=first_admit' $E | grep -o ' k=[0-9]*' | sort | uniq -c
#     100  k=1
grep 'event=first_admit' $E | grep -o 'admitted=\[[0-9,]*\]' | sort | uniq -c
#     100 admitted=[0]

# 3. Siblings were released by the leader's prefill, not by a safety valve
grep 'event=group_admitted' $E | grep -o 'release=[a-z_]*' | sort | uniq -c
#     100 release=leader_prefill_done        (leader_finished is fine too; any valve_* -> investigate)

# 4. All siblings had arrived before the release (makes the full pattern deterministic)
grep 'event=admit ' $E | grep -c -v 'pending=0'      # 0

# 5. The first token came from child 0, and TTFT is child 0's
grep '^\[CHILD_SUMMARY\]' $O | grep -o 'first_idx=[0-9]*' | sort | uniq -c
#     100 first_idx=0

# 6. Determinism across runs of the same config: identical per-request patterns
pat() { grep 'event=group_admitted' "$1" | sed -E 's/.*parent=([^ ]+).*pattern=([^ ]+)/\1 \2/' | sort; }
diff <(pat run1.err) <(pat run2.err)                  # no output
pat $E | awk '{print $2}' | sort | uniq -c | sort -rn | head
#  LMCache configs (hits): mostly  0:[0]|1:[1,2,3,4,5,6,7,8]
```

**What the step lines look like** (local GPU run, 1,900-token prompt, budget 512):

```
[SIBLING_GATE] step=104 mode=leader parent=s002 n=9 event=first_admit k=1 admitted=[0] held=[1,2,3,4,5,6,7] queued=[] pending=1 leader=512/1900,out=0,RUNNING gate=holding
[SIBLING_GATE] step=105 mode=leader parent=s002 n=9 event=hold admitted=[] held=[1,2,3,4,5,6,7,8] queued=[] pending=0 leader=1024/1900,out=0,RUNNING gate=holding
[SIBLING_GATE] step=107 mode=leader parent=s002 n=9 event=hold admitted=[] held=[1,2,3,4,5,6,7,8] queued=[] pending=0 leader=1900/1900,out=0,RUNNING gate=holding
[SIBLING_GATE] step=108 mode=leader parent=s002 n=9 event=admit admitted=[1] held=[] queued=[2,3,4,5,6,7,8] pending=0 leader=1901/1900,out=1,RUNNING gate=released:leader_prefill_done@108
```

**Field meanings**
- `held`: held by the gate this step.
- `queued`: in waiting but not admitted (budget).
- `pending`: children not yet arrived.
- `leader=<computed incl. this step>/<prompt>,out=<output tokens>`.
- The `group_admitted` summary adds `k_first`, `steps_to_all`, `hold_steps`, `release` and `pattern`.

**Cross-checks in LMCache configs**
- `build_master.first_step_size()` (from `Reqid:` timestamps) should now give `new_k_first_step = 1` for all n=9 requests.
- The siblings' `Reqid: <i>_<parent>` lines come after child 0's first token.

## 8. Expected effect on TTFT

| config | n=9 TTFT before | n=9 TTFT with leader-first |
|---|---|---|
| (a) vanilla, prefix caching off | prefill of every co-admitted child, budget-limited (≈2× for 8k prompts, up to 9× for short ones) | one full prefill ≈ the greedy n=1 TTFT of the same prompt |
| (a) vanilla, prefix caching on | ≈ one prefill + (k−1) partial blocks (intra-step block sharing, §4.5) | leader hits the greedy predecessor's cached blocks (if not evicted): ≈ last-block compute; variance gone |
| (b) LMCache whole prompt | hot: old code k × load, new code 1 load + (k−1) × ~1 ms copies; cold: k × prefill | one load (hot) / one prefill (cold) |
| (c) blend | hot: 1 blend + (k−1) copies (new code); cold: k × prefill (`778002650f1a80fe`: 4.63 s) | one blend (hot) / one prefill (cold, ≈0.3 s for that request) |

**Implications for the analysis**
- baseline/blend now compares one child against one child in every configuration, so the n=9 vertical spread at Δ=0 (`split_by_n_delta.png`) should collapse.
- Expect n=9 to look like the k=1 cases. The actual level depends on the new baselines (R2); don't carry over the 0.69× from 930run1.
- Unchanged: n=1 requests (§5.1) and the TTFT definition (first token of any child; now always child 0).
- Changed: n=9 *generation time* rises by about one engine step. In vanilla with prefix caching off, sibling prefills now follow the first token instead of preceding it. Report it as such if total latency is used.

## 9. Risks and limits

1. **Gating needs the server's `extra_args`.** Both servers set it. Other clients (e.g. Agentless against an OpenAI-compatible vLLM server) are not gated unless they pass the same `extra_args`.
2. **It depends on vLLM's child-id format `"<idx>_<parent>"`** (0.10.1.1 and 0.10.2). If a later vLLM changes request ids, the gate logs `[SIBLING_GATE] WARNING … NOT gated` and falls back to stock behaviour. It fails safe and loud; check grep 1.
3. **It relies on vLLM's non-public scheduler interface** (vLLM prints a warning). Only `add_request()` and `schedule()` are overridden. The gate uses `self.waiting` (`remove_requests`, `add_request`, iteration, deque), `self.requests` and `Request.num_output_tokens` / `is_finished()`. Construction raises a clear error if one is missing, and also on `async_scheduling`.
4. **The full admission pattern (not just k=1) is deterministic only if all siblings arrive before the leader's first token.** The measured arrival spread is 2–12 ms, against a leader step of ≥100 ms on the cluster. Grep 4 checks it; k=1 holds regardless.
5. **FCFS bypass.** While siblings are held, requests behind them can be admitted first. That is irrelevant for the sequential replay and a deliberate policy change under concurrency.
6. **Safety valves are step-based** (`MAX_STALL_STEPS=64`, `MAX_HOLD_STEPS=4096`). They should never fire in these runs, and grep 3 shows it if they do.
7. **Pre-existing, not caused by the gate (noticed while reading, not tested on GPU).**
   - A request that is preempted and later resumed is looked up in LMCache again (`get_num_new_matched_tokens`), so vLLM counts the hit tokens as computed.
   - But `build_connector_meta` pops `load_specs` and loads only for `scheduled_new_reqs`. Resumed requests come back as cached requests and are never loaded.
   - Preemption is rare in the sequential replay, and the gate reduces concurrent prefills.

## 10. Follow-up (not done): siblings copy from a cold leader

**Today.** `build_connector_meta` registers a KV donor only when the request had an LMCache hit (`lmcache_cached_tokens > 0`). So for a cold prompt, the first released sibling blends from LMCache and the others copy from it.

**Change.**
- *Scheduler side:* when a request's last prefill step is scheduled and dedup is on, register it as donor for its prompt digest with `prompt_len` tokens even if it had 0 hits, and flag the `ReqMeta`.
- *Worker side:* on that flag, record `_kv_donor_slots[req_id] = (slot_mapping[:prompt_len], prompt_len)`. Under leader-first, receivers are scheduled at least one step later, so the copy in their `start_load_kv` reads complete KV. Positions are identical, so no RoPE fix is needed.

**Safety.**
- TP agreement holds: the decision is made on the scheduler side and every rank gets the same metadata.
- Block lifetime holds: the donor is RUNNING at release, and eviction is unchanged.

**Value.**
- It replaces one post-TTFT blend with copies, for the 2 cold n=9 prompts in this trace.
- Siblings get exact rather than blended KV.
- It does not change TTFT, and it changes GPU-side code that cannot be tested here, so it is left for later.

## 11. Files

| file | change |
|---|---|
| `examples/blend_kv_v1/sibling_gate.py` | new: `SiblingGateScheduler`, `make_extra_args`, `engine_kwargs`, `emit_child_lines` |
| `examples/blend_kv_v1/blend_server_ttft.py` | `--sibling-gate` (default `leader`); `build_llm_with_lmcache(..., sibling_gate)`; `extra_args` in the mock and TTFT handlers; per-child `[CHILD*]` lines in the mock handler; `sibling_gate` in `[MANIFEST]` |
| `examples/blend_kv_v1/blend_server_ttft_no_lmcache.py` | `--sibling-gate`, `--max-num-batched-tokens` (default unchanged: vLLM's 2048); `extra_args` in the mock and HTTP handlers; per-child lines in the mock handler |
| `examples/blend_kv_v1/tests/test_sibling_gate.py` | new: CPU tests (§5.1); `python tests/test_sibling_gate.py --report` prints the pattern table |
| `examples/blend_kv_v1/tests/e2e_sibling_gate_smoke.py` | new: one-GPU smoke test (§5.2, §6.1) |
| `examples/blend_kv_v1/SIBLING_GATING.md` | this note |
