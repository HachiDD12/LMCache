# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sibling_gate.SiblingGateScheduler against the real vLLM V1
Scheduler (no GPU, no model weights; only the model's HF config is read).

What is real here:
- vllm.v1.core.sched.scheduler.Scheduler (budget, chunked prefill, KV block
  allocation, preemption, prefix caching, FCFS/priority queues);
- the client-to-engine path of an n>1 request: SamplingParams.extra_args ->
  ParentRequest.get_child_info (child ids, shallow param copy) ->
  EngineCoreRequest -> msgspec encode/decode (MsgpackEncoder/Decoder, as on
  the ZMQ socket) -> Request.from_engine_core_request.

What is simulated:
- the engine busy loop (EngineCoreProc): requests "arrive" between steps,
  step() runs only while has_requests();
- the model: every scheduled request whose prefill is complete samples one
  token (never EOS); partial prefills sample nothing (as gpu_model_runner);
- LMCache (configs b/c): a scheduler-side fake connector. A prompt is cached
  once some request has finished prefilling it (LMCache stores on the last
  prefill step), and a cached prompt is a full hit (num_tokens - 1 external
  tokens, not charged to the budget). It records lookups like "Reqid:" lines.

Run (from examples/blend_kv_v1):
    python -m pytest tests/test_sibling_gate.py -q
    python tests/test_sibling_gate.py --report    # prints the pattern table

Env:
    SIBLING_GATE_TEST_MODEL            HF model whose config is used
                                       (default facebook/opt-125m)
    SIBLING_GATE_TEST_CANNED_MODELINFO 1 = skip vLLM's model-architecture
                                       probe subprocess (only needed when the
                                       vLLM Python sources and compiled ops
                                       don't match, e.g. testing 0.10.2 sources
                                       on a 0.10.1.1 install)
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import sys
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # examples/blend_kv_v1

import torch  # noqa: E402

import vllm  # noqa: E402
from vllm.config import (CacheConfig, ModelConfig, SchedulerConfig,  # noqa: E402
                         VllmConfig)
from vllm.sampling_params import RequestOutputKind, SamplingParams  # noqa: E402
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,  # noqa: E402
                                         init_none_hash)
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.engine import EngineCoreRequest  # noqa: E402
from vllm.v1.engine.parallel_sampling import ParentRequest  # noqa: E402
from vllm.v1.kv_cache_interface import (FullAttentionSpec,  # noqa: E402
                                        KVCacheConfig, KVCacheGroupSpec)
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

import sibling_gate  # noqa: E402
from sibling_gate import SiblingGateScheduler, make_extra_args  # noqa: E402

EOS = 2
TOK = 7  # every sampled token; never EOS
BLOCK = 16
MODEL = os.environ.get("SIBLING_GATE_TEST_MODEL", "facebook/opt-125m")

# --------------------------------------------------------------------------
# vLLM construction (mirrors vllm tests/v1/core/utils.py:create_scheduler)
# --------------------------------------------------------------------------

if os.environ.get("SIBLING_GATE_TEST_CANNED_MODELINFO") == "1":
    from vllm.model_executor.models import registry as _R

    def _canned_modelinfo(self):
        fields = {f.name: False for f in dataclasses.fields(_R._ModelInfo)}
        fields.update(architecture=self.class_name,
                      is_text_generation_model=True, supports_pp=True)
        if "default_pooling_type" in fields:
            fields["default_pooling_type"] = "LAST"
        return _R._ModelInfo(**fields)

    _R._LazyRegisteredModel.inspect_model_cls = _canned_modelinfo

_MODEL_CONFIG: Optional[ModelConfig] = None


def _model_config() -> ModelConfig:
    global _MODEL_CONFIG
    if _MODEL_CONFIG is None:
        _MODEL_CONFIG = ModelConfig(model=MODEL, trust_remote_code=True,
                                    dtype="float16", seed=42,
                                    skip_tokenizer_init=True)
    return _MODEL_CONFIG


try:
    from vllm.utils import sha256 as _HASH
    init_none_hash(_HASH)
except Exception:  # older vLLM
    _HASH = hash
    init_none_hash(_HASH)
_BLOCK_HASHER = get_request_block_hasher(BLOCK, _HASH)


def build_scheduler(cls=Scheduler, *, mode: Optional[str] = None,
                    budget: int = 1024, num_blocks: int = 4000,
                    apc: bool = False, policy: str = "fcfs",
                    max_num_seqs: int = 64, env: Optional[dict] = None):
    sched_cfg = SchedulerConfig(max_num_seqs=max_num_seqs,
                                max_num_batched_tokens=budget,
                                max_model_len=2048,
                                long_prefill_token_threshold=0,
                                enable_chunked_prefill=True, policy=policy)
    cache_cfg = CacheConfig(block_size=BLOCK, gpu_memory_utilization=0.9,
                            swap_space=0, cache_dtype="auto",
                            enable_prefix_caching=apc)
    vcfg = VllmConfig(scheduler_config=sched_cfg,
                      model_config=_model_config(), cache_config=cache_cfg)
    kv_cfg = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(
            ["layer"], FullAttentionSpec(BLOCK, 1, 1, torch.float32, False))])
    cache_cfg.num_gpu_blocks = num_blocks
    saved = {}
    env = dict(env or {})
    if mode is not None:
        env["SIBLING_GATE_MODE"] = mode
    for k, v in env.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = str(v)
    try:
        return cls(vllm_config=vcfg, kv_cache_config=kv_cfg, log_stats=True,
                   structured_output_manager=StructuredOutputManager(vcfg))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------
# Requests through the real client -> engine path
# --------------------------------------------------------------------------

_ENC = MsgpackEncoder()
_DEC = MsgpackDecoder(EngineCoreRequest)
_NO_DEFAULT = EngineCoreRequest.__struct_fields__[
    :len(EngineCoreRequest.__struct_fields__)
    - len(EngineCoreRequest.__struct_defaults__)]


def _wire(request_id: str, prompt: list[int], sp: SamplingParams,
          arrival: float) -> Request:
    kw = {f: None for f in _NO_DEFAULT}
    kw.update(request_id=request_id, prompt_token_ids=list(prompt),
              sampling_params=sp, eos_token_id=EOS, arrival_time=arrival)
    ecr = EngineCoreRequest(**kw)
    decoded = _DEC.decode(_ENC.encode(ecr))  # what the engine input thread sees
    return Request.from_engine_core_request(decoded, _BLOCK_HASHER)


def make_group(parent: str, prompt: list[int], n: int = 9,
               max_tokens: int = 8, gate: bool = True,
               distinct_arrival: bool = False) -> list[Request]:
    """Children of an n>1 request, built exactly like AsyncLLM.add_request
    (which gives every child the same arrival_time; distinct_arrival=True
    works around stock vLLM's priority heap, which cannot order ties)."""
    sp = SamplingParams(n=n, temperature=0.7, max_tokens=max_tokens,
                        output_kind=RequestOutputKind.DELTA,
                        extra_args=make_extra_args(parent, n) if gate else None)
    pr = ParentRequest(parent, sp)
    arrival = time.time()
    out = []
    for idx in range(n):
        cid, csp = pr.get_child_info(idx)
        out.append(_wire(cid, prompt, csp,
                         arrival + (idx * 1e-6 if distinct_arrival else 0)))
    return out


def make_single(req_id: str, prompt: list[int], max_tokens: int = 8,
                gate: bool = True) -> Request:
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens,
                        output_kind=RequestOutputKind.DELTA,
                        extra_args=make_extra_args(req_id, 1) if gate else None)
    return _wire(req_id, prompt, sp, time.time())


def prompt_of(seed: int, length: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(10, 30000) for _ in range(length)]


# --------------------------------------------------------------------------
# Fake LMCache (scheduler side) and the engine-loop simulator
# --------------------------------------------------------------------------

class FakeLMCache:
    """Scheduler-side stand-in for the LMCache connector (see module doc)."""

    def __init__(self):
        self.stored: set[tuple[int, ...]] = set()
        self.lookups: list[tuple[int, str]] = []  # (step, req_id)
        self.step = 0

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.lookups.append((self.step, request.request_id))
        if tuple(request.prompt_token_ids) in self.stored:
            return request.num_tokens - num_computed_tokens - 1, False
        return 0, False

    def update_state_after_alloc(self, *args, **kwargs):
        pass

    def build_connector_meta(self, scheduler_output):
        return None

    def request_finished(self, request, block_ids):
        return False, None

    def take_events(self):
        return ()

    def update_connector_output(self, *args, **kwargs):
        pass

    def shutdown(self):
        pass


_MRO_FIELDS = {f.name for f in dataclasses.fields(ModelRunnerOutput)}


def _mro(req_ids: list[str], sampled: list[list[int]]) -> ModelRunnerOutput:
    kw = dict(req_ids=req_ids,
              req_id_to_index={r: i for i, r in enumerate(req_ids)},
              sampled_token_ids=sampled, logprobs=None,
              prompt_logprobs_dict={}, pooler_output=[])
    if "spec_token_ids" in _MRO_FIELDS:
        kw["spec_token_ids"] = None
    return ModelRunnerOutput(**kw)


@dataclasses.dataclass
class StepRec:
    step: int
    new: list[str]
    resumed: list[str]
    tokens: dict[str, int]
    sampled: list[str]


class Sim:
    """EngineCoreProc stand-in: arrivals between steps, step() while
    has_requests(), schedule -> fake execute -> update_from_output."""

    def __init__(self, sched, lmcache: bool = False):
        self.s = sched
        self.conn = FakeLMCache() if lmcache else None
        if self.conn is not None:
            sched.connector = self.conn
        self.recs: list[StepRec] = []
        self.first_token_step: dict[str, int] = {}
        self.admit_step: dict[str, int] = {}

    def step(self) -> StepRec:
        n = len(self.recs) + 1
        if self.conn is not None:
            self.conn.step = n
        out = self.s.schedule()
        ids = list(out.num_scheduled_tokens)
        sampled = []
        for rid in ids:
            r = self.s.requests[rid]
            done = r.num_computed_tokens >= r.num_tokens
            sampled.append([TOK] if done else [])
            if self.conn is not None and \
                    r.num_computed_tokens >= r.num_prompt_tokens:
                self.conn.stored.add(tuple(r.prompt_token_ids))
        self.s.update_from_output(out, _mro(ids, sampled))
        cached = out.scheduled_cached_reqs
        resumed = [rid for rid, res in zip(cached.req_ids,
                                           cached.resumed_from_preemption)
                   if res]
        rec = StepRec(n, [r.req_id for r in out.scheduled_new_reqs], resumed,
                      dict(out.num_scheduled_tokens),
                      [rid for rid, s in zip(ids, sampled) if s])
        for rid in rec.new:
            self.admit_step.setdefault(rid, n)
        for rid in rec.sampled:
            self.first_token_step.setdefault(rid, n)
        self.recs.append(rec)
        return rec

    def run(self, arrivals: dict[int, list[Request]], max_steps: int = 2000,
            hooks: Optional[dict] = None) -> "Sim":
        """``arrivals[k]``: requests added after k steps (0 = before the first
        schedule). ``hooks[k]``: callable(sim) run after k steps."""
        last = max(list(arrivals) + list(hooks or {}) + [0])
        k = 0
        while True:
            for r in arrivals.get(k, []):
                self.s.add_request(r)
            if hooks and k in hooks:
                hooks[k](self)
            if self.s.has_requests():
                self.step()
            elif k >= last:
                break
            k += 1
            assert k < max_steps, "simulation did not terminate"
        return self


def group_pattern(sim: Sim, parent: str, n: int = 9) -> str:
    """Admission pattern of a group relative to its first admission step,
    e.g. '0:[0]|1:[1,2,3,4,5,6,7,8]' (same format as the gate's log)."""
    steps = {i: sim.admit_step.get(f"{i}_{parent}") for i in range(n)}
    assert all(v is not None for v in steps.values()), steps
    first = min(steps.values())
    by: dict[int, list[int]] = {}
    for i, st in steps.items():
        by.setdefault(st - first, []).append(i)
    return "|".join(f"{o}:[{','.join(map(str, sorted(v)))}]"
                    for o, v in sorted(by.items()))


def k_first(sim: Sim, parent: str, n: int = 9) -> int:
    steps = [sim.admit_step[f"{i}_{parent}"] for i in range(n)]
    return steps.count(min(steps))


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def __enter__(self):
        logging.getLogger("sibling_gate").addHandler(self)
        return self

    def __exit__(self, *a):
        logging.getLogger("sibling_gate").removeHandler(self)


def _field(line: str, key: str) -> Optional[str]:
    for tok in line.split():
        if tok.startswith(key + "="):
            return tok[len(key) + 1:]
    return None


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

# Configurations at the scheduler level:
#   a0 = vanilla vLLM, prefix caching off; a1 = vanilla, prefix caching on;
#   lmc = LMCache (whole-prompt and blend look identical to the scheduler).
# Prompt kinds: hot = identical greedy n=1 predecessor already served (98 of
# the 100 n=9 requests); cold = no predecessor; long = cold and longer than
# the per-step budget (chunked prefill over several steps).
CONFIGS = {
    "a0": dict(apc=False, lmcache=False),
    "a1": dict(apc=True, lmcache=False),
    "lmc": dict(apc=False, lmcache=True),
}
BUDGET = 1024
# kind -> (per-step token budget, prompt length). "long": 4 prefill steps.
KINDS = {"hot": (BUDGET, 900), "cold": (BUDGET, 900), "long": (512, 1900)}


def scenario(cls, mode: Optional[str], config: str, kind: str, arrival: str,
             env: Optional[dict] = None):
    """Run one n=9 request (with its greedy predecessor when hot). Returns
    (sim, parent, captured gate log lines)."""
    cfg = CONFIGS[config]
    budget, length = KINDS[kind]
    sched = build_scheduler(cls, mode=mode, budget=budget, apc=cfg["apc"],
                            env=env)
    sim = Sim(sched, lmcache=cfg["lmcache"])
    prompt = prompt_of(sum(map(ord, config + kind)), length)
    parent = f"p_{config}_{kind}_{arrival}"
    kids = make_group(parent, prompt, n=9, max_tokens=6)
    pre: dict[int, list[Request]] = {}
    with LogCapture() as cap:
        if kind == "hot":
            sim.run({0: [make_single(f"g_{parent}", prompt, max_tokens=4)]})
        base = 0
        if arrival == "A":    # all 9 queued before the engine wakes
            pre = {base: kids}
        elif arrival == "B":  # engine wakes on child 0; the rest one step later
            pre = {base: kids[:1], base + 1: kids[1:]}
        elif arrival == "C":  # staggered over the leader's prefill steps
            pre = {base: kids[:1], base + 1: kids[1:3], base + 2: kids[3:6],
                   base + 3: kids[6:]}
        elif arrival == "D":  # siblings arrive late (after the leader decodes)
            pre = {base: kids[:1], base + 6: kids[1:]}
        sim.run(pre)
    return sim, parent, cap.lines


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_extra_args_survive_fanout_and_msgspec():
    """extra_args reaches every child through ParentRequest + msgspec; child
    ids are '<idx>_<parent>'; n=1 carries nothing."""
    kids = make_group("abc123#1", [5] * 40, n=9)
    for idx, r in enumerate(kids):
        assert r.request_id == f"{idx}_abc123#1"
        assert r.sampling_params.n == 1
        assert r.sampling_params.extra_args == {
            "sibling_gate": {"parent": "abc123#1", "n": 9, "leader": 0}}
    assert make_extra_args("x", 1) is None
    assert make_extra_args("x", None) is None
    single = make_single("solo", [5] * 40)
    assert single.sampling_params.extra_args is None
    s = build_scheduler(SiblingGateScheduler, mode="leader")
    assert s._sg_parse(kids[3]) == ("abc123#1", 9, 0, 3)
    assert s._sg_parse(single) is None


def test_bad_ids_are_not_gated():
    s = build_scheduler(SiblingGateScheduler, mode="leader")
    sp = SamplingParams(max_tokens=4, extra_args=make_extra_args("P", 9))
    for rid in ("P", "x_P", "3_Q", "12_P", "3-P"):
        r = _wire(rid, [5] * 20, sp, time.time())
        assert s._sg_parse(r) is None, rid


def _trace_equal(cls_a, cls_b, mode_b, arrivals_fn, **build_kw):
    a = Sim(build_scheduler(cls_a, **build_kw))
    b = Sim(build_scheduler(cls_b, mode=mode_b, **build_kw))
    a.run(arrivals_fn())
    b.run(arrivals_fn())
    ta = [(r.new, r.resumed, r.tokens, r.sampled) for r in a.recs]
    tb = [(r.new, r.resumed, r.tokens, r.sampled) for r in b.recs]
    assert ta == tb
    return len(ta)


def _n1_workload(seed: int, shared_prefix: bool):
    rng = random.Random(seed)
    base = prompt_of(seed, 400)
    arrivals: dict[int, list[Request]] = {}
    for i in range(40):
        length = rng.randrange(5, 1500)
        p = (base[:rng.randrange(0, 400)] + prompt_of(seed * 100 + i, length)
             if shared_prefix else prompt_of(seed * 100 + i, length))[:2000]
        arrivals.setdefault(rng.randrange(0, 60), []).append(
            make_single(f"r{seed}_{i}", p, max_tokens=rng.randrange(1, 25)))
    return arrivals


def test_n1_scheduling_unchanged():
    """With only n=1 requests the gate (leader mode) is the stock scheduler:
    identical per-step output, including chunked prefill, prefix caching and
    preemption under memory pressure."""
    steps = 0
    for seed in range(4):
        for kw in (dict(budget=512, num_blocks=4000, apc=False),
                   dict(budget=512, num_blocks=4000, apc=True),
                   dict(budget=256, num_blocks=180, apc=False),
                   dict(budget=256, num_blocks=180, apc=True)):
            steps += _trace_equal(
                Scheduler, SiblingGateScheduler, "leader",
                lambda: _n1_workload(seed, shared_prefix=kw["apc"]), **kw)
    assert steps > 500


def test_observe_mode_equals_stock_with_groups():
    """mode=observe never changes scheduling, also with n=9 groups present."""
    for arrival in ("A", "B", "C"):
        for config in CONFIGS:
            def arr():
                prompt = prompt_of(11, 700)
                kids = make_group("obs", prompt, n=9, max_tokens=5)
                others = [make_single(f"o{i}", prompt_of(20 + i, 300), 5)
                          for i in range(3)]
                if arrival == "A":
                    return {0: kids + others[:1], 2: others[1:]}
                if arrival == "B":
                    return {0: kids[:1], 1: kids[1:] + others}
                return {0: kids[:1], 1: kids[1:4], 2: kids[4:] + others}
            _trace_equal(Scheduler, SiblingGateScheduler, "observe", arr,
                         budget=BUDGET, apc=CONFIGS[config]["apc"])


def test_stock_scheduler_shows_the_race():
    """Sanity check of the harness: without the gate, k depends on arrival."""
    ks = {}
    for arrival in ("A", "B"):
        sim, parent, _ = scenario(Scheduler, None, "lmc", "hot", arrival)
        ks[arrival] = k_first(sim, parent)
    assert ks == {"A": 9, "B": 1}, ks


def arrivals_for(kind: str) -> tuple[str, ...]:
    """Arrival orders in which every sibling arrives before the leader's
    first token (C staggers them over the leader's prefill steps, so it only
    exists for the multi-step 'long' prompt)."""
    return ("A", "B", "C") if kind == "long" else ("A", "B")


def test_leader_mode_deterministic_everywhere():
    """k_first == 1 and an identical admission pattern for every arrival
    order, in every configuration; siblings are admitted exactly one step
    after the leader's first token; held siblings are never looked up."""
    for config in CONFIGS:
        for kind in KINDS:
            patterns = {}
            for arrival in arrivals_for(kind):
                sim, parent, lines = scenario(SiblingGateScheduler, "leader",
                                              config, kind, arrival)
                assert k_first(sim, parent) == 1, (config, kind, arrival)
                leader = f"0_{parent}"
                t1 = sim.first_token_step[leader]
                sib = [sim.admit_step[f"{i}_{parent}"] for i in range(1, 9)]
                assert min(sib) == t1 + 1, (config, kind, arrival, t1, sib)
                if sim.conn is not None:
                    sib_ids = {f"{i}_{parent}" for i in range(1, 9)}
                    looked = [st for st, rid in sim.conn.lookups
                              if rid in sib_ids]
                    assert min(looked) == t1 + 1
                pat = group_pattern(sim, parent)
                assert pat.startswith("0:[0]|"), (config, kind, arrival, pat)
                patterns[arrival] = pat
                summ = [ln for ln in lines if "event=group_admitted" in ln]
                assert len(summ) == 1 and _field(summ[0], "pattern") == pat
                assert _field(summ[0], "k_first") == "1"
                first = [ln for ln in lines if "event=first_admit" in ln]
                assert len(first) == 1 and _field(first[0], "k") == "1"
                assert _field(first[0], "admitted") == "[0]"
            assert len(set(patterns.values())) == 1, (config, kind, patterns)


def test_leader_mode_expected_patterns():
    """Concrete patterns: hot/cold = 900-token prompt, budget 1024; long =
    1900-token prompt, budget 512 (leader prefills in 4 steps)."""
    all8 = "[1,2,3,4,5,6,7,8]"
    want = {
        # LMCache / prefix-cache hits cost ~1 token: all 8 siblings fit
        ("lmc", "hot"): f"0:[0]|1:{all8}",
        ("lmc", "cold"): f"0:[0]|1:{all8}",   # leader stored the prompt
        ("lmc", "long"): f"0:[0]|4:{all8}",
        ("a1", "hot"): f"0:[0]|1:{all8}",
        ("a1", "cold"): f"0:[0]|1:{all8}",    # leader's blocks are cached
        ("a1", "long"): f"0:[0]|4:{all8}",
        # no reuse: siblings prefill in full, budget-limited (1023 tokens
        # left after the leader's decode token -> one 900-token prefill plus
        # the start of the next one)
        ("a0", "hot"): "0:[0]|1:[1,2]|",
        ("a0", "cold"): "0:[0]|1:[1,2]|",
        ("a0", "long"): "0:[0]|4:[1]|",
    }
    for (config, kind), pat in want.items():
        sim, parent, _ = scenario(SiblingGateScheduler, "leader", config,
                                  kind, "A")
        got = group_pattern(sim, parent)
        if pat.endswith("|"):
            assert got.startswith(pat), (config, kind, got)
        else:
            assert got == pat, (config, kind, got)


def test_late_siblings_still_k1():
    """Siblings that arrive after the leader already decodes are admitted on
    arrival (release reason leader_prefill_done); k_first stays 1."""
    for config in CONFIGS:
        sim, parent, lines = scenario(SiblingGateScheduler, "leader", config,
                                      "hot", "D")
        assert k_first(sim, parent) == 1
        summ = [ln for ln in lines if "event=group_admitted" in ln][0]
        assert _field(summ, "release").startswith("leader_prefill_done@")


def test_all_mode_barrier_deterministic():
    """mode=all: every child waits for the whole group, then they are
    admitted together: k_first = 9 for hits, budget-limited for misses,
    identical across arrival orders (the 'atomic add_request' semantics)."""
    for config in CONFIGS:
        for kind in ("hot", "cold"):
            pats = set()
            for arrival in ("A", "B", "C", "D"):
                sim, parent, lines = scenario(SiblingGateScheduler, "all",
                                              config, kind, arrival)
                pats.add(group_pattern(sim, parent))
                k = k_first(sim, parent)
                # Prefix caching (a1) dedupes co-admitted siblings even when
                # cold: vLLM V1 caches full blocks at allocation time, so a
                # sibling scheduled after child 0 in the same step hits them.
                reuse = config == "a1" or (config == "lmc" and kind == "hot")
                if reuse:
                    assert k == 9, (config, kind, arrival, k)
                else:
                    assert k < 9, (config, kind, arrival, k)
                if arrival == "D":
                    # 6 identical empty barrier steps -> one logged hold line
                    assert sum("event=hold" in ln for ln in lines) == 1
            assert len(pats) == 1, (config, kind, pats)


def test_leader_preempted_during_prefill():
    """The leader is preempted mid chunked prefill (memory pressure from an
    unrelated decoding request); its siblings stay held until its first
    token, then everything completes."""
    sched = build_scheduler(SiblingGateScheduler, mode="leader", budget=64,
                            num_blocks=10)
    sim = Sim(sched)
    other = make_single("other", prompt_of(1, 31), max_tokens=12)
    kids = make_group("pre", prompt_of(2, 100), n=9, max_tokens=3)
    with LogCapture() as cap:
        sim.run({0: [other], 1: kids})
    resumed = [rid for rec in sim.recs for rid in rec.resumed]
    assert "0_pre" in resumed, "scenario must preempt the leader"
    t1 = sim.first_token_step["0_pre"]
    for i in range(1, 9):
        assert sim.admit_step[f"{i}_pre"] > t1
    assert all(sim.s.requests.get(r.request_id) is None for r in kids)
    assert any("event=resume" in ln or "resumed=[0]" in ln
               for ln in cap.lines)
    assert k_first(sim, "pre") == 1


def test_abort_leader_releases_siblings():
    sched = build_scheduler(SiblingGateScheduler, mode="leader", budget=256)
    sim = Sim(sched)
    kids = make_group("ab", prompt_of(3, 900), n=9, max_tokens=3)

    def abort_leader(s):
        s.s.finish_requests("0_ab", RequestStatus.FINISHED_ABORTED)

    with LogCapture() as cap:
        sim.run({0: kids}, hooks={2: abort_leader})
    assert "0_ab" not in sim.first_token_step
    for i in range(1, 9):
        assert f"{i}_ab" in sim.first_token_step
    summ = [ln for ln in cap.lines if "event=group_admitted" in ln][0]
    assert _field(summ, "release") == "leader_finished@3"
    assert not sched._sg_groups  # cleaned up


def test_abort_whole_group_cleans_up():
    sched = build_scheduler(SiblingGateScheduler, mode="leader", budget=256)
    sim = Sim(sched)
    kids = make_group("gone", prompt_of(4, 900), n=9, max_tokens=3)

    def abort_all(s):
        s.s.finish_requests([r.request_id for r in kids],
                            RequestStatus.FINISHED_ABORTED)

    later = make_single("after", prompt_of(5, 300), max_tokens=3)
    with LogCapture() as cap:
        sim.run({0: kids, 3: [later]}, hooks={2: abort_all})
    assert not sched.has_requests()
    assert not sched._sg_groups
    assert any("event=group_gone" in ln for ln in cap.lines)
    assert "after" in sim.first_token_step
    # held siblings were never admitted
    assert all(f"{i}_gone" not in sim.admit_step for i in range(1, 9))


def test_leader_finishes_before_release():
    sched = build_scheduler(SiblingGateScheduler, mode="leader", budget=1024)
    sim = Sim(sched)
    kids = make_group("one", prompt_of(6, 100), n=9, max_tokens=1)
    with LogCapture() as cap:
        sim.run({0: kids})
    assert group_pattern(sim, "one") == "0:[0]|1:[1,2,3,4,5,6,7,8]"
    summ = [ln for ln in cap.lines if "event=group_admitted" in ln][0]
    assert _field(summ, "release") == "leader_finished@2"


def test_stall_valve_and_empty_steps():
    """No leader ever arrives: siblings are held for max_stall_steps empty
    (harmless) steps, then released."""
    sched = build_scheduler(SiblingGateScheduler, mode="leader",
                            env={"SIBLING_GATE_MAX_STALL_STEPS": 3})
    sim = Sim(sched)
    kids = make_group("nolead", prompt_of(7, 100), n=9, max_tokens=2)
    with LogCapture() as cap:
        sim.run({0: kids[1:]})
    # (the last step is stock vLLM's finished-request notification step)
    empty = [r for r in sim.recs if not r.tokens and r.step < 4]
    assert len(empty) == 3
    assert all(sim.admit_step[f"{i}_nolead"] == 4 for i in range(1, 9))
    assert any("released:valve_stall@4" in ln for ln in cap.lines)


def test_waiting_order_restored():
    """Held siblings go back to their original queue positions; requests
    behind them keep their relative order."""
    sched = build_scheduler(SiblingGateScheduler, mode="leader", budget=128)
    kids = make_group("ord", prompt_of(8, 1000), n=9, max_tokens=2)
    x = make_single("x", prompt_of(9, 50), max_tokens=2)
    y = make_single("y", prompt_of(10, 50), max_tokens=2)
    for r in kids + [x, y]:
        sched.add_request(r)
    Sim(sched).step()  # leader takes the whole budget (1000 > 128)
    order = [r.request_id for r in sched.waiting]
    assert order == [f"{i}_ord" for i in range(1, 9)] + ["x", "y"], order
    # Budget for x: the leader's chunk leaves room once its prompt is short.
    sched2 = build_scheduler(SiblingGateScheduler, mode="leader", budget=1024)
    kids2 = make_group("ord2", prompt_of(8, 900), n=9, max_tokens=2)
    x2 = make_single("x2", prompt_of(9, 50), max_tokens=2)
    y2 = make_single("y2", prompt_of(10, 200), max_tokens=2)
    for r in kids2 + [x2, y2]:
        sched2.add_request(r)
    rec = Sim(sched2).step()
    assert rec.new == ["0_ord2", "x2", "y2"][:len(rec.new)]
    assert [r.request_id for r in sched2.waiting][:8] == \
        [f"{i}_ord2" for i in range(1, 9)]


def test_priority_policy_supported():
    """The gate's priority-queue path. (Stock vLLM 0.10.x cannot queue n>1
    children under policy=priority at all: equal (priority, arrival_time)
    keys make heapq compare Request objects. The servers use FCFS.)"""
    sched = build_scheduler(SiblingGateScheduler, mode="leader",
                            policy="priority")
    sim = Sim(sched)
    kids = make_group("prio", prompt_of(12, 100), n=9, max_tokens=3,
                      distinct_arrival=True)
    sim.run({0: kids})
    assert group_pattern(sim, "prio") == "0:[0]|1:[1,2,3,4,5,6,7,8]"


def test_bad_mode_fails_loudly():
    try:
        build_scheduler(SiblingGateScheduler, mode="bogus")
    except ValueError as e:
        assert "SIBLING_GATE_MODE" in str(e)
    else:
        raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------
# Report / standalone runner
# --------------------------------------------------------------------------

def report() -> None:
    print(f"vLLM {vllm.__version__} from {os.path.dirname(vllm.__file__)}")
    print("n=9; hot/cold: 900-token prompt, budget 1024; long: 1900-token "
          "prompt, budget 512; arrivals A=all before first step, B=child 0 "
          "then the rest, C=staggered over the leader's prefill steps")
    print(f"{'config':5} {'kind':5} {'arrival':7} {'policy':7} "
          f"{'k':>2}  pattern (step offset: admitted child indices)")
    for config in CONFIGS:
        for kind in KINDS:
            for policy in ("stock", "leader", "all"):
                for arrival in arrivals_for(kind):
                    cls = Scheduler if policy == "stock" else \
                        SiblingGateScheduler
                    mode = None if policy == "stock" else policy
                    sim, parent, _ = scenario(cls, mode, config, kind,
                                              arrival)
                    print(f"{config:5} {kind:5} {arrival:7} {policy:7} "
                          f"{k_first(sim, parent):>2}  "
                          f"{group_pattern(sim, parent)}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        logging.getLogger("sibling_gate").setLevel(logging.WARNING)
        report()
        sys.exit(0)
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
