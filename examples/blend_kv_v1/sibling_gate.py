# SPDX-License-Identifier: Apache-2.0
"""Deterministic admission of parallel-sampling children (vLLM V1, n > 1).

Problem
-------
``AsyncLLM.add_request`` fans an n>1 request out into n child requests
``"<idx>_<parent_id>"``. Each child is sent to the engine process as its own
ZMQ message. ``EngineCoreProc._process_input_queue`` wakes on the first
message, drains only what is already queued, and steps. How many children land
in the first scheduler step (k) is therefore a thread-timing race. TTFT (first
token of any child) grows with k, because the first step prefills, loads or
blends every co-admitted child.

Policy (``mode="leader"``, the default)
---------------------------------------
Admit exactly one child, the leader (index 0). Hold its siblings in the waiting
queue until the leader's prefill has executed (it has an output token), or the
leader has finished or been aborted, or a safety valve fires. Then release them
and let vLLM schedule them normally (token budget, FCFS, preemption). Result:
k = 1 in every configuration and every run. The siblings can reuse the leader's
KV (LMCache hit plus sibling dedup, or vLLM's prefix cache when enabled).

Other modes:
- ``"all"``: barrier. Hold every child until all n have arrived, then release
  them together. These are the "atomic add_request" semantics.
- ``"observe"``: no gating, log only. Scheduling is identical to stock vLLM.

Mechanism
---------
``SiblingGateScheduler`` subclasses ``vllm.v1.core.sched.scheduler.Scheduler``
and overrides two methods:
- ``add_request()`` registers group membership;
- ``schedule()`` takes held siblings out of ``self.waiting`` for the duration of
  ``super().schedule()``, then puts them back at their original queue positions.

Between steps, held requests sit in ``self.waiting``, so abort, finish and
stats code paths see them exactly as in stock vLLM. While no sibling group is
live (every n=1 request), ``schedule()`` is ``super().schedule()`` unchanged.

The client declares group membership with
``SamplingParams.extra_args["sibling_gate"] = {"parent": <request_id>,
"n": n, "leader": 0}`` (see ``make_extra_args``). vLLM shallow-copies
``extra_args`` into every child (``ParentRequest._get_child_sampling_params``),
and msgspec carries it to the engine process. The child index is parsed from
the child id ``"<idx>_<parent>"`` and checked against the declared parent id.
Requests without the key, or with a mismatching id, are never gated.

Enable it with ``AsyncEngineArgs(scheduler_cls="sibling_gate.SiblingGateScheduler")``.
This module must be importable in the engine process; ``engine_kwargs()``
arranges that. Written against vLLM 0.10.2 (cluster); also tested on 0.10.1.1.
Not compatible with ``async_scheduling`` (vLLM then forces its AsyncScheduler).

Environment knobs, read in the engine process when the scheduler is built:
  SIBLING_GATE_MODE               leader | all | observe              (leader)
  SIBLING_GATE_MAX_STALL_STEPS    leader mode: release if the leader gets no
                                  tokens for this many consecutive steps
                                  while its siblings wait                (64)
  SIBLING_GATE_MAX_HOLD_STEPS     leader mode: release after this many
                                  holding steps                        (4096)
  SIBLING_GATE_BARRIER_TIMEOUT_S  all mode: longest wait for all n children
                                  to arrive, in seconds                 (5.0)
  SIBLING_GATE_LOG                1: one [SIBLING_GATE] line on stderr per
                                  group per step with activity            (1)
"""

from __future__ import annotations

import collections
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from vllm.v1.core.sched.scheduler import Scheduler

EXTRA_ARGS_KEY = "sibling_gate"
MODES = ("leader", "all", "observe")
SERVER_MODES = MODES + ("off",)  # "off" = stock vLLM scheduler class

_logger = logging.getLogger("sibling_gate")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


# ---------------------------------------------------------------------------
# Client-side helpers (API / server process)
# ---------------------------------------------------------------------------

def make_extra_args(parent_id: str, n: Optional[int],
                    leader: int = 0) -> Optional[dict[str, Any]]:
    """``SamplingParams.extra_args`` declaring an n>1 request as a sibling
    group, or None for n <= 1 (those requests are never gated)."""
    if n is None or int(n) <= 1:
        return None
    return {EXTRA_ARGS_KEY: {"parent": str(parent_id), "n": int(n),
                             "leader": int(leader)}}


def emit_child_lines(req_id: str, n: int, child_ttft: dict, child_text: dict,
                     child_finish: dict) -> None:
    """Server-side per-child record for an n>1 request (stdout).

    ``[CHILD]`` has one line per child: first-token time (s since submission),
    finish reason, characters. ``[CHILD_SUMMARY]`` gives child-0 TTFT, the
    median over children and the last child, plus which child produced the
    request's first token. ``[CHILD_OUTPUT]`` gives one JSON line per child
    with its own text, for quality checks (``[INFO] Final output``
    interleaves the children)."""
    import json
    import statistics
    for idx in range(n):
        t = child_ttft.get(idx)
        print(f"[CHILD] req_id={req_id} idx={idx} "
              f"ttft={'NA' if t is None else f'{t:.4f}'} "
              f"finish={child_finish.get(idx)} "
              f"chars={len(child_text.get(idx, ''))}")
    times = sorted(child_ttft.values())
    first_idx = (min(child_ttft, key=child_ttft.get) if child_ttft else None)

    def _f(x):
        return "NA" if x is None else f"{x:.4f}"

    print(f"[CHILD_SUMMARY] req_id={req_id} n={n} "
          f"children_with_text={len(child_ttft)} first_idx={first_idx} "
          f"child0_ttft={_f(child_ttft.get(0))} "
          f"median_ttft={_f(statistics.median(times) if times else None)} "
          f"last_ttft={_f(times[-1] if times else None)}")
    for idx in range(n):
        print("[CHILD_OUTPUT] " + json.dumps(
            {"req_id": req_id, "idx": idx,
             "text": child_text.get(idx, "")}, ensure_ascii=False))
    sys.stdout.flush()


def engine_kwargs(mode: str) -> dict[str, Any]:
    """Prepare the engine process for ``mode`` and return the extra
    ``AsyncEngineArgs`` kwargs. ``"off"`` returns {} (stock scheduler)."""
    if mode == "off":
        return {}
    if mode not in MODES:
        raise ValueError(f"sibling gate mode must be one of {SERVER_MODES}, "
                         f"got {mode!r}")
    os.environ["SIBLING_GATE_MODE"] = mode
    # The scheduler class is resolved by name in the engine process
    # (resolve_obj_by_qualname). Forked children inherit sys.path and spawned
    # ones get it from multiprocessing; PYTHONPATH also covers anything that
    # starts a fresh interpreter.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if here not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([here] + parts)
    return {"scheduler_cls": f"{__name__}.SiblingGateScheduler"}


# ---------------------------------------------------------------------------
# Engine-side scheduler
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw in (None, "") else float(raw)


def _fmt(idxs) -> str:
    return "[" + ",".join(str(i) for i in idxs) + "]"


@dataclass
class _Group:
    parent: str
    n: int
    leader: int
    created_step: int
    created_mono: float
    members: dict[int, str] = field(default_factory=dict)  # idx -> req id
    admitted: dict[int, int] = field(default_factory=dict)  # idx -> step
    release_reason: Optional[str] = None
    release_step: Optional[int] = None
    first_admit_step: Optional[int] = None
    k_first: Optional[int] = None
    stall_steps: int = 0
    hold_steps: int = 0
    summary_logged: bool = False
    dead_steps: int = 0
    last_hold_sig: Optional[tuple] = None  # suppresses repeated hold lines

    def pattern(self) -> str:
        """Admission pattern relative to the first admission step, e.g.
        ``0:[0]|1:[1,2,3,4,5,6,7,8]``: the per-run determinism fingerprint."""
        if self.first_admit_step is None:
            return "-"
        by_step: dict[int, list[int]] = collections.defaultdict(list)
        for idx, st in self.admitted.items():
            by_step[st - self.first_admit_step].append(idx)
        return "|".join(f"{off}:{_fmt(sorted(v))}"
                        for off, v in sorted(by_step.items()))


class SiblingGateScheduler(Scheduler):
    """vLLM V1 scheduler that makes the first-step admission of n>1 children
    deterministic. See the module docstring."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        mode = os.environ.get("SIBLING_GATE_MODE", "leader").strip().lower()
        if mode not in MODES:
            raise ValueError(
                f"[SIBLING_GATE] SIBLING_GATE_MODE must be one of {MODES}, "
                f"got {mode!r}. To disable the gate, do not pass "
                f"scheduler_cls at all.")
        self._sg_mode = mode
        self._sg_max_stall = _env_int("SIBLING_GATE_MAX_STALL_STEPS", 64)
        self._sg_max_hold = _env_int("SIBLING_GATE_MAX_HOLD_STEPS", 4096)
        self._sg_barrier_s = _env_float("SIBLING_GATE_BARRIER_TIMEOUT_S", 5.0)
        self._sg_log_on = os.environ.get("SIBLING_GATE_LOG", "1") not in (
            "0", "false", "False", "")
        self._sg_step = 0
        self._sg_groups: dict[str, _Group] = {}
        self._sg_warned: set[str] = set()
        self._sg_check_compat()
        try:
            import vllm
            vllm_version = getattr(vllm, "__version__", "?")
        except Exception:  # pragma: no cover
            vllm_version = "?"
        _logger.info(
            "[SIBLING_GATE] init scheduler=%s mode=%s max_stall_steps=%d "
            "max_hold_steps=%d barrier_timeout_s=%.3f queue=%s vllm=%s",
            type(self).__name__, self._sg_mode, self._sg_max_stall,
            self._sg_max_hold, self._sg_barrier_s,
            type(self.waiting).__name__, vllm_version)

    # -- compatibility -----------------------------------------------------

    def _sg_check_compat(self) -> None:
        missing = [a for a in ("waiting", "running", "requests")
                   if not hasattr(self, a)]
        missing += [f"waiting.{m}" for m in ("remove_requests", "add_request",
                                             "__iter__")
                    if not hasattr(getattr(self, "waiting", None), m)]
        if missing:
            raise RuntimeError(
                "[SIBLING_GATE] the vLLM Scheduler lacks " + ", ".join(missing)
                + "; sibling_gate.py was written against vLLM 0.10.1.1/0.10.2. "
                "Run without the gate (--sibling-gate off) or port it.")
        if getattr(self.scheduler_config, "async_scheduling", False):
            raise RuntimeError(
                "[SIBLING_GATE] async_scheduling is not supported: the release "
                "condition reads the leader's output tokens, which lag a step "
                "under async scheduling.")

    def _sg_warn(self, key: str, msg: str, *args) -> None:
        if key not in self._sg_warned:
            self._sg_warned.add(key)
            _logger.warning(msg, *args)

    # -- group membership -------------------------------------------------

    def _sg_parse(self, request) -> Optional[tuple[str, int, int, int]]:
        sp = getattr(request, "sampling_params", None)
        extra = getattr(sp, "extra_args", None) if sp is not None else None
        if not extra:
            return None
        spec = extra.get(EXTRA_ARGS_KEY)
        if not spec:
            return None
        try:
            parent = str(spec["parent"])
            n = int(spec["n"])
            leader = int(spec.get("leader", 0))
        except (KeyError, TypeError, ValueError):
            self._sg_warn("spec", "[SIBLING_GATE] WARNING malformed "
                          "extra_args[%r]=%r on %s; request NOT gated",
                          EXTRA_ARGS_KEY, spec, request.request_id)
            return None
        if n <= 1:
            return None
        head, sep, tail = request.request_id.partition("_")
        if not sep or not (head.isascii() and head.isdigit()) \
                or tail != parent:
            self._sg_warn(
                "id", "[SIBLING_GATE] WARNING request id %r is not "
                "'<idx>_%s' (vLLM child-id format changed?); request NOT "
                "gated", request.request_id, parent)
            return None
        idx = int(head)
        if not (0 <= idx < n and 0 <= leader < n):
            self._sg_warn("range", "[SIBLING_GATE] WARNING child %s: index "
                          "%d / leader %d outside n=%d; request NOT gated",
                          request.request_id, idx, leader, n)
            return None
        return parent, n, leader, idx

    def add_request(self, request) -> None:
        super().add_request(request)
        info = self._sg_parse(request)
        if info is None:
            return
        parent, n, leader, idx = info
        g = self._sg_groups.get(parent)
        if g is not None and idx in g.members:
            # A finished group under a reused parent id: start over.
            g = None
        if g is None:
            g = _Group(parent=parent, n=n, leader=leader,
                       created_step=self._sg_step,
                       created_mono=time.monotonic())
            if self._sg_mode == "observe":
                g.release_reason, g.release_step = "observe", self._sg_step
            self._sg_groups[parent] = g
        g.members[idx] = request.request_id

    # -- scheduling -------------------------------------------------------

    def schedule(self):
        self._sg_step += 1
        if not self._sg_groups:
            return super().schedule()
        step = self._sg_step
        hold_ids = self._sg_hold_ids(step)
        held: list = []
        snapshot: list = []
        if hold_ids:
            snapshot = list(self.waiting)
            held = [r for r in snapshot if r.request_id in hold_ids]
            if held:
                self.waiting.remove_requests(held)
        try:
            out = super().schedule()
        finally:
            if held:
                self._sg_reinsert(snapshot, held)
        self._sg_account(out, {r.request_id for r in held}, step)
        return out

    def _sg_release_reason(self, g: _Group) -> Optional[str]:
        if self._sg_mode == "leader":
            rid = g.members.get(g.leader)
            if rid is not None:
                leader = self.requests.get(rid)
                if leader is None or leader.is_finished():
                    return "leader_finished"
                if leader.num_output_tokens > 0:
                    # update_from_output of the step that computed the last
                    # prompt token has run: the leader's KV is in its blocks
                    # (and, with LMCache, stored).
                    return "leader_prefill_done"
            if g.stall_steps >= self._sg_max_stall:
                return "valve_stall"
            if g.hold_steps >= self._sg_max_hold:
                return "valve_hold"
            return None
        # mode == "all": barrier on arrival of all n children.
        if len(g.members) >= g.n:
            return "all_arrived"
        for rid in g.members.values():
            r = self.requests.get(rid)
            if r is None or r.is_finished():
                return "member_finished"
        if time.monotonic() - g.created_mono >= self._sg_barrier_s:
            return "valve_barrier_timeout"
        return None

    def _sg_hold_ids(self, step: int) -> set[str]:
        hold: set[str] = set()
        for g in self._sg_groups.values():
            if g.release_reason is not None:
                continue
            reason = self._sg_release_reason(g)
            if reason is not None:
                g.release_reason, g.release_step = reason, step
                continue
            for idx, rid in g.members.items():
                if idx in g.admitted:
                    continue
                if self._sg_mode == "leader" and idx == g.leader:
                    continue
                hold.add(rid)
        return hold

    def _sg_reinsert(self, snapshot: list, held: list) -> None:
        """Put held requests back where they were. Order of everything else is
        left exactly as ``super().schedule()`` produced it."""
        if not isinstance(self.waiting, collections.deque):
            # Priority queue: ordering is (priority, arrival_time) anyway.
            for r in held:
                self.waiting.add_request(r)
            return
        pos = {id(r): i for i, r in enumerate(snapshot)}
        result = list(self.waiting)
        present = {id(r) for r in result}
        for h in held:  # snapshot order
            anchor = None
            for j in range(pos[id(h)] - 1, -1, -1):
                if id(snapshot[j]) in present:
                    anchor = snapshot[j]
                    break
            if anchor is None:
                # After requests that entered the queue during this step
                # (preempted ones are prepended), before all older ones.
                at = 0
                while at < len(result) and id(result[at]) not in pos:
                    at += 1
            else:
                at = next(i for i, r in enumerate(result) if r is anchor) + 1
            result.insert(at, h)
            present.add(id(h))
        self.waiting.clear()
        self.waiting.extend(result)

    def _sg_account(self, out, held_ids: set[str], step: int) -> None:
        new_ids = {r.req_id for r in out.scheduled_new_reqs}
        resumed_ids: set[str] = set()
        cached = out.scheduled_cached_reqs
        if not isinstance(cached, list):
            resumed_ids = {
                rid for rid, res in zip(cached.req_ids,
                                        cached.resumed_from_preemption)
                if res
            }
        scheduled = out.num_scheduled_tokens
        waiting_ids: Optional[set[str]] = None
        gone: list[str] = []

        for parent, g in self._sg_groups.items():
            if len(g.admitted) < g.n:
                newly = sorted(i for i, rid in g.members.items()
                               if rid in new_ids and i not in g.admitted)
                for i in newly:
                    g.admitted[i] = step
                resumed = sorted(i for i, rid in g.members.items()
                                 if rid in resumed_ids)
                held = sorted(i for i, rid in g.members.items()
                              if rid in held_ids)
                if newly and g.first_admit_step is None:
                    g.first_admit_step, g.k_first = step, len(newly)
                    event = "first_admit"
                elif newly:
                    event = "admit"
                elif held:
                    event = "hold"
                elif resumed:
                    event = "resume"
                elif g.release_step == step:
                    event = "release"
                else:
                    event = None
                if g.release_reason is None:
                    g.hold_steps += 1
                    lrid = g.members.get(g.leader)
                    if lrid is not None and lrid in scheduled:
                        g.stall_steps = 0
                    else:
                        g.stall_steps += 1
                if event == "hold":
                    # Log a hold step only when something changed (arrivals,
                    # leader progress); the all-mode barrier can spin through
                    # many identical empty steps.
                    lr = self.requests.get(g.members.get(g.leader, ""))
                    sig = (tuple(held), len(g.members),
                           lr.num_computed_tokens if lr is not None else -1)
                    if sig == g.last_hold_sig:
                        event = None
                    g.last_hold_sig = sig
                if self._sg_log_on and event is not None:
                    if waiting_ids is None:
                        waiting_ids = {r.request_id for r in self.waiting}
                    queued = sorted(i for i, rid in g.members.items()
                                    if rid in waiting_ids
                                    and rid not in held_ids)
                    self._sg_log_step(g, step, event, newly, resumed, held,
                                      queued)
                if len(g.admitted) >= g.n:
                    self._sg_log_summary(g, step, "group_admitted")

            if any(rid in self.requests for rid in g.members.values()):
                g.dead_steps = 0
            else:
                g.dead_steps += 1
                if len(g.members) >= g.n or g.dead_steps > 1024:
                    gone.append(parent)

        for parent in gone:
            g = self._sg_groups.pop(parent)
            if not g.summary_logged:
                self._sg_log_summary(g, step, "group_gone")

    # -- logging ----------------------------------------------------------

    def _sg_gate_str(self, g: _Group) -> str:
        if g.release_reason is None:
            return "holding"
        return f"released:{g.release_reason}@{g.release_step}"

    def _sg_log_step(self, g: _Group, step: int, event: str, newly, resumed,
                     held, queued) -> None:
        lrid = g.members.get(g.leader)
        lr = self.requests.get(lrid) if lrid is not None else None
        if lrid is None:
            leader = "absent"
        elif lr is None:
            leader = "gone"
        else:
            leader = (f"{lr.num_computed_tokens}/{lr.num_prompt_tokens},"
                      f"out={lr.num_output_tokens},{lr.status.name}")
        k = f" k={g.k_first}" if event == "first_admit" else ""
        res = f" resumed={_fmt(resumed)}" if resumed else ""
        _logger.info(
            "[SIBLING_GATE] step=%d mode=%s parent=%s n=%d event=%s%s "
            "admitted=%s%s held=%s queued=%s pending=%d leader=%s gate=%s",
            step, self._sg_mode, g.parent, g.n, event, k, _fmt(newly), res,
            _fmt(held), _fmt(queued), g.n - len(g.members), leader,
            self._sg_gate_str(g))

    def _sg_log_summary(self, g: _Group, step: int, event: str) -> None:
        g.summary_logged = True
        if not self._sg_log_on:
            return
        last = max(g.admitted.values()) if g.admitted else None
        span = (last - g.first_admit_step) if last is not None and \
            g.first_admit_step is not None else None
        _logger.info(
            "[SIBLING_GATE] step=%d mode=%s parent=%s n=%d event=%s "
            "k_first=%s first_admit_step=%s steps_to_all=%s admitted=%d/%d "
            "hold_steps=%d release=%s pattern=%s",
            step, self._sg_mode, g.parent, g.n, event, g.k_first,
            g.first_admit_step, span, len(g.admitted), g.n, g.hold_steps,
            "-" if g.release_reason is None else
            f"{g.release_reason}@{g.release_step}", g.pattern())
