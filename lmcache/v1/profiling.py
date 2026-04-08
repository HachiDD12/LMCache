# SPDX-License-Identifier: Apache-2.0
"""
KV Cache profiling utilities for identifying bottlenecks in LMCache.

Enable profiling by setting the environment variable:
    LMCACHE_PROFILE=1

Control output location:
    LMCACHE_PROFILE_OUTPUT=lmcache_profile.jsonl  (default)

Each store/retrieve operation emits one JSON line with phase-level timings
(in milliseconds). GPU timings use torch.cuda.synchronize() for accuracy.

Operations form a stack per thread; nested ops automatically inherit the
``req_id`` of their parent, so once an adapter wraps a per-request block in
``profiler.op(req_id=...)`` every cache_engine / gpu_connector op underneath
is keyed by that request ID without further plumbing.
"""

# Standard
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Optional
import json
import os
import threading
import time

# Third Party
import torch

_ENABLED = os.environ.get("LMCACHE_PROFILE", "0") == "1"
_OUTPUT_PATH = os.environ.get("LMCACHE_PROFILE_OUTPUT", "lmcache_profile.jsonl")


class KVCacheProfiler:
    """Singleton profiler for KV cache transfer operations.

    All timing values are in milliseconds. GPU phases call
    torch.cuda.synchronize() before/after to capture real device time.

    Thread-safety: each thread maintains its own op stack via ``threading.local``,
    so TP workers do not bleed phases between requests. The output JSONL writer
    is guarded by a process-wide lock so concurrent flushes do not interleave.
    """

    _instance: Optional["KVCacheProfiler"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.enabled = _ENABLED
        self._records: dict[str, list[float]] = defaultdict(list)
        self._tls = threading.local()
        self._write_lock = threading.Lock()
        self._output_path = _OUTPUT_PATH

        if self.enabled and os.path.exists(self._output_path):
            os.remove(self._output_path)

    @classmethod
    def get(cls) -> "KVCacheProfiler":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Per-thread op stack
    # ------------------------------------------------------------------

    def _stack(self) -> list[dict]:
        stack = getattr(self._tls, "stack", None)
        if stack is None:
            stack = []
            self._tls.stack = stack
        return stack

    # ------------------------------------------------------------------
    # Operation lifecycle
    # ------------------------------------------------------------------

    def start_op(self, op_type: str, req_id: Optional[str] = None, **metadata: Any) -> None:
        """Begin a new operation (e.g. 'store', 'retrieve').

        If ``req_id`` is not given, the new op inherits the ``req_id`` of its
        parent op (if any). This lets adapter-level wrappers stamp the request
        ID once and have nested cache_engine / gpu_connector ops pick it up
        automatically.
        """
        if not self.enabled:
            return
        stack = self._stack()
        if req_id is None and stack:
            req_id = stack[-1].get("req_id")
        op: dict[str, Any] = {
            "op": op_type,
            "phases": {},
            **metadata,
        }
        if req_id is not None:
            op["req_id"] = req_id
        stack.append(op)

    def end_op(self) -> None:
        """Finish the most recently started op; flush its record to JSONL."""
        if not self.enabled:
            return
        stack = self._stack()
        if not stack:
            return
        op = stack.pop()
        op["wall_ts"] = time.time()
        op["thread"] = threading.get_ident()
        with self._write_lock:
            with open(self._output_path, "a") as f:
                f.write(json.dumps(op) + "\n")

    @contextmanager
    def op(self, op_type: str, req_id: Optional[str] = None, **metadata: Any):
        """Context manager wrapping ``start_op`` / ``end_op``.

        Preferred entry point for new instrumentation -- guarantees ``end_op``
        runs even if the body raises.
        """
        self.start_op(op_type, req_id=req_id, **metadata)
        try:
            yield
        finally:
            self.end_op()

    # ------------------------------------------------------------------
    # Phase timing helpers
    # ------------------------------------------------------------------

    @contextmanager
    def phase(self, name: str, sync_cuda: bool = True):
        """Context manager that times a named phase.

        The elapsed value is attached to the innermost active op (if any) and
        also accumulated into the global ``_records`` dict for the summary.

        Args:
            name: human-readable phase label, e.g. 'store.gpu_offload'.
            sync_cuda: if True, call torch.cuda.synchronize() for
                accurate GPU timing (adds overhead, but needed for
                async CUDA ops).
        """
        if not self.enabled:
            yield
            return

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._records[name].append(elapsed_ms)
        stack = self._stack()
        if stack:
            stack[-1]["phases"][name] = elapsed_ms

    def record(self, name: str, elapsed_ms: float) -> None:
        """Manually record a timing value for a phase."""
        if not self.enabled:
            return
        self._records[name].append(elapsed_ms)
        stack = self._stack()
        if stack:
            stack[-1]["phases"][name] = elapsed_ms

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of all recorded phases."""
        if not self._records:
            return "(no profiling data collected)"

        lines = [
            "",
            "=" * 72,
            "  LMCache KV-Cache Profile Summary",
            "=" * 72,
            f"  {'Phase':<40} {'Count':>6} {'Avg ms':>10} "
            f"{'Total ms':>10} {'Max ms':>10}",
            "-" * 72,
        ]
        for phase in sorted(self._records):
            vals = self._records[phase]
            n = len(vals)
            avg = sum(vals) / n
            total = sum(vals)
            mx = max(vals)
            lines.append(
                f"  {phase:<40} {n:>6} {avg:>10.2f} "
                f"{total:>10.2f} {mx:>10.2f}"
            )
        lines.append("=" * 72)
        return "\n".join(lines)
