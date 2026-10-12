# SPDX-License-Identifier: Apache-2.0
"""One-minute GPU smoke test of sibling_gate on a real vLLM engine.

Covers what the CPU tests cannot: AsyncLLM's n=9 fan-out over ZMQ, the
EngineCoreProc busy loop and its arrival race, and the scheduler class
resolved by name inside the engine process. Uses a tiny model on one GPU;
no LMCache, no TP.

Replays R pairs (greedy n=1, then n=9 with the identical prompt), one logical
request at a time, like BlendServer.run_mock_experiment. Prompt lengths cycle
through 150/600/1900/200/1200 tokens.

    cd examples/blend_kv_v1
    python tests/e2e_sibling_gate_smoke.py leader 10 > smoke.out 2> smoke.err
    grep 'event=first_admit' smoke.err | grep -o ' k=[0-9]*' | sort | uniq -c
      -> leader: "10 k=1";  observe: a mix (the race);  all: one k per length

usage: e2e_sibling_gate_smoke.py <leader|all|observe|off> <R> [apc|noapc]
                                 [max_num_batched_tokens] [model]
"""
import asyncio
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sibling_gate import emit_child_lines, engine_kwargs, make_extra_args  # noqa

from vllm import AsyncLLMEngine, SamplingParams  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402

mode = sys.argv[1]
R = int(sys.argv[2])
apc = len(sys.argv) > 3 and sys.argv[3] == "apc"
budget = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
model = sys.argv[5] if len(sys.argv) > 5 else "facebook/opt-125m"


async def one(llm, req_id, prompt, n):
    sp = SamplingParams(
        n=n, temperature=0.7 if n > 1 else 0.0, max_tokens=16,
        ignore_eos=True, output_kind=RequestOutputKind.DELTA,
        extra_args=None if mode == "off" else make_extra_args(req_id, n))
    t0 = time.perf_counter()
    ttft = None
    child_ttft, child_text, child_finish = {}, {}, {}
    async for out in llm.generate(TokensPrompt(prompt_token_ids=prompt), sp,
                                  request_id=req_id):
        c = out.outputs[0]
        if c.text:
            now = time.perf_counter() - t0
            if ttft is None:
                ttft = now
            child_ttft.setdefault(c.index, now)
            child_text[c.index] = child_text.get(c.index, "") + c.text
        if c.finish_reason is not None:
            child_finish[c.index] = c.finish_reason
        if out.finished:
            break
    print(f"[INFO] TTFT: {ttft:.4f} seconds (req_id={req_id}) n={n} "
          f"prompt_tokens={len(prompt)}", flush=True)
    if n > 1:
        emit_child_lines(req_id, n, child_ttft, child_text, child_finish)


async def main():
    llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=model, gpu_memory_utilization=0.45, max_model_len=2048,
        max_num_batched_tokens=budget, enable_prefix_caching=apc,
        enforce_eager=True, seed=42, **engine_kwargs(mode)))
    try:
        rng = random.Random(0)
        for i in range(R):
            length = [150, 600, 1900, 200, 1200][i % 5]
            prompt = [rng.randrange(100, 30000) for _ in range(length)]
            await one(llm, f"g{i:03d}", prompt, 1)
            await one(llm, f"s{i:03d}", prompt, 9)
    finally:
        llm.shutdown()
    print("[DONE]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
