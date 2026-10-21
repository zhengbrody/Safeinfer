"""Our paged engine in static-batching mode as a static_replay backend ("engine A")."""

from __future__ import annotations

import time

from safeinfer.bench.static_replay import CanAdmit, RunBatch
from safeinfer.bench.workload import Request
from safeinfer.engine.model_runner import ModelRunner
from safeinfer.engine.static_batch import generate_static


def engine_batch_runner(runner: ModelRunner) -> RunBatch:
    def run_batch(batch: list[Request], t0: float) -> list[float]:
        step_times: list[float] = []
        generate_static(
            runner,
            [r.prompt_token_ids for r in batch],
            [r.output_len for r in batch],
            on_step=lambda: step_times.append(time.perf_counter() - t0),
        )
        return step_times

    return run_batch


def engine_can_admit(runner: ModelRunner) -> CanAdmit:
    """Admit while the batch's worst-case KV (prompt + full output) fits the free blocks.

    Static mode has no preemption, so it must reserve for the full output up
    front; continuous batching (week 3) admits optimistically instead.
    """
    bm = runner.block_manager

    def blocks(r: Request) -> int:
        return -(-(r.prompt_len + r.output_len) // bm.block_size)

    def can_admit(batch: list[Request], candidate: Request) -> bool:
        return sum(blocks(r) for r in batch) + blocks(candidate) <= bm.num_free_blocks

    return can_admit
