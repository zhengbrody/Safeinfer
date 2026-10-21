"""Static batching on the paged engine (experiment group "engine A").

A fixed set of sequences is prefilled together and decoded until each hits
its token budget; nothing joins mid-batch. Compared with HF this isolates
what paging alone buys (no padding, tighter memory, so larger batches fit);
continuous batching (week 3) then removes the "wait for the batch" cost.
"""

from __future__ import annotations

from collections.abc import Callable

from safeinfer.engine.model_runner import ModelRunner, SeqStep


def generate_static(
    runner: ModelRunner,
    prompts: list[list[int]],
    max_new_tokens: list[int],
    on_step: Callable[[], None] | None = None,
) -> list[list[int]]:
    """Greedy-decode every prompt for exactly its `max_new_tokens`; EOS is ignored.

    Sequence ids are the prompt indices. `on_step` runs after each step's
    tokens reach the host (the benchmark clock hooks in here). Finished
    sequences leave the batch and free their blocks immediately.
    """
    if len(prompts) != len(max_new_tokens):
        raise ValueError("prompts and max_new_tokens differ in length")
    outputs: list[list[int]] = [[] for _ in prompts]
    pending = [SeqStep(i, list(p)) for i, p in enumerate(prompts)]
    try:
        while pending:
            # .tolist() synchronizes with the GPU, so on_step sees finished work.
            next_tokens = runner.step(pending).logits.argmax(-1).tolist()
            if on_step:
                on_step()
            still_running = []
            for s, tok in zip(pending, next_tokens, strict=True):
                outputs[s.seq_id].append(tok)
                if len(outputs[s.seq_id]) < max_new_tokens[s.seq_id]:
                    still_running.append(SeqStep(s.seq_id, [tok]))
                else:
                    runner.free(s.seq_id)
            pending = still_running
    finally:
        # Non-empty only if a step raised; release whatever those sequences hold.
        for s in pending:
            if runner.block_manager.num_tokens(s.seq_id):
                runner.free(s.seq_id)
    return outputs
