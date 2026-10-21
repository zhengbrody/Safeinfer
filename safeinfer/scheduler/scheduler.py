"""Iteration-level scheduler: decides, every step, which sequences run and how many tokens each.

Policy (FCFS, close to vLLM's):

1. Running sequences go first, in admission order. Each asks for its
   uncomputed tokens (1 when decoding; the rest of its prompt mid chunked
   prefill), capped by the step's token budget.
2. If the KV pool cannot hold a running sequence's new tokens, the most
   recently admitted running sequence is preempted: its blocks are freed and
   it rejoins the *front* of the waiting queue to be recomputed from scratch
   later (prompt + tokens generated so far). Recompute beats swapping to CPU
   for short sequences and keeps the memory model simple.
3. Only if nothing was preempted this step are waiting sequences admitted, in
   arrival order, while the token budget, `max_num_seqs`, and free blocks
   allow. Admission is optimistic: a sequence reserves blocks for the tokens
   it computes now, not for its whole future output; preemption covers the
   cases where that bet loses.

Without chunked prefill a prompt must fit the remaining budget in one step;
with it, a prompt is split across steps and shares them with decodes.

The scheduler only does bookkeeping. It never allocates blocks: the
ModelRunner grows each scheduled sequence when it runs the step. It does
free blocks, on preemption and when sequences finish.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from safeinfer.memory.block_manager import BlockManager
from safeinfer.scheduler.sequence import Sequence, SequenceStatus


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int = 64  # concurrently running sequences
    max_num_batched_tokens: int = 2048  # tokens per forward step
    max_model_len: int = 2048  # prompt + output, per sequence
    enable_chunked_prefill: bool = False

    def __post_init__(self):
        if min(self.max_num_seqs, self.max_num_batched_tokens, self.max_model_len) < 1:
            raise ValueError("scheduler limits must be positive")
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            # A preempted sequence is recomputed in one step, so the budget must cover the longest one.
            raise ValueError("without chunked prefill, max_num_batched_tokens must be >= max_model_len")


@dataclass(frozen=True)
class ScheduledSeq:
    seq: Sequence
    num_tokens: int  # computed this step, starting at seq.num_computed_tokens

    @property
    def token_ids(self) -> list[int]:
        start = self.seq.num_computed_tokens
        return self.seq.token_ids(start, start + self.num_tokens)

    @property
    def samples(self) -> bool:
        """True if this step reaches the sequence's last token, so it yields a next token."""
        return self.seq.num_computed_tokens + self.num_tokens == self.seq.num_tokens


@dataclass(frozen=True)
class SchedulerOutput:
    scheduled: list[ScheduledSeq]
    preempted: list[Sequence]


class Scheduler:
    def __init__(self, config: SchedulerConfig, block_manager: BlockManager, eos_token_ids: tuple[int, ...] = ()):
        bm = block_manager
        if -(-config.max_model_len // bm.block_size) > bm.num_blocks:
            raise ValueError(
                f"KV pool ({bm.num_blocks} blocks of {bm.block_size}) cannot hold one max_model_len sequence"
            )
        self.config = config
        self.block_manager = bm
        self.eos_token_ids = frozenset(eos_token_ids)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # admission order: first = highest priority
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        if not seq.prompt_token_ids:
            raise ValueError("empty prompt")
        if len(seq.prompt_token_ids) + seq.params.max_tokens > self.config.max_model_len:
            raise ValueError(
                f"seq {seq.seq_id}: prompt {len(seq.prompt_token_ids)} + max_tokens {seq.params.max_tokens} "
                f"exceeds max_model_len {self.config.max_model_len}"
            )
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        bm = self.block_manager
        budget = self.config.max_num_batched_tokens
        free = bm.num_free_blocks
        scheduled: list[ScheduledSeq] = []
        preempted: list[Sequence] = []

        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            n = min(seq.num_uncomputed_tokens, budget)
            need = bm.blocks_needed(seq.seq_id, n)
            # Victims come from the tail, which has not been scheduled yet this step.
            while need > free and self.running[-1] is not seq:
                free += self._preempt(self.running.pop(), preempted)
            if need > free:
                # seq is now the lowest-priority runner and still does not fit: it yields.
                self._preempt(self.running.pop(), preempted)
                break
            scheduled.append(ScheduledSeq(seq, n))
            budget -= n
            free -= need
            i += 1

        if not preempted:
            while self.waiting and budget > 0 and len(self.running) < self.config.max_num_seqs:
                seq = self.waiting[0]
                n = seq.num_uncomputed_tokens
                if n > budget:
                    if not self.config.enable_chunked_prefill:
                        break
                    n = budget
                need = bm.blocks_needed(seq.seq_id, n)
                if need > free:
                    break
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled.append(ScheduledSeq(seq, n))
                budget -= n
                free -= need

        return SchedulerOutput(scheduled, preempted)

    def update(self, scheduled: list[ScheduledSeq], next_tokens: list[int | None]) -> list[Sequence]:
        """Record a finished step. `next_tokens[i]` is the token sampled for `scheduled[i]`,
        None where the step did not reach the sequence's end (mid chunked prefill).

        Returns the sequences that finished; their blocks are already freed.
        """
        finished: list[Sequence] = []
        for s, tok in zip(scheduled, next_tokens, strict=True):
            if (tok is None) == s.samples:
                raise ValueError(f"seq {s.seq.seq_id}: sampled token does not match schedule")
            seq = s.seq
            seq.num_computed_tokens += s.num_tokens
            if tok is None:
                continue
            seq.output_token_ids.append(tok)
            reason = self._stop_reason(seq, tok)
            if reason:
                self._release(seq, reason)
                finished.append(seq)
        if finished:
            self.running = [s for s in self.running if s.status is SequenceStatus.RUNNING]
        return finished

    def finish(self, seq: Sequence, reason: str) -> None:
        """End a running sequence early (e.g. a safety stop); frees its blocks."""
        if seq.status is not SequenceStatus.RUNNING:
            raise ValueError(f"seq {seq.seq_id} is {seq.status.value}, not running")
        self._release(seq, reason)
        self.running.remove(seq)

    def _release(self, seq: Sequence, reason: str) -> None:
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        self.block_manager.free(seq.seq_id)

    def _stop_reason(self, seq: Sequence, tok: int) -> str | None:
        p = seq.params
        if tok in p.stop_token_ids or (not p.ignore_eos and tok in self.eos_token_ids):
            return "stop"
        if len(seq.output_token_ids) >= p.max_tokens:
            return "length"
        return None

    def _preempt(self, seq: Sequence, preempted: list[Sequence]) -> int:
        """Free seq's blocks and requeue it at the front for recompute; returns blocks freed."""
        freed = len(self.block_manager.block_table(seq.seq_id))
        self.block_manager.free(seq.seq_id)
        seq.num_computed_tokens = 0
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1
        preempted.append(seq)
        return freed
