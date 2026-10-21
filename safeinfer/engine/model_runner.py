"""Runs one forward step for a set of sequences against the paged KV cache.

The runner turns "these sequences get these new tokens" into a flattened
batch plus AttentionMetadata, runs the model, and returns logits for each
sequence's last token. It does no scheduling: callers decide which sequences
run and must free them when done. The week-3 scheduler sits on top of this
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from safeinfer.attention.backend import AttentionMetadata
from safeinfer.attention.torch_paged import TorchPagedAttention
from safeinfer.memory.block_manager import BlockManager, OutOfBlocksError
from safeinfer.memory.kv_cache import KVCache
from safeinfer.model.llama import LlamaForCausalLM


@dataclass(frozen=True)
class SeqStep:
    seq_id: int
    token_ids: list[int]  # new tokens: the whole prompt, a prompt chunk, or one decoded token


@dataclass
class StepOutput:
    logits: torch.Tensor  # [num_seqs, vocab], fp32, last token of each sequence
    captured: torch.Tensor | None  # [num_seqs, hidden], residual stream after capture_layer


class ModelRunner:
    def __init__(
        self,
        model: LlamaForCausalLM,
        num_blocks: int,
        block_size: int = 16,
        capture_layer: int | None = None,
    ):
        self.model = model
        weight = model.model.embed_tokens.weight
        self.device = weight.device
        self.block_manager = BlockManager(num_blocks, block_size)
        self.kv_cache = KVCache(model.config, num_blocks, block_size, weight.dtype, self.device)
        self.backend = TorchPagedAttention(self.kv_cache)
        self.capture_layer = capture_layer

    def step(self, batch: list[SeqStep]) -> StepOutput:
        """Append each sequence's new tokens to its cache and return next-token logits.

        All-or-nothing: raises OutOfBlocksError before touching any state if
        the batch as a whole does not fit. Sequence ids must be distinct.
        """
        bm = self.block_manager
        need = sum(bm.blocks_needed(s.seq_id, len(s.token_ids)) for s in batch)
        if need > bm.num_free_blocks:
            raise OutOfBlocksError(f"batch needs {need} blocks, {bm.num_free_blocks} free")

        input_ids, positions, slots, query_lens, context_lens = [], [], [], [], []
        for s in batch:
            start = bm.num_tokens(s.seq_id)
            slots += bm.grow(s.seq_id, len(s.token_ids))
            input_ids += s.token_ids
            positions += range(start, start + len(s.token_ids))
            query_lens.append(len(s.token_ids))
            context_lens.append(start + len(s.token_ids))

        tables = [bm.block_table(s.seq_id) for s in batch]
        width = max(len(t) for t in tables)
        block_tables = torch.tensor([t + [0] * (width - len(t)) for t in tables], device=self.device)

        meta = AttentionMetadata(
            query_lens=query_lens,
            context_lens=context_lens,
            slot_mapping=torch.tensor(slots, device=self.device),
            block_tables=block_tables,
        )
        hidden, captured = self.model(
            torch.tensor(input_ids, device=self.device),
            torch.tensor(positions, device=self.device),
            self.backend,
            meta,
            capture_layer=self.capture_layer,
        )
        last = meta.last_token_indices(self.device)
        return StepOutput(
            logits=self.model.compute_logits(hidden[last]),
            captured=None if captured is None else captured[last],
        )

    def free(self, seq_id: int) -> None:
        self.block_manager.free(seq_id)
