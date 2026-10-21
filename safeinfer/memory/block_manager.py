"""Block allocator for the paged KV cache.

KV memory is a pool of fixed-size physical blocks (block_size tokens each).
Each sequence owns a block table mapping its logical blocks, in order, to
physical block ids, and grows one block at a time as tokens arrive. Waste is
bounded by the unfilled tail of each sequence's last block.

The manager only does bookkeeping: it never touches tensors. KVCache holds
the memory; the backend reads and writes it through the slots handed out here.
A slot is `physical_block * block_size + offset`, the token's row in the
cache flattened over (block, offset).
"""

from __future__ import annotations

from dataclasses import dataclass, field


class OutOfBlocksError(RuntimeError):
    pass


@dataclass
class _SeqAlloc:
    blocks: list[int] = field(default_factory=list)
    num_tokens: int = 0


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Pop from the end; reversed so blocks are handed out 0, 1, 2, ...
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._seqs: dict[int, _SeqAlloc] = {}

    # --- queries -----------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def num_tokens(self, seq_id: int) -> int:
        """Tokens stored for `seq_id` (0 for an unknown sequence)."""
        alloc = self._seqs.get(seq_id)
        return alloc.num_tokens if alloc else 0

    def block_table(self, seq_id: int) -> list[int]:
        return list(self._seqs[seq_id].blocks)

    def blocks_needed(self, seq_id: int, num_new_tokens: int) -> int:
        """Additional blocks `seq_id` needs to store `num_new_tokens` more tokens."""
        alloc = self._seqs.get(seq_id)
        have_blocks = len(alloc.blocks) if alloc else 0
        total = self.num_tokens(seq_id) + num_new_tokens
        return max(0, -(-total // self.block_size) - have_blocks)

    def can_grow(self, seq_id: int, num_new_tokens: int) -> bool:
        return self.blocks_needed(seq_id, num_new_tokens) <= self.num_free_blocks

    def utilization(self) -> float:
        """Fraction of allocated slots that hold a token (1 - internal fragmentation)."""
        used_blocks = self.num_blocks - self.num_free_blocks
        if used_blocks == 0:
            return 1.0
        stored = sum(a.num_tokens for a in self._seqs.values())
        return stored / (used_blocks * self.block_size)

    # --- mutations ---------------------------------------------------------

    def grow(self, seq_id: int, num_new_tokens: int) -> list[int]:
        """Reserve slots for `num_new_tokens` more tokens of `seq_id`; returns their slots.

        Creates the sequence on first use. All-or-nothing: on OutOfBlocksError
        nothing is allocated.
        """
        if num_new_tokens <= 0:
            raise ValueError("num_new_tokens must be positive")
        need = self.blocks_needed(seq_id, num_new_tokens)
        if need > self.num_free_blocks:
            raise OutOfBlocksError(f"seq {seq_id} needs {need} blocks, {self.num_free_blocks} free")

        alloc = self._seqs.setdefault(seq_id, _SeqAlloc())
        alloc.blocks.extend(self._free.pop() for _ in range(need))
        start = alloc.num_tokens
        alloc.num_tokens += num_new_tokens
        bs = self.block_size
        return [alloc.blocks[pos // bs] * bs + pos % bs for pos in range(start, alloc.num_tokens)]

    def free(self, seq_id: int) -> None:
        """Return all of `seq_id`'s blocks to the pool. Unknown ids are an error."""
        alloc = self._seqs.pop(seq_id)
        self._free.extend(reversed(alloc.blocks))

    # --- testing -----------------------------------------------------------

    def check_invariants(self) -> None:
        owned = [b for a in self._seqs.values() for b in a.blocks]
        assert len(owned) == len(set(owned)), "a block is owned twice"
        assert not set(owned) & set(self._free), "a block is both owned and free"
        assert len(owned) + len(self._free) == self.num_blocks, "blocks leaked"
        for a in self._seqs.values():
            assert len(a.blocks) == -(-a.num_tokens // self.block_size), "table size != ceil(tokens/bs)"
