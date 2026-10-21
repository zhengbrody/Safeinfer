from __future__ import annotations

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from safeinfer.memory.block_manager import BlockManager, OutOfBlocksError


def test_slots_follow_block_table():
    bm = BlockManager(num_blocks=8, block_size=4)
    assert bm.grow(0, 6) == [0, 1, 2, 3, 4, 5]  # blocks 0, 1
    bm.grow(1, 1)  # block 2
    # seq 0's 7th and 8th tokens fill block 1; its 9th opens block 3, not 2.
    assert bm.grow(0, 3) == [6, 7, 12]
    assert bm.block_table(0) == [0, 1, 3]
    assert bm.num_tokens(0) == 9


def test_blocks_needed_counts_partial_tail():
    bm = BlockManager(num_blocks=8, block_size=4)
    assert bm.blocks_needed(0, 4) == 1
    assert bm.blocks_needed(0, 5) == 2
    bm.grow(0, 3)
    assert bm.blocks_needed(0, 1) == 0  # fits in the tail of block 0
    assert bm.blocks_needed(0, 2) == 1


def test_out_of_blocks_is_all_or_nothing():
    bm = BlockManager(num_blocks=2, block_size=4)
    bm.grow(0, 5)
    with pytest.raises(OutOfBlocksError):
        bm.grow(1, 1)
    assert bm.num_tokens(1) == 0 and bm.num_free_blocks == 0
    bm.check_invariants()


def test_free_returns_blocks_and_utilization():
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.grow(0, 5)  # 2 blocks, 5 of 8 slots used
    assert bm.utilization() == pytest.approx(5 / 8)
    bm.free(0)
    assert bm.num_free_blocks == 4
    with pytest.raises(KeyError):
        bm.free(0)


class BlockManagerMachine(RuleBasedStateMachine):
    """Random grow/free sequences; a shadow model tracks expected token counts."""

    def __init__(self):
        super().__init__()
        self.bm = BlockManager(num_blocks=16, block_size=4)
        self.expected: dict[int, int] = {}
        self.slots_seen: dict[int, list[int]] = {}

    @rule(seq_id=st.integers(0, 7), n=st.integers(1, 20))
    def grow(self, seq_id, n):
        fits = self.bm.can_grow(seq_id, n)
        try:
            slots = self.bm.grow(seq_id, n)
        except OutOfBlocksError:
            assert not fits
            return
        assert fits
        self.expected[seq_id] = self.expected.get(seq_id, 0) + n
        self.slots_seen.setdefault(seq_id, []).extend(slots)

    @precondition(lambda self: self.expected)
    @rule(data=st.data())
    def free(self, data):
        seq_id = data.draw(st.sampled_from(sorted(self.expected)))
        self.bm.free(seq_id)
        del self.expected[seq_id]
        del self.slots_seen[seq_id]

    @invariant()
    def consistent(self):
        self.bm.check_invariants()
        for seq_id, n in self.expected.items():
            assert self.bm.num_tokens(seq_id) == n
        # No two live tokens share a cache slot.
        live = [s for slots in self.slots_seen.values() for s in slots]
        assert len(live) == len(set(live))


BlockManagerMachine.TestCase.settings = settings(max_examples=300, stateful_step_count=60, deadline=None)
test_block_manager_state_machine = BlockManagerMachine.TestCase
