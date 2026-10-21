"""In-loop safety monitoring: probe scores on every sampled position, from activations the forward already made.

At output position t the engine has the residual stream of the token that
predicts token t (position 0 = the prompt's last token). The output probe
scores every position; a sequence is *flagged* once `consecutive` positions
in a row have a logit at or above the probe threshold, which trades a little
latency for robustness to single-token spikes. With `stop=True` a flagged
sequence ends before token t is emitted.

The optional input probe scores the prompt once, at position 0. It only
reports: a harmful request the model refuses is fine, so stopping on it
would just add over-refusal.

State is keyed by output position, not by forward pass. A preempted
sequence keeps its outputs and resumes at the next position, so recompute
never rescores a position or double-counts the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from safeinfer.safety.probe import LinearProbe


@dataclass(frozen=True)
class SafetySignal:
    logit: float  # output probe at this position
    input_logit: float | None  # input probe on the prompt; position 0 only
    flagged: bool  # `consecutive` positions in a row reached the threshold, here or earlier


@dataclass
class _SeqState:
    next_position: int = 0
    run: int = 0
    flagged: bool = False


class SafetyMonitor:
    def __init__(
        self,
        output_probe: LinearProbe,
        consecutive: int = 1,
        stop: bool = False,
        input_probe: LinearProbe | None = None,
    ):
        if consecutive < 1:
            raise ValueError("consecutive must be >= 1")
        self.output_probe = output_probe
        self.input_probe = input_probe
        self.consecutive = consecutive
        self.stop = stop
        self._state: dict[int, _SeqState] = {}

    @property
    def layers(self) -> tuple[int, ...]:
        """Layers the runner must capture."""
        probes = [self.output_probe] + ([self.input_probe] if self.input_probe else [])
        return tuple(sorted({p.layer for p in probes}))

    def observe(
        self,
        seq_ids: list[int],
        positions: list[int],
        activations: dict[int, torch.Tensor],  # layer -> [len(seq_ids), hidden]
    ) -> list[SafetySignal]:
        """Score one sampled position per sequence; positions must advance by one per call."""
        logits = self.output_probe.logit(activations[self.output_probe.layer]).tolist()
        input_logits: list[float | None] = [None] * len(seq_ids)
        first = [i for i, pos in enumerate(positions) if pos == 0]
        if self.input_probe and first:
            vals = self.input_probe.logit(activations[self.input_probe.layer][first]).tolist()
            for i, v in zip(first, vals, strict=True):
                input_logits[i] = v

        signals = []
        for seq_id, pos, logit, input_logit in zip(seq_ids, positions, logits, input_logits, strict=True):
            st = self._state.setdefault(seq_id, _SeqState())
            if pos != st.next_position:
                raise ValueError(f"seq {seq_id}: observed position {pos}, expected {st.next_position}")
            st.next_position += 1
            st.run = st.run + 1 if logit >= self.output_probe.threshold else 0
            st.flagged = st.flagged or st.run >= self.consecutive
            signals.append(SafetySignal(logit, input_logit, st.flagged))
        return signals

    def release(self, seq_id: int) -> None:
        """Drop a finished sequence's state (no-op if it was never observed)."""
        self._state.pop(seq_id, None)
