from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from Scripts.Agent.mamba4rec import MambaBlock


@dataclass
class MambaState:
    """Light-weight container for cached inputs passed between decoding steps."""

    cached_inputs: Optional[torch.Tensor]

    def clone(self) -> "MambaState":
        if self.cached_inputs is None:
            return MambaState(None)
        return MambaState(self.cached_inputs.clone())


class MambaSequenceModel(nn.Module):
    """Wrap :class:`MambaBlock` to expose an RNN-like interface.

    The vanilla :class:`MambaBlock` processes a full sequence at once.  The SRC
    agent, however, expects recurrent modules with ``forward`` returning both
    the hidden sequence and an updated state that can be reused in the next
    decoding step.  ``MambaSequenceModel`` mimics the required behaviour by
    caching the projected inputs that have been seen so far and concatenating
    them with the next mini sequence before feeding them through the block.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = (
            nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.mamba = MambaBlock(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> MambaState:
        del batch_size, device, dtype
        return MambaState(None)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def forward(
        self,
        inputs: torch.Tensor,
        state: Optional[MambaState] = None,
    ) -> tuple[torch.Tensor, MambaState]:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        projected = self._project(inputs)
        cached = state.cached_inputs if state is not None else None
        if cached is not None:
            combined = torch.cat((cached, projected), dim=1)
        else:
            combined = projected

        outputs = self.mamba(combined)
        step_outputs = outputs[:, -projected.shape[1] :, :]
        step_outputs = self.layer_norm(step_outputs)

        new_state = MambaState(combined.detach())
        return step_outputs, new_state


def adapt_mamba_state(state: Optional[MambaState]) -> Optional[MambaState]:
    if state is None or isinstance(state, MambaState):
        return state
    if isinstance(state, dict):
        cached = state.get("cached_inputs")
        return MambaState(cached)
    raise TypeError(f"Unsupported state type for MambaSequenceModel: {type(state)!r}")