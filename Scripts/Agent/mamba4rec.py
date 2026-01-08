from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_impl
except Exception:  # pragma: no cover - runtime fallback when optional dep missing/incompatible

    def _selective_scan_impl(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: torch.Tensor,
        delta_bias: Optional[torch.Tensor] = None,
        delta_softplus: bool = False,
    ) -> torch.Tensor:
        """Minimal PyTorch selective scan used when the CUDA kernel is unavailable."""

        batch, d_inner, seq_len = u.shape
        d_state = A.shape[1]

        u_dtype = u.dtype
        A = A.to(u_dtype)
        D = D.to(u_dtype)
        delta = delta.to(u_dtype)

        if delta_bias is not None:
            delta = delta + delta_bias.view(1, -1, 1).to(delta.dtype)
        if delta_softplus:
            delta = F.softplus(delta)

        h = u.new_zeros((batch, d_inner, d_state))
        outputs = []

        A_expander = A.unsqueeze(0)
        inv_A = torch.where(A != 0, 1.0 / A, torch.zeros_like(A)).unsqueeze(0)
        D_expanded = D.view(1, d_inner)

        for t in range(seq_len):
            dt = delta[:, :, t].unsqueeze(-1)
            u_t = u[:, :, t].unsqueeze(-1)

            B_t = B[:, :, t].unsqueeze(1).expand(-1, d_inner, -1)
            C_t = C[:, :, t].unsqueeze(1).expand(-1, d_inner, -1)

            A_dt = torch.exp(dt * A_expander)
            delta_B_u = (A_dt - 1.0) * inv_A * (B_t * u_t)
            h = A_dt * h + delta_B_u

            y_t = (C_t * h).sum(-1) + D_expanded * u[:, :, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=-1)
        y = y * torch.sigmoid(z)
        return y

    selective_scan_fn = _selective_scan_impl
else:
    selective_scan_fn = _selective_scan_impl

from einops import rearrange, repeat

try:  # Optional acceleration from causal-conv1d
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:  # pragma: no cover - best effort import
    causal_conv1d_fn, causal_conv1d_update = None, None


class Mamba4Rec(nn.Module):
    def __init__(
        self,
        items_num: int,
        hidden_size: int,
        d_state: int,
        d_conv: int,
        expand: int,
        num_layers: int,
        dropout_prob: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.item_embedding = nn.Embedding(items_num + 1, hidden_size, padding_idx=0)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

        self.mamba_layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout_prob,
                    num_layers=num_layers,
                )
                for _ in range(num_layers)
            ]
        )

        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def gather_indexes(self, output: torch.Tensor, gather_index: torch.Tensor) -> torch.Tensor:
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(
        self, item_seq: torch.Tensor, item_seq_len: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, labels)
        return loss

    def full_sort_predict(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


class MambaLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_conv * 4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.mamba(x)
        if self.num_layers == 1:
            hidden = self.LayerNorm(self.dropout(hidden))
        else:
            hidden = self.LayerNorm(self.dropout(hidden) + x)
        return self.ffn(hidden)


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(in_features=self.d_model, out_features=self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            in_features=self.d_inner,
            out_features=self.dt_rank + self.d_state * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(in_features=self.dt_rank, out_features=self.d_inner)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"{dt_init} initialization method not implemented")

        dt = torch.exp(
            torch.randn(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(in_features=self.d_inner, out_features=self.d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(x, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seq_len,
        )

        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        x, z = xz.chunk(2, dim=1)

        if causal_conv1d_fn is not None:
            x = self.act(self.conv1d(x)[..., :seq_len])
        else:
            assert self.activation in ["silu", "swish"], (
                f"{self.activation} is not supported in causal-conv1d, please use 'silu' or 'swish'"
            )
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seq_len)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seq_len).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seq_len).contiguous()

        y = selective_scan_fn(
            u=x,
            delta=dt,
            A=A,
            B=B,
            C=C,
            D=self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, inner_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, inner_size)
        self.fc2 = nn.Linear(inner_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = gelu
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.gelu(self.fc1(x))
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden + x)
        return hidden


__all__ = [
    "FeedForward",
    "Mamba4Rec",
    "MambaBlock",
    "MambaLayer",
]