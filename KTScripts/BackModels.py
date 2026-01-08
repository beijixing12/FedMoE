# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from typing import Callable, Optional, Sequence

import math
import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: Sequence[int],
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.LeakyReLU,
                 bias: bool = True,
                 dropout: float = 0.0):

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        super().__init__(*layers)


def nll_loss(y_, y, mask):
    y_ = y_.reshape(-1, y_.shape[-1]).to(dtype=torch.float32)

    mask = mask.reshape(-1).to(y_.device)
    if mask.dtype != torch.float32:
        mask = mask.to(dtype=torch.float32)

    y = y.reshape(-1)
    if y.dtype != torch.long:
        y = y.to(dtype=torch.long)
    if y.device != y_.device:
        y = y.to(y_.device)

    log_probs = torch.log(y_.clamp_min_(1e-9))
    losses = F.nll_loss(log_probs, y, reduction='none')
    losses = losses * mask
    denom = mask.sum().clamp_min_(1e-9)
    return losses.sum() / denom


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, hidden_sizes, dropout_rate, input_sizes=None):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * 4
        if input_sizes is None:
            input_sizes = hidden_sizes
        for hidden_size in hidden_sizes:
            assert hidden_size % head == 0
        self.head = head
        self.head_size = hidden_sizes[0] // head
        self.hidden_size = hidden_sizes[-1]
        self.d_k = math.sqrt(hidden_sizes[0] // head)
        self.linear_s = nn.ModuleList(
            [nn.Linear(input_size, hidden_size) for (input_size, hidden_size) in zip(input_sizes, hidden_sizes)])
        self.dropout = nn.Dropout(p=dropout_rate)

    def attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        query, key, value = [
            l(x.reshape(-1, x.shape[-1])).view(*x.shape[:2], self.head, self.head_size).transpose(1, 2)
            for l, x in zip(self.linear_s, (query, key, value))]
        x, _ = self.attention(query, key, value, mask)  # (B, Head, L, D_H)
        x = x.transpose(1, 2)
        return self.linear_s[-1](x.reshape(-1, self.head * self.head_size)).view(*x.shape[:2], self.hidden_size)


class FeedForward(nn.Module):
    def __init__(self, head, input_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.mh = MultiHeadedAttention(head, input_size, dropout_rate)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.activate = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)

    def forward(self, s, mask):
        s = s + self.dropout1(self.mh(s, s, s, mask))
        s = self.ln1(s)
        s_ = self.activate(self.fc1(s))
        s_ = self.dropout2(self.fc2(s_))
        s = self.ln2(s + s_)
        return s


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, head=1, b=1, position=False, transformer_mask=True):
        super(Transformer, self).__init__()
        self.position = position
        if position:
            self.pe = PositionalEncoding(input_size, 0.5)
        self.fc = nn.Linear(input_size, hidden_size)
        self.SAs = nn.ModuleList([MultiHeadedAttention(head, hidden_size, dropout_rate) for _ in range(b)])
        self.FFNs = nn.ModuleList([FeedForward(head, hidden_size, dropout_rate) for _ in range(b)])
        self.b = b
        self.transformer_mask = transformer_mask

    def forward(self, inputs, mask=None):
        if self.position:
            inputs = self.pe(inputs)
        inputs = self.fc(inputs)
        max_len = inputs.shape[1]
        if self.transformer_mask:
            mask = torch.tril(torch.ones((1, max_len, max_len), dtype=torch.bool, device=inputs.device))
        elif mask is not None:
            mask = mask.unsqueeze(1)
        if mask is not None:
            mask = mask.unsqueeze(1)
        for i in range(self.b):
            inputs = self.SAs[i](inputs, inputs, inputs, mask)
            inputs = self.FFNs[i](inputs, mask)
        return inputs


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model // 2]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class CoKT(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, head=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.ma_inter = MultiHeadedAttention(head, hidden_size, dropout_rate, input_sizes=(
            hidden_size + input_size - 1, hidden_size + input_size - 1, hidden_size + input_size, hidden_size))
        self.ma_intra = MultiHeadedAttention(head, hidden_size, dropout_rate, input_sizes=(
            input_size - 1, input_size - 1, hidden_size + 1, hidden_size))
        self.wr = nn.Parameter(torch.randn(1, 1, 2))
        self.ln = nn.Linear(2 * hidden_size + input_size - 1, hidden_size)

    def forward(self, intra_x, inter_his, inter_r, intra_mask, inter_len):
        # (B, L, I), (B, L*R, L, I), (B, L, R, I), (B, L), (B, L, R)
        intra_mask = intra_mask.unsqueeze(-1)  # (B, L, 1)

        intra_h, _ = self.rnn(intra_x)  # (B, L, H)
        intra_h_mask = intra_h.masked_select(intra_mask.bool()).view(-1, self.hidden_size)
        intra_x_mask = intra_x.masked_select(intra_mask.bool()).view(-1, self.input_size)
        # inter attention
        intra_mask_ = intra_mask.unsqueeze(-1)  # (B, L, 1, 1)
        inter_his, _ = self.rnn(
            inter_his.view(inter_his.shape[0] * inter_his.shape[1], *inter_his.shape[2:]))
        inter_his = inter_his[torch.arange(inter_his.shape[0], device=inter_len.device), inter_len.view(-1) - 1]
        inter_his = inter_his.view(*inter_len.shape, self.hidden_size)
        inter_his = inter_his.masked_select(intra_mask_.bool()).view(-1, *inter_his.shape[2:])
        inter_r = inter_r.masked_select(intra_mask_.bool()).view(-1, *inter_r.shape[2:])
        M_rv = torch.cat((inter_his, inter_r), -1).view(
            *inter_r.shape[:2], self.hidden_size + self.input_size)
        M_pv = M_rv[:, :, :-1].view(*M_rv.shape[:2], self.input_size + self.hidden_size - 1)
        m_pv = torch.cat((intra_h_mask, intra_x_mask[:, :-1]), 1).view(
            M_pv.shape[0], 1, self.hidden_size + self.input_size - 1)
        v_v = self.ma_inter(m_pv, M_pv, M_rv).squeeze(1)  # (seq_sum, H)
        # intra attention
        intra_x_p = intra_x[:, :, :-1]
        intra_h_p = torch.cat((intra_h, intra_x[:, :, -1:]), -1)
        intra_mask_attn = torch.tril(torch.ones((1, 1, intra_x_p.shape[1], intra_x_p.shape[1]), dtype=torch.bool,
                                               device=intra_x.device))
        v_h = self.ma_intra(intra_x_p, intra_x_p, intra_h_p, mask=intra_mask_attn)
        v_h = v_h.masked_select(intra_mask.bool()).view(-1, v_h.shape[-1])
        weights = torch.softmax(self.wr, dim=-1)
        v = torch.sum(weights * torch.stack((v_v, v_h), -1), -1)
        return self.ln(torch.cat((v, intra_h_mask, intra_x_mask[:, :-1]), 1))

    def deal_inter(self, inter_his, inter_r, inter_len):
        inter_his, _ = self.rnn(
            inter_his.view(inter_his.shape[0] * inter_his.shape[1], *inter_his.shape[2:]))
        inter_his = inter_his[torch.arange(inter_his.shape[0], device=inter_len.device), inter_len.view(-1) - 1]
        inter_his = inter_his.view(*inter_len.shape, self.hidden_size)
        M_rv = torch.cat((inter_his, inter_r), -1).view(
            *inter_r.shape[:3], self.hidden_size + self.input_size)
        M_pv = M_rv[:, :, :-1].view(*M_rv.shape[:3], self.input_size + self.hidden_size - 1)
        return M_rv, M_pv

    def step(self, m_rv, M_pv, intra_x, o, intra_h_p=None):
        # M_*: (B, R, H)
        # intra_h_p:(B, L-1, H+1), with the y
        # intra_x:(B, L, I-1), without the y
        # o: y from last step
        concat_input = torch.cat((intra_x[:, -1:], o), dim=-1)
        if intra_h_p is not None:
            hidden = intra_h_p[:, -1, :-1].unsqueeze(0)
            hidden = hidden.contiguous()
        else:
            hidden = None
        intra_h_next, hidden_next = self.rnn(concat_input, hidden)
        m_pv = torch.cat((intra_h_next, intra_x[:, -1:]), -1)
        v_v = self.ma_inter(m_pv, M_pv, m_rv)

        intra_x_p = intra_x
        intra_h_next = torch.cat((intra_h_next, o), dim=-1)
        intra_h_p = intra_h_next if intra_h_p is None else torch.cat((intra_h_p, intra_h_next), 1)
        v_h = self.ma_intra(intra_x_p[:, -1:], intra_x_p, intra_h_p)
        weights = torch.softmax(self.wr, dim=-1)
        v = torch.sum(weights * torch.stack((v_v, v_h), -1), -1)
        return self.ln(torch.cat((v, intra_h_p[:, -1:, :-1], intra_x[:, -1:]), -1)), intra_h_p


if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoKT(16, 14, 0.5).to(device)
    seq_len_ = list(range(10, 20))
    max_len_ = 50
    for j in range(len(seq_len_)):
        t0 = time.perf_counter()
        seq_len = torch.tensor(seq_len_[:j + 1], device=device)
        mask_ = torch.arange(max_len_, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        x_ = torch.rand(len(seq_len), max_len_, 16, device=device)
        his = torch.rand(len(seq_len), max_len_ * 5, max_len_, 16, device=device)
        r = torch.rand(len(seq_len), max_len_, 5, 16, device=device)
        inter_len_ = torch.randint(1, max_len_, (len(seq_len), max_len_, 5), device=device)
        t1 = time.perf_counter()
        output = model(x_, his, r, mask_, inter_len_)
        print(f"Batch {j} forward took {time.perf_counter() - t1:.4f}s")
