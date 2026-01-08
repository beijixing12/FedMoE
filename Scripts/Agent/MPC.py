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
import torch
from torch import nn
from collections import OrderedDict
from KTScripts.BackModels import MLP


class MPC(nn.Module):
    def __init__(self, skill_num, input_size, hidden_size, pre_hidden_sizes, dropout, hor):
        super().__init__()
        if getattr(self, "_modules", None) is None:
            nn.Module.__init__(self)
        if getattr(self, "_modules", None) is None:
            object.__setattr__(self, "_parameters", OrderedDict())
            object.__setattr__(self, "_buffers", OrderedDict())
            object.__setattr__(self, "_non_persistent_buffers_set", set())
            object.__setattr__(self, "_backward_hooks", OrderedDict())
            object.__setattr__(self, "_forward_hooks", OrderedDict())
            object.__setattr__(self, "_forward_pre_hooks", OrderedDict())
            object.__setattr__(self, "_state_dict_hooks", OrderedDict())
            object.__setattr__(self, "_load_state_dict_pre_hooks", OrderedDict())
            object.__setattr__(self, "_is_full_backward_hook", False)
            self.training = True
        self.l1 = nn.Sequential(
            nn.Linear(input_size + 1, input_size),
            nn.LeakyReLU(),
            nn.Dropout(p=1 - dropout)
        )
        self.embed = nn.Embedding(skill_num, input_size)
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = MLP(hidden_size, pre_hidden_sizes + [1], dropout=dropout, norm_layer=None)
        self.hor = hor

    def sample(self, b, n):
        candidate_order = torch.rand(b, n, device=self.embed.weight.device)
        candidate_order = torch.sort(candidate_order, dim=-1).indices  # (B*Hor, n)
        return candidate_order

    def test(self, targets, states):  # (B, H) or (B*Hor, H)
        x, _ = self.encoder(targets, states)
        x = x[:, -1]
        x = torch.sigmoid(self.decoder(x).squeeze(-1))
        return x

    def begin_episode(self, targets, initial_logs, initial_log_scores):
        # targets: (B, K), where K is the num of targets in this batch
        targets = self.embed(targets).mean(dim=1, keepdim=True)  # (B, 1, I)
        targets_repeat = targets.repeat(self.hor, 1, 1).view(-1, 1, targets.size(-1))  # (B*Hor, 1, I)
        if initial_logs is not None:
            states = self.step(initial_logs, initial_log_scores, None)
        else:
            zeros = torch.zeros(1, targets.size(0), targets.size(-1), device=targets.device, dtype=targets.dtype)
            states = (zeros, zeros)
        return targets, targets_repeat, states

    def step(self, x, score, states):
        x = self.embed(x)
        if score is not None:
            x = self.l1(torch.cat((x, score.unsqueeze(-1)), -1))
        _, states = self.encoder(x, states)
        return states

    def construct(self, targets, initial_logs, initial_log_scores, origin_path, n):
        targets, targets_repeat, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        unselected = torch.ones_like(origin_path, dtype=torch.bool)
        a1 = torch.arange(targets.shape[0], device=origin_path.device)
        a2 = a1
        a1 = a1.unsqueeze(1).repeat(1, n).repeat(self.hor, 1)  # (B*Hor, n)
        b1 = torch.arange(unselected.shape[0], device=origin_path.device).unsqueeze(1).repeat(1, unselected.shape[1])
        b2 = torch.arange(unselected.shape[1], device=origin_path.device).unsqueeze(0).repeat(unselected.shape[0], 1)
        result_path = []
        target_args = None
        max_len, batch = origin_path.shape[1], targets_repeat.shape[0]
        for i in range(n):
            candidate_args = self.sample(batch, max_len - i)[:, :(n - i)]  # (B*H, n-i)
            if i > 0:
                candidate_args = candidate_args.view(-1, self.hor, n - i)
                candidate_args[:, -1] = target_args
                candidate_args = candidate_args.view(-1, n - i)
            candidate_paths = origin_path.masked_select(unselected)
            candidate_paths = candidate_paths.view(-1, max_len - i)[a1, candidate_args]  # (B*Hor, n-i)
            a1 = a1[:, :-1]
            states_repeat = [state.repeat(1, self.hor, 1) for state in states]
            _, states_repeat = self.encoder(self.embed(candidate_paths), states_repeat)  # (B*Hor, L, H)
            candidate_scores = self.test(targets_repeat, states_repeat).view(-1, self.hor)
            selected_hor = torch.argmax(candidate_scores, dim=1)  # (B,)

            target_args = candidate_args.view(-1, self.hor, n - i)[a2, selected_hor]
            target_path = candidate_paths.view(-1, self.hor, n - i)[a2, selected_hor]
            result_path.append(target_path[:, 0])

            modified = unselected.masked_select(unselected).view(unselected.shape[0], -1)
            modified[a2, target_args[:, 0]] = False
            temp1, temp2 = b1.masked_select(unselected), b2.masked_select(unselected)
            unselected[temp1, temp2] = modified.view(temp1.shape)
            target_args = torch.where(target_args > target_args[:, :1], target_args - 1, target_args)
            target_args = target_args[:, 1:]

            states = self.step(target_path[:, :1], None, states)
        result_path = torch.stack(result_path, dim=1)
        return result_path

    def backup(self, targets, initial_logs, initial_log_scores, result_path):
        _, _, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        history_states, _ = self.encoder(self.embed(result_path), states)
        history_scores = torch.sigmoid(self.decoder(history_states)).squeeze(-1)  # (B, L)
        return history_scores
