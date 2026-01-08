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


class ModelWithLoss(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, *data):
        data, rewards = data[:-1], data[-1]
        output_data = self.model(*data)
        return self.criterion(output_data[1], rewards)

    def backup(self, *data):
        data, rewards = data[:-1], data[-1]
        output_data = self.model.backup(*data)
        return self.criterion(output_data, rewards)


class ModelWithOptimizer(nn.Module):
    def __init__(self, model_with_loss, optimizer, clip_value=20.0):
        super().__init__()
        self.model_with_loss = model_with_loss
        self.optimizer = optimizer
        self.clip_value = clip_value

    def forward(self, *data):
        self.optimizer.zero_grad(set_to_none=True)
        was_training_wrapper = self.model_with_loss.training
        base_model = getattr(self.model_with_loss, "model", None)
        was_training_model = base_model.training if isinstance(base_model, nn.Module) else None
        if not was_training_wrapper:
            self.model_with_loss.train()
        if isinstance(base_model, nn.Module) and not was_training_model:
            base_model.train()
        try:
            loss = self.model_with_loss.backup(*data)
            loss.backward()
            if self.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model_with_loss.model.parameters(), self.clip_value)
            self.optimizer.step()
        finally:
            if isinstance(base_model, nn.Module) and was_training_model is False:
                base_model.eval()
            if was_training_wrapper is False:
                self.model_with_loss.eval()
        return loss.detach()

