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
import os

import torch


from KTScripts.options import get_exp_configure
from KTScripts.utils import load_model


def load_d_agent(model_name, dataset_name, skill_num, with_label=True):
    model_parameters = get_exp_configure(model_name)
    model_parameters.update({'feat_nums': skill_num, 'model': model_name, 'without_label': not with_label})
    if model_name == 'GRU4Rec':
        model_parameters.update({'output_size': skill_num})
    model = load_model(model_parameters)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = '/home/zengxiangyu/SRC-py/SavedModels'
    model_path = os.path.join(model_folder, f'{model_name}_{dataset_name}')
    if not with_label:
        model_path += '_without'
    checkpoint_file = f'{model_path}.pt'
    if not os.path.exists(checkpoint_file):
        print(
            f"Warning: diagnoser checkpoint '{checkpoint_file}' was not found. "
            "Proceeding with randomly initialised weights."
        )
        model.eval()
        return model

    state_dict = torch.load(checkpoint_file, map_location='cpu')
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        message = str(exc)
        if 'size mismatch' not in message:
            raise

        print(
            'Warning: checkpoint parameter sizes do not match the current '
            'diagnoser architecture. Attempting to reconcile shapes.'
        )

        current_state = model.state_dict()
        reconciled_state = {}

        for name, param in current_state.items():
            if name not in state_dict:
                # Keep the randomly initialised parameter when missing.
                reconciled_state[name] = param
                continue

            saved_param = state_dict[name]
            if saved_param.shape == param.shape:
                reconciled_state[name] = saved_param
                continue

            if (
                saved_param.dim() == param.dim() == 2
                and saved_param.size(1) == param.size(1)
            ):
                # Resize embeddings/linear weights along the vocab dimension.
                rows = min(saved_param.size(0), param.size(0))
                updated = param.clone()
                updated[:rows] = saved_param[:rows]
                reconciled_state[name] = updated
                continue

            if saved_param.dim() == param.dim() == 1:
                length = min(saved_param.size(0), param.size(0))
                updated = param.clone()
                updated[:length] = saved_param[:length]
                reconciled_state[name] = updated
                continue

            # Fallback: keep the parameter initialisation when we cannot
            # sensibly reshape the stored tensor.
            reconciled_state[name] = param

        model.load_state_dict(reconciled_state, strict=False)
    model.eval()
    return model


def episode_reward(initial_score, final_score, full_score) -> (int, float):
    delta = final_score - initial_score
    normalize_factor = full_score - initial_score + 1e-9
    return delta / normalize_factor
