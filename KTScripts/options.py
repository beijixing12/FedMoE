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
from argparse import ArgumentParser, Namespace

import torch


def get_exp_configure(model):
    configure_list = {
        'embed_size': 128,
        'hidden_size': 128,
        'output_size': 1,
        'batch_size': 128,
        'dropout': 0.5,
        'decay_step': 100,
        'l2_reg': 1e-4,
        'pre_hidden_sizes': [256, 64, 16],
        'retrieval': False,
        'forRec': False
    }
    if model == 'CoKT':
        configure_list.update({'batch_size': 8, 'retrieval': True, 'embed_size': 48,
                               'hidden_size': 64, 'pre_hidden_sizes': [128, 64, 16]})
    if model == 'GRU4Rec':
        configure_list.update({'forRec': True, 'without_label': True})
    return configure_list

def _has_option(parser: ArgumentParser, option: str) -> bool:
    """Return True if the parser already defines the given option string."""

    option_map = getattr(parser, "_option_string_actions", None)
    if option_map is not None:
        return option in option_map
    return any(option in action.option_strings for action in parser._actions)

def get_options(parser: ArgumentParser, reset_args=None):
    if reset_args is None:
        reset_args = {}
    model = ['DKT', 'Transformer', 'CoKT', 'GRU4Rec']
    dataset = ['junyi', 'assist09', 'assist15', 'assist15', 'OLI']
    parser.add_argument('-m', '--model', type=str, choices=model, default='DKT', help="Model to use")
    parser.add_argument('-d', '--dataset', type=str, choices=dataset, default='OLI', help="Dataset to use")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./SavedModels')
    parser.add_argument('--visual_dir', type=str, default='./VisualResults')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--withKT', action='store_true', default=False, help='Whether to use KT as a secondary task')
    parser.add_argument('--binary', action='store_true', default=False, help='Whether the reward is binary')
    parser.add_argument(
        '--prerequisite_graph',
        type=str,
        default=None,
        help='Path to the prerequisite graph JSON (defaults to common project locations).'
    )
    parser.add_argument(
        '--similarity_graph',
        type=str,
        default=None,
        help='Path to the similarity graph JSON (defaults to common project locations).'
    )
    parser.add_argument('--without_label', action='store_true')
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-e', "--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--valid_step', type=int, default=100)
    parser.add_argument("--postfix", type=str, default='', help="a string appended to the file name of the saved model")
    parser.add_argument("--rand_seed", type=int, default=-1, help="random seed for torch and numpy")
    parser.set_defaults(**reset_args)
    args = parser.parse_args().__dict__
    # Get experiment configuration
    exp_configure = get_exp_configure(args['model'])
    args.update(exp_configure)
    args = Namespace(**args)
    args.exp_name = '_'.join([args.model, args.dataset])
    if args.without_label:
        args.exp_name += '_without'
    if args.postfix != '':
        args.exp_name += f'_{args.postfix}'
    if args.cuda >= 0 and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.cuda}')
    else:
        args.device = torch.device('cpu')
    return args
