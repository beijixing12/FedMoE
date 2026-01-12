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
import copy
from argparse import ArgumentParser
from typing import Dict, Iterable, List

import torch

from Scripts.Envs import KESEnv
from Scripts.federated_server import (
    build_difficulty_prototypes_from_payload,
    update_prototypes_ema,
)
from Scripts.options import get_options
from Scripts.utils import get_data, load_agent
from Scripts.Agent.utils import pl_loss


def fedavg_state_dicts(state_dicts: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    state_dicts = list(state_dicts)
    if not state_dicts:
        raise ValueError("No state dicts provided for FedAvg.")
    averaged: Dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        averaged[key] = stacked.mean(dim=0)
    return averaged


def local_train_step(model, env, args, expert_id: int) -> int:
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    for _ in range(args.local_steps):
        targets, initial_logs, origin_path = get_data(
            args.batch_size, args.skill_num, 3, 10, args.path, args.steps
        )
        targets = targets.to(args.device)
        initial_logs = initial_logs.to(args.device)
        origin_path = origin_path.to(args.device)
        initial_log_scores = env.begin_episode(targets, initial_logs)
        data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
        result = model(*data, expert_id=expert_id)
        rewards = env.end_episode()
        rewards_tensor = rewards.to(args.device) if isinstance(rewards, torch.Tensor) else torch.tensor(
            rewards, device=args.device
        )
        loss = pl_loss(result[2], rewards_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.concept_exercise_map:
            _, _, routed_experts = env.run_micro_loop(model, result[0], device=args.device)
            expert_id = int(routed_experts[0, -1].item())
    return expert_id


def build_server_prototypes(model, payload_path: str) -> torch.Tensor:
    embeddings = model.graph_encoder()
    prototype_state = build_difficulty_prototypes_from_payload(embeddings, payload_path)
    return prototype_state.prototypes


def run_federated(args):
    from KTScripts.DataLoader import KTDataset

    if not args.data_dir or not args.dataset:
        raise ValueError("Both data_dir and dataset are required for federated training.")
    dataset = KTDataset(f"{args.data_dir}/{args.dataset}")
    envs: List[KESEnv] = []
    models = []
    for _ in range(args.clients):
        env = KESEnv(
            dataset,
            args.model,
            args.dataset,
            concept_exercise_map=args.concept_exercise_map,
            mastery_threshold=args.mastery_threshold,
        )
        args.skill_num = env.skill_num
        model = load_agent(args).to(args.device)
        envs.append(env)
        models.append(model)
    server_model = copy.deepcopy(models[0])
    server_prototypes = server_model.prototypes.detach().clone()
    for round_idx in range(args.rounds):
        if args.difficulty_payload:
            prototypes = build_server_prototypes(server_model, args.difficulty_payload)
            server_prototypes = update_prototypes_ema(
                server_prototypes,
                prototypes,
                stage=args.stage,
                mu_stage1=args.mu_stage1,
                mu_stage2=args.mu_stage2,
                freeze_stage2=args.freeze_stage2,
            )
            for model in models:
                model.update_prototypes(server_prototypes)
        expert_id = 0
        client_shared_states = []
        for model, env in zip(models, envs):
            expert_id = local_train_step(model, env, args, expert_id)
            client_shared_states.append(model.get_shared_state_dict())
        aggregated = fedavg_state_dicts(client_shared_states)
        for model in models:
            model.load_shared_state_dict(aggregated)
        server_model.load_shared_state_dict(aggregated)
        if args.difficulty_payload:
            mu = args.mu_stage2 if args.stage != "stage1" else args.mu_stage1
            print(
                f"Completed round {round_idx + 1}/{args.rounds} "
                f"(stage={args.stage}, mu={mu:.3f})"
            )
        else:
            print(f"Completed round {round_idx + 1}/{args.rounds}")


if __name__ == "__main__":
    parser = ArgumentParser("FedMoE Training")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--local_steps", type=int, default=1)
    parser.add_argument("--difficulty_payload", type=str, default=None)
    parser.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--mu_stage1", type=float, default=0.2)
    parser.add_argument("--mu_stage2", type=float, default=0.8)
    parser.add_argument("--freeze_stage2", action="store_true", default=False)
    args = get_options(parser, {"agent": "SRC", "simulator": "KES"})
    run_federated(args)
