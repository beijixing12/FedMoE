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
from typing import Dict, List

import torch

from Scripts.Envs import KESEnv
from Scripts.federated_server import (
    build_difficulty_prototypes_from_payload,
    update_prototypes_ema,
)
from Scripts.options import get_options
from Scripts.utils import get_data, load_agent


def _estimate_state_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    total = 0
    for value in state_dict.values():
        total += value.numel() * value.element_size()
    return total


def _collect_routing_distribution(model, env, args, batches: int) -> List[float]:
    model.eval()
    expert_ids = []
    with torch.no_grad():
        for _ in range(batches):
            targets, initial_logs, origin_path = get_data(
                args.batch_size, args.skill_num, 3, 10, args.path, args.steps
            )
            targets = targets.to(args.device)
            initial_logs = initial_logs.to(args.device)
            origin_path = origin_path.to(args.device)
            initial_log_scores = env.begin_episode(targets, initial_logs)
            data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
            model(*data, expert_id=0)
            state_output = model._latest_state_output
            if state_output is None:
                continue
            if state_output.dim() == 3:
                state_output = state_output[:, -1, :]
            if state_output.dim() != 2:
                continue
            proto_dim = model.prototypes.shape[1]
            state_dim = state_output.shape[1]
            if state_dim > proto_dim:
                state_output = state_output[:, :proto_dim]
            elif state_dim < proto_dim:
                pad = torch.zeros(
                    state_output.shape[0],
                    proto_dim - state_dim,
                    device=state_output.device,
                    dtype=state_output.dtype,
                )
                state_output = torch.cat([state_output, pad], dim=1)
            routed = model.route_expert_with_state(state_output).detach().cpu()
            expert_ids.append(routed)
    if not expert_ids:
        return []
    stacked = torch.cat(expert_ids, dim=0)
    counts = torch.bincount(stacked, minlength=model.num_experts).float()
    distribution = (counts / counts.sum()).tolist()
    return distribution


def _evaluate_round(model, env, args, batches: int) -> float:
    model.eval()
    rewards = []
    with torch.no_grad():
        for _ in range(batches):
            targets, initial_logs, origin_path = get_data(
                args.batch_size, args.skill_num, 3, 10, args.path, args.steps
            )
            targets = targets.to(args.device)
            initial_logs = initial_logs.to(args.device)
            origin_path = origin_path.to(args.device)
            initial_log_scores = env.begin_episode(targets, initial_logs)
            data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
            result = model(*data, expert_id=0)
            env.n_step(result[0], binary=args.binary)
            rewards_tensor = env.end_episode()
            if isinstance(rewards_tensor, torch.Tensor):
                rewards.append(rewards_tensor.mean().item())
            else:
                rewards.append(torch.tensor(rewards_tensor).mean().item())
    return sum(rewards) / max(len(rewards), 1)


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
                stage="stage2",
                mu_stage1=args.mu_stage1,
                mu_stage2=args.mu_stage2,
                freeze_stage2=False,
            )
            for model in models:
                model.update_prototypes(server_prototypes)
        shared_state = server_model.state_dict()
        comm_bytes = _estimate_state_bytes(shared_state)
        for model in models:
            model.load_state_dict(shared_state)
        reward_mean = _evaluate_round(server_model, envs[0], args, args.eval_batches)
        routing_distribution = _collect_routing_distribution(
            server_model, envs[0], args, args.routing_batches
        )
        print(
            f"Completed round {round_idx + 1}/{args.rounds} "
            f"(reward={reward_mean:.4f}, comm_bytes={comm_bytes}, local_training=disabled)"
        )
        print(f"Routing distribution: {routing_distribution}")


if __name__ == "__main__":
    parser = ArgumentParser("FedMoE Training")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--difficulty_payload", type=str, default=None)
    parser.add_argument("--eval_batches", type=int, default=5)
    parser.add_argument("--routing_batches", type=int, default=5)
    parser.add_argument("--mu_stage1", type=float, default=0.2)
    parser.add_argument("--mu_stage2", type=float, default=0.8)
    args = get_options(parser, {"agent": "SRC", "simulator": "KES"})
    run_federated(args)
