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
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn.functional as F

from KTScripts.DataLoader import KTDataset
from KTScripts.utils import set_random_seed
from Scripts.Agent.utils import pl_loss
from Scripts.Envs import KESEnv
from Scripts.Optimizer import ModelWithLoss, ModelWithOptimizer
from Scripts.options import get_options
from Scripts.utils import load_agent, get_data


def _build_model_path(args, run_idx):
    run_suffix = f'_run{run_idx + 1}' if args.num_runs > 1 else ''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    return os.path.join(args.save_dir, args.exp_name + str(args.path) + run_suffix)


def _build_visual_path(args, run_idx):
    run_suffix = f'_run{run_idx + 1}' if args.num_runs > 1 else ''
    if not os.path.exists(args.visual_dir):
        os.makedirs(args.visual_dir)
    return os.path.join(args.visual_dir, f'{args.exp_name}_{args.path}{run_suffix}')


def _configure_optimizer(model, lr):
    return Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def train_one_client(model, env, args, local_epochs):
    model.train()
    criterion = pl_loss
    model_with_loss = ModelWithLoss(model, criterion)
    stage1_epochs = min(args.stage1_epochs or 0, local_epochs)
    stage2_epochs = local_epochs - stage1_epochs
    total_epochs = stage1_epochs + stage2_epochs
    if total_epochs == 0:
        total_epochs = local_epochs
        stage2_epochs = total_epochs
    expert_id = 0
    skill_num, batch_size = args.skill_num, args.batch_size
    steps_per_epoch = 200
    for epoch in range(total_epochs):
        if epoch < stage1_epochs:
            model.freeze_expert_params()
            optimizer = _configure_optimizer(model, args.lr)
        else:
            model.unfreeze_expert_params()
            optimizer = _configure_optimizer(model, args.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: (1 - min(step, 200) / 200) ** 0.5)
        model_train = ModelWithOptimizer(model_with_loss, optimizer)
        for _ in range(steps_per_epoch):
            targets, initial_logs, origin_path = get_data(
                batch_size, skill_num, 3, 10, args.path, args.steps
            )
            targets = targets.to(args.device)
            initial_logs = initial_logs.to(args.device)
            origin_path = origin_path.to(args.device)
            initial_log_scores = env.begin_episode(targets, initial_logs)
            data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
            result = model(*data, expert_id=expert_id)
            scores = env.n_step(result[0], binary=args.binary)
            rewards = env.end_episode()
            if args.concept_exercise_map:
                _, _, routed_experts = env.run_micro_loop(model, result[0], device=args.device)
                expert_id = int(routed_experts[0, -1].item())
            rewards_tensor = (
                rewards.to(args.device)
                if isinstance(rewards, torch.Tensor)
                else torch.tensor(rewards, device=args.device)
            )
            if epoch < stage1_epochs and args.withKT:
                kt_output = result[3]
                kt_targets = scores.to(args.device)
                if kt_targets.dim() == 2:
                    kt_targets = kt_targets.unsqueeze(-1)
                loss = F.binary_cross_entropy(kt_output, kt_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                model_train(*data[:-1], result[2], rewards_tensor)
        scheduler.step()
    sample_count = total_epochs * steps_per_epoch * batch_size
    return model.state_dict(), sample_count


def _train_and_evaluate(args, run_idx, run_seed):
    print('=' * 20 + f' Run {run_idx + 1}/{args.num_runs} ' + '=' * 20)
    if run_seed >= 0:
        print(f'Using random seed: {run_seed}')
    set_random_seed(run_seed)
    dataset = KTDataset(os.path.join(args.data_dir, args.dataset))
    env = KESEnv(
        dataset,
        args.model,
        args.dataset,
        concept_exercise_map=args.concept_exercise_map,
        mastery_threshold=args.mastery_threshold,
    )
    args.skill_num = env.skill_num
    # Create Agent
    model = load_agent(args).to(args.device)
    model_path = _build_model_path(args, run_idx)
    if args.load_model and os.path.exists(f'{model_path}.pt'):
        state_dict = torch.load(f'{model_path}.pt', map_location=args.device)
        model.load_state_dict(state_dict)
        print(f"Load Model From {model_path}")
    model.train()
    criterion = pl_loss
    model_with_loss = ModelWithLoss(model, criterion)
    model_train = None
    scheduler = None
    all_mean_rewards, all_rewards = [], []
    skill_num, batch_size = args.skill_num, args.batch_size
    targets, result = None, None
    best_reward = -1e9
    print('-' * 20 + "Training Start" + '-' * 20)
    
    stage1_epochs = args.stage1_epochs or 0
    stage2_epochs = args.stage2_epochs or 0
    total_epochs = stage1_epochs + stage2_epochs
    if total_epochs == 0:
        total_epochs = args.num_epochs
        stage2_epochs = total_epochs
    expert_id = 0
    for epoch in range(total_epochs):
        if epoch < stage1_epochs:
            model.freeze_expert_params()
            optimizer = _configure_optimizer(model, args.lr)
        else:
            model.unfreeze_expert_params()
            optimizer = _configure_optimizer(model, args.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: (1 - min(step, 200) / 200) ** 0.5)
        model_train = ModelWithOptimizer(model_with_loss, optimizer)
        avg_time = 0
        epoch_mean_rewards = []
        for i in tqdm(range(200)):
            t0 = time.perf_counter()
            targets, initial_logs, origin_path = get_data(batch_size, skill_num, 3, 10, args.path, args.steps)
            targets = targets.to(args.device)
            initial_logs = initial_logs.to(args.device)
            origin_path = origin_path.to(args.device)
            initial_log_scores = env.begin_episode(targets, initial_logs)
            data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
            result = model(*data, expert_id=expert_id)
            scores = env.n_step(result[0], binary=args.binary)
            rewards = env.end_episode()
            if args.concept_exercise_map:
                _, _, routed_experts = env.run_micro_loop(model, result[0], device=args.device)
                expert_id = int(routed_experts[0, -1].item())
            rewards_tensor = rewards.to(args.device) if isinstance(rewards, torch.Tensor) else torch.tensor(rewards, device=args.device)
            if epoch < stage1_epochs and args.withKT:
                kt_output = result[3]
                kt_targets = scores.to(args.device)
                if kt_targets.dim() == 2:
                    kt_targets = kt_targets.unsqueeze(-1)
                loss = F.binary_cross_entropy(kt_output, kt_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = loss.item()
            else:
                loss = model_train(*data[:-1], result[2], rewards_tensor).item()
            mean_reward = rewards_tensor.mean().item()
            avg_time += time.perf_counter() - t0
            epoch_mean_rewards.append(mean_reward)
            all_rewards.append(mean_reward)
            print('Epoch:{}\tbatch:{}\tavg_time:{:.4f}\tloss:{:.4f}\treward:{:.4f}'
                  .format(epoch, i, avg_time / (i + 1), loss, mean_reward))
        print(targets[:10], '\n', result[0][:10])
        all_mean_rewards.append(np.mean(epoch_mean_rewards))
        if all_mean_rewards[-1] > best_reward:
            best_reward = all_mean_rewards[-1]
            torch.save(model.state_dict(), f'{model_path}.pt')
            print("New Best Result Saved!")
        print(f"Best Reward Now:{best_reward:.4f}")
        scheduler.step()
    for i in all_mean_rewards:
        print(i)
    np.save(_build_visual_path(args, run_idx), np.array(all_rewards))

    print('-' * 20 + "Testing Start" + '-' * 20)
    test_rewards = []
    model_with_loss.eval()
    model.load_state_dict(torch.load(f'{model_path}.pt', map_location=args.device))
    expert_id = 0
    for i in tqdm(range(200)):
        targets, initial_logs, origin_path = get_data(batch_size, skill_num, 3, 10, args.path, args.steps)
        targets = targets.to(args.device)
        initial_logs = initial_logs.to(args.device)
        origin_path = origin_path.to(args.device)
        initial_log_scores = env.begin_episode(targets, initial_logs)
        data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
        result = model(*data, expert_id=expert_id)
        env.n_step(result[0], binary=True)
        rewards = env.end_episode()
        if args.concept_exercise_map:
            _, _, routed_experts = env.run_micro_loop(model, result[0], device=args.device)
            expert_id = int(routed_experts[0, -1].item())
        rewards_tensor = rewards.to(args.device) if isinstance(rewards, torch.Tensor) else torch.tensor(rewards, device=args.device)
        loss = criterion(result[1], rewards_tensor).item()
        mean_reward = rewards_tensor.mean().item()
        test_rewards.append(mean_reward)
        print(f'batch:{i}\tloss:{loss:.4f}\treward:{mean_reward:.4f}')
    print(result[0][:10])
    mean_test_reward = np.mean(test_rewards)
    print(f"Mean Reward for Test:{mean_test_reward}")
    return mean_test_reward


if __name__ == '__main__':
    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser, {'agent': 'SRC', 'simulator': 'KES'})
    run_results = []
    for run_idx in range(args_.num_runs):
        if args_.rand_seed >= 0:
            run_seed = args_.rand_seed + run_idx
        else:
            run_seed = args_.rand_seed
        run_results.append(_train_and_evaluate(args_, run_idx, run_seed))
    if len(run_results) > 1:
        print('=' * 20 + ' Summary ' + '=' * 20)
        for idx, reward in enumerate(run_results):
            print(f'Run {idx + 1}: Mean Reward for Test = {reward}')
        print(f'Average Mean Reward over {len(run_results)} runs: {np.mean(run_results)}')
