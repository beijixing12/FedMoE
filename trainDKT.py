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
from argparse import Namespace
import shutil
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import BCELoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from KTScripts.BackModels import nll_loss
from KTScripts.DataLoader import KTDataset, RecDataset, RetrievalDataset
from KTScripts.PredictModel import ModelWithLoss, ModelWithLossMask, ModelWithOptimizer
from KTScripts.utils import set_random_seed, load_model, evaluate_utils


def main(args: Namespace):
    print()
    set_random_seed(args.rand_seed)
    dataset_cls = RecDataset if args.forRec else (RetrievalDataset if args.retrieval else KTDataset)
    dataset = dataset_cls(os.path.join(args.data_dir, args.dataset))
    args.feat_nums, args.user_nums = dataset.feats_num, dataset.users_num
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.rand_seed if args.rand_seed >= 0 else torch.seed())
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if args.forRec:
        args.output_size = args.feat_nums
    # Model
    model = load_model(args).to(args.device)
    model_path = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.load_model and os.path.exists(f'{model_path}.pt'):
        model.load_state_dict(torch.load(f'{model_path}.pt', map_location=args.device))
        print(f"Load Model From {model_path}")
    # Optimizer
    steps_per_epoch = max(len(train_loader) // 10 + 1, 1)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: (1 - min(step, steps_per_epoch) / steps_per_epoch) ** 0.5)
    if args.forRec:
        model_with_loss = ModelWithLossMask(model, nll_loss)
    else:
        model_with_loss = ModelWithLoss(model, BCELoss(reduction='mean'))
    model_train = ModelWithOptimizer(model_with_loss, optimizer, args.forRec)
    best_val_auc = 0
    print('-' * 20 + "Training Start" + '-' * 20)
    for epoch in range(args.num_epochs):
        avg_time = 0
        model_train.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            t0 = time.perf_counter()
            batch = [torch.tensor(item, device=args.device) if not torch.is_tensor(item) else item.to(args.device)
                     for item in data]
            loss, output_data = model_train(*batch)
            loss = loss.item()
            acc, auc = evaluate_utils(*output_data)
            avg_time += time.perf_counter() - t0
            print('Epoch:{}\tbatch:{}\tavg_time:{:.4f}\tloss:{:.4f}\tacc:{:.4f}\tauc:{:.4f}'
                  .format(epoch, i, avg_time / (i + 1), loss, acc, auc))
        print('-' * 20 + "Validating Start" + '-' * 20)
        val_eval = [[], []]
        loss_total, data_total = 0, 0
        model_with_loss.eval()
        with torch.no_grad():
            for data in tqdm(test_loader, total=len(test_loader)):
                batch = [torch.tensor(item, device=args.device) if not torch.is_tensor(item) else item.to(args.device)
                         for item in data]
                loss, output_data = model_with_loss.output(*batch)
                val_eval[0].append(output_data[0].detach().cpu().numpy())
                val_eval[1].append(output_data[1].detach().cpu().numpy())
                loss_total += loss.item() * len(batch[0])
                data_total += len(batch[0])
        val_eval = [np.concatenate(_) for _ in val_eval]
        acc, auc = evaluate_utils(*val_eval)
        print(f"Validating loss:{loss_total / data_total:.4f} acc:{acc:.4f} auc:{auc:.4f}")
        if auc >= best_val_auc:
            best_val_auc = auc
            best_model_file = f'{model_path}.pt'
            torch.save(model.state_dict(), best_model_file)
            print("New best result Saved!")
            meta_dir = os.path.join(os.path.dirname(__file__), 'Scripts', 'Envs', 'KES', 'meta_data')
            os.makedirs(meta_dir, exist_ok=True)
            meta_model_path = os.path.join(meta_dir, f'{args.exp_name}.pt')
            shutil.copyfile(best_model_file, meta_model_path)
            print(f"Model copied to {meta_model_path}")
        print(f"Best Auc Now:{best_val_auc:.4f}")
        scheduler.step()

    print('-' * 20 + "Testing Start" + '-' * 20)
    val_eval = [[], []]
    loss_total, data_total = 0, 0
    model_with_loss.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            batch = [torch.tensor(item, device=args.device) if not torch.is_tensor(item) else item.to(args.device)
                     for item in data]
            loss, output_data = model_with_loss.output(*batch)
            val_eval[0].append(output_data[0].detach().cpu().numpy())
            val_eval[1].append(output_data[1].detach().cpu().numpy())
            loss_total += loss.item() * len(batch[0])
            data_total += len(batch[0])
    val_eval = [np.concatenate(_) for _ in val_eval]
    print(val_eval[0], val_eval[0].mean())
    print(val_eval[1], val_eval[1].mean())
    acc, auc = evaluate_utils(*val_eval)
    print(f"Testing loss:{loss_total / data_total:.4f} acc:{acc:.4f} auc:{auc:.4f}")


if __name__ == '__main__':
    from argparse import ArgumentParser

    from KTScripts.options import get_options

    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser)
    main(args_)
