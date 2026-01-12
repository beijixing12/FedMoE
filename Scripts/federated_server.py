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
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch


@dataclass
class DifficultyPrototypeState:
    bins: List[float]
    buckets: List[List[int]]
    prototypes: torch.Tensor
    stage: Optional[str] = None
    ema_mu: Optional[float] = None


def load_difficulty_payload(payload_path: str | Path) -> Dict[str, object]:
    path = Path(payload_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object with difficulty metadata.")
    return payload


def bucket_exercises_by_error_rate(
    difficulty_bins: Sequence[float],
    exercise_error_rate: Dict[str, Dict[str, float]],
) -> List[List[int]]:
    bins = sorted(float(b) for b in difficulty_bins)
    buckets: List[List[int]] = [[] for _ in bins]
    for exercise_id, stats in exercise_error_rate.items():
        error_rate = float(stats.get("error_rate", 0.0))
        bucket_idx = next((i for i, edge in enumerate(bins) if error_rate <= edge), len(bins) - 1)
        buckets[bucket_idx].append(int(exercise_id))
    return buckets


def build_bucket_prototypes(
    embeddings: torch.Tensor,
    buckets: Iterable[Iterable[int]],
) -> torch.Tensor:
    prototype_list = []
    for bucket in buckets:
        bucket_ids = list(bucket)
        if not bucket_ids:
            prototype_list.append(torch.zeros(embeddings.shape[-1], device=embeddings.device))
            continue
        bucket_embeddings = embeddings[torch.tensor(bucket_ids, device=embeddings.device)]
        prototype_list.append(bucket_embeddings.mean(dim=0))
    return torch.stack(prototype_list, dim=0)


def build_difficulty_prototypes(
    embeddings: torch.Tensor,
    difficulty_bins: Sequence[float],
    exercise_error_rate: Dict[str, Dict[str, float]],
) -> DifficultyPrototypeState:
    if difficulty_bins is None or exercise_error_rate is None:
        raise ValueError("Difficulty bins and exercise error rates are required.")
    buckets = bucket_exercises_by_error_rate(difficulty_bins, exercise_error_rate)
    prototypes = build_bucket_prototypes(embeddings, buckets)
    return DifficultyPrototypeState(bins=list(difficulty_bins), buckets=buckets, prototypes=prototypes)


def build_difficulty_prototypes_from_payload(
    embeddings: torch.Tensor,
    payload_path: str | Path,
) -> DifficultyPrototypeState:
    payload = load_difficulty_payload(payload_path)
    difficulty_bins = payload.get("difficulty_bins")
    exercise_error_rate = payload.get("exercise_error_rate")
    if difficulty_bins is None or exercise_error_rate is None:
        raise ValueError("Payload must include difficulty_bins and exercise_error_rate.")
    return build_difficulty_prototypes(embeddings, difficulty_bins, exercise_error_rate)


def update_prototypes_ema(
    previous_prototypes: torch.Tensor,
    new_prototypes: torch.Tensor,
    *,
    stage: Optional[str] = None,
    mu_stage1: float = 0.2,
    mu_stage2: float = 0.8,
    freeze_stage2: bool = False,
) -> torch.Tensor:
    if stage is not None and stage.lower() == "stage2" and freeze_stage2:
        return previous_prototypes
    if stage is None:
        mu = mu_stage2
    else:
        mu = mu_stage1 if stage.lower() == "stage1" else mu_stage2
    return previous_prototypes * mu + new_prototypes * (1 - mu)
