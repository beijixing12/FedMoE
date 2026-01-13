#!/usr/bin/env python3
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
"""Split a KT dataset archive into school-specific NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _resolve_npz_key(available: Iterable[str], primary: str, alternatives: Sequence[str]) -> str:
    available_set = list(available)
    if primary in available_set:
        return primary
    for candidate in alternatives:
        if candidate in available_set:
            return candidate
    raise KeyError(
        f"Missing required field '{primary}'. Available fields: {sorted(available_set)}"
    )


def _load_dataset(
    npz_path: Path,
) -> Tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(npz_path, allow_pickle=True) as data:
        files = set(data.files)
        skill_key = _resolve_npz_key(files, "skill", ["skills", "skill_seq", "concept_ids"])
        label_key = _resolve_npz_key(files, "y", ["label", "labels"])
        length_key = _resolve_npz_key(files, "real_len", ["real_length", "seq_len", "length"])
        user_key = None
        for candidate in ("user_id", "user_ids", "uid", "user"):
            if candidate in files:
                user_key = candidate
                break
        skills = data[skill_key]
        labels = data[label_key]
        lengths = data[length_key]
        user_ids = data[user_key] if user_key else None
    return skill_key, label_key, length_key, skills, labels, lengths, user_ids


def _load_mapping(mapping_path: Path) -> Mapping[str, List[str]]:
    if mapping_path.suffix.lower() == ".json":
        with mapping_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if all(isinstance(value, list) for value in payload.values()):
                return {str(k): [str(v) for v in values] for k, values in payload.items()}
            if all(isinstance(value, str) for value in payload.values()):
                by_school: Dict[str, List[str]] = {}
                for user_id, school in payload.items():
                    by_school.setdefault(str(school), []).append(str(user_id))
                return by_school
        raise ValueError("JSON mapping must be {school: [user_ids]} or {user_id: school}.")
    if mapping_path.suffix.lower() == ".csv":
        by_school: Dict[str, List[str]] = {}
        with mapping_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "user_id" not in reader.fieldnames or "school" not in reader.fieldnames:
                raise ValueError("CSV mapping must include 'user_id' and 'school' columns.")
            for row in reader:
                by_school.setdefault(str(row["school"]), []).append(str(row["user_id"]))
        return by_school
    raise ValueError("Mapping file must be .json or .csv")


def _index_users(
    user_ids: np.ndarray | None,
    count: int,
) -> Dict[str, int]:
    if user_ids is None:
        return {str(idx): idx for idx in range(count)}
    return {str(user_id): idx for idx, user_id in enumerate(user_ids)}


def _select_rows(values: np.ndarray, indices: List[int]) -> np.ndarray:
    values_array = np.asarray(values)
    return values_array[indices]


def _write_school_npz(
    output_dir: Path,
    school: str,
    skills: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    user_ids: np.ndarray | None,
    skill_key: str,
    label_key: str,
    length_key: str,
    user_key: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{school}.npz"
    payload = {
        skill_key: skills,
        label_key: labels,
        length_key: lengths,
    }
    if user_ids is not None and user_key is not None:
        payload[user_key] = user_ids
    np.savez_compressed(output_path, **payload)


def split_dataset_by_school(npz_path: Path, mapping_path: Path, output_dir: Path) -> None:
    skill_key, label_key, length_key, skills, labels, lengths, user_ids = _load_dataset(npz_path)
    user_key = None
    if user_ids is not None:
        user_key = "user_id" if user_ids.ndim == 1 else "user_ids"
    mapping = _load_mapping(mapping_path)
    user_index = _index_users(user_ids, len(skills))
    for school, users in mapping.items():
        indices = []
        for user_id in users:
            if user_id not in user_index:
                raise KeyError(f"User '{user_id}' not found in dataset.")
            indices.append(user_index[user_id])
        if not indices:
            continue
        school_skills = _select_rows(skills, indices)
        school_labels = _select_rows(labels, indices)
        school_lengths = _select_rows(lengths, indices)
        school_user_ids = _select_rows(user_ids, indices) if user_ids is not None else None
        _write_school_npz(
            output_dir,
            school,
            school_skills,
            school_labels,
            school_lengths,
            school_user_ids,
            skill_key,
            label_key,
            length_key,
            user_key,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset NPZ by school mapping.")
    parser.add_argument("--input", required=True, help="Path to the dataset .npz archive")
    parser.add_argument("--mapping", required=True, help="JSON/CSV mapping of user_id to school")
    parser.add_argument("--output_dir", required=True, help="Directory to write per-school .npz files")
    args = parser.parse_args()
    npz_path = Path(args.input)
    mapping_path = Path(args.mapping)
    output_dir = Path(args.output_dir)
    split_dataset_by_school(npz_path, mapping_path, output_dir)


if __name__ == "__main__":
    main()
