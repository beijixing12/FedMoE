"""Generate concept-to-exercise mapping JSON from an NPZ dataset."""
import argparse
import json
import os
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Union

import numpy as np


def _resolve_npz_key(
    available: Iterable[str],
    primary: str,
    alternatives: Sequence[str],
) -> Optional[str]:
    available_set = set(available)
    if primary in available_set:
        return primary
    for candidate in alternatives:
        if candidate in available_set:
            return candidate
    return None


def _normalise_sequence(values: np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.size >= length:
        return array[:length]
    padded = np.empty(length, dtype=array.dtype)
    padded[: array.size] = array
    padded[array.size :] = -1
    return padded


def _safe_int(value: object) -> Optional[int]:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def build_mapping(npz_path: str) -> Dict[int, List[int]]:
    with np.load(npz_path, allow_pickle=True) as data:
        files = data.files
        skill_key = _resolve_npz_key(files, "skill", ("skills", "skill_seq"))
        if skill_key is None:
            raise KeyError(
                "Dataset archive missing a skill/concept sequence column "
                "(expected 'skill', 'skills', or 'skill_seq')."
            )
        exercise_key = _resolve_npz_key(
            files,
            "question_id",
            ("problem_id", "exercise_id", "item_id", "problemId", "question_id_seq"),
        )
        if exercise_key is None:
            raise KeyError(
                "Dataset archive missing an exercise identifier column "
                "(expected 'question_id', 'problem_id', 'exercise_id', or similar)."
            )
        length_key = _resolve_npz_key(files, "real_len", ("length", "seq_len"))

        raw_skills = data[skill_key]
        raw_exercises = data[exercise_key]
        real_lens = (
            np.asarray(data[length_key], dtype=np.int64).copy() if length_key else None
        )

    skills = [np.asarray(seq).copy() if seq is not None else None for seq in raw_skills]
    exercises = [
        np.asarray(seq).copy() if seq is not None else None for seq in raw_exercises
    ]
    exercises = [
        np.asarray(seq).copy() if seq is not None else None for seq in raw_exercises
    ]

    mapping: MutableMapping[int, set[int]] = {}
    for seq_idx, skill_seq in enumerate(skills):
        if skill_seq is None:
            continue
        seq_len = int(real_lens[seq_idx]) if real_lens is not None else len(skill_seq)
        if seq_len <= 0:
            continue
        skill_values = _normalise_sequence(skill_seq, seq_len)
        exercise_values = _normalise_sequence(exercises[seq_idx], seq_len)

        for pos in range(seq_len):
            concept_id = _safe_int(skill_values[pos])
            exercise_id = _safe_int(exercise_values[pos])
            if concept_id is None or concept_id < 0:
                continue
            if exercise_id is None or exercise_id < 0:
                continue
            mapping.setdefault(concept_id, set()).add(exercise_id)

    return {cid: sorted(exercises_set) for cid, exercises_set in mapping.items()}

def build_error_rates(
    npz_path: str,
) -> tuple[Dict[int, float], Dict[int, int]]:
    with np.load(npz_path, allow_pickle=True) as data:
        files = data.files
        skill_key = _resolve_npz_key(files, "skill", ("skills", "skill_seq"))
        if skill_key is None:
            raise KeyError(
                "Dataset archive missing a skill/concept sequence column "
                "(expected 'skill', 'skills', or 'skill_seq')."
            )
        exercise_key = _resolve_npz_key(
            files,
            "question_id",
            ("problem_id", "exercise_id", "item_id", "problemId", "question_id_seq"),
        )
        if exercise_key is None:
            raise KeyError(
                "Dataset archive missing an exercise identifier column "
                "(expected 'question_id', 'problem_id', 'exercise_id', or similar)."
            )
        label_key = _resolve_npz_key(files, "y", ("label", "labels"))
        if label_key is None:
            raise KeyError(
                "Dataset archive missing a label sequence column "
                "(expected 'y', 'label', or 'labels')."
            )
        length_key = _resolve_npz_key(files, "real_len", ("length", "seq_len"))

        raw_skills = data[skill_key]
        raw_exercises = data[exercise_key]
        raw_labels = data[label_key]
        real_lens = (
            np.asarray(data[length_key], dtype=np.int64).copy() if length_key else None
        )

    skills = [np.asarray(seq).copy() if seq is not None else None for seq in raw_skills]
    exercises = [
        np.asarray(seq).copy() if seq is not None else None for seq in raw_exercises
    ]
    labels = [np.asarray(seq).copy() if seq is not None else None for seq in raw_labels]

    total: MutableMapping[int, int] = {}
    incorrect: MutableMapping[int, int] = {}
    for seq_idx, skill_seq in enumerate(skills):
        if skill_seq is None:
            continue
        seq_len = int(real_lens[seq_idx]) if real_lens is not None else len(skill_seq)
        if seq_len <= 0:
            continue
        skill_values = _normalise_sequence(skill_seq, seq_len)
        exercise_values = _normalise_sequence(exercises[seq_idx], seq_len)
        label_values = _normalise_sequence(labels[seq_idx], seq_len)

        for pos in range(seq_len):
            _ = _safe_int(skill_values[pos])
            exercise_id = _safe_int(exercise_values[pos])
            label_val = _safe_int(label_values[pos])
            if exercise_id is None or exercise_id < 0:
                continue
            if label_val is None:
                continue
            total[exercise_id] = total.get(exercise_id, 0) + 1
            if label_val <= 0:
                incorrect[exercise_id] = incorrect.get(exercise_id, 0) + 1

    error_rates = {
        exercise_id: incorrect.get(exercise_id, 0) / count
        for exercise_id, count in total.items()
        if count > 0
    }
    return error_rates, dict(total)

def _merge_error_rates(
    error_rates: Dict[int, float],
    attempt_counts: Dict[int, int],
) -> Dict[int, Dict[str, Union[float, int]]]:
    merged: Dict[int, Dict[str, Union[float, int]]] = {}
    for exercise_id, error_rate in error_rates.items():
        merged[exercise_id] = {
            "error_rate": error_rate,
            "attempts": attempt_counts.get(exercise_id, 0),
        }
    return merged

def _bucket_counts(values: Sequence[float], k: int) -> tuple[List[float], List[int]]:
    if k <= 0:
        return [], []
    counts = [0 for _ in range(k)]
    width = 1.0 / k
    edges = [(idx + 1) * width for idx in range(k)]
    if not values:
        return edges, counts
    for value in values:
        if value is None:
            continue
        bucket_idx = None
        for idx, upper in enumerate(edges):
            lower = edges[idx - 1] if idx > 0 else float("-inf")
            if value > lower and value <= upper:
                bucket_idx = idx
                break
        if bucket_idx is None:
            if value <= 0:
                bucket_idx = 0
            else:
                bucket_idx = k - 1
        counts[bucket_idx] += 1
    return edges, counts

def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 NPZ 数据集中生成概念-习题映射 JSON 文件。"
    )
    parser.add_argument("--input", required=True, help="输入 NPZ 文件路径。")
    parser.add_argument(
        "--output",
        help="输出 JSON 文件路径（默认与输入同目录，追加 _concept_exercise.json）。",
    )
    parser.add_argument(
        "--include-error-rate",
        default=True,
        action="store_true",
        help="在同一个 JSON 中包含概念错误率（可选）。",
    )

    parser.add_argument(
        "--difficulty-k",
        type=int,
        default=4,
        help="等宽分桶数量（左开右闭），基于习题错误率统计每桶习题数量。",
    )

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_concept_exercise.json"

    mapping = build_mapping(input_path)
    payload: Dict[str, object] = {
        "concept_exercise_map": {str(k): v for k, v in mapping.items()},
    }
    if args.include_error_rate:
        error_rates, attempt_counts = build_error_rates(input_path)
        payload["exercise_error_rate"] = {
            str(k): v for k, v in _merge_error_rates(error_rates, attempt_counts).items()
        }
        if args.difficulty_k > 0:
            edges, counts = _bucket_counts(
                list(error_rates.values()),
                args.difficulty_k,
            )
            payload["difficulty_bins"] = edges
            payload["difficulty_counts"] = counts

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.include_error_rate:
        print(f"已写入概念-习题映射与错误率到 {output_path}")
    else:
        print(f"已写入概念-习题映射到 {output_path}")

if __name__ == "__main__":
    main()

    # python ./generate_concept_exercise_map.py --input /home/zengxiangyu/FedMoE-main/data/assist09/assist09.npz --output _concept_exercise.json
    # python ./generate_concept_exercise_map.py --input /home/zengxiangyu/FedMoE-main/data/assist12/assist12.npz --output assist12.json
    # python ./generate_concept_exercise_map.py --input /home/zengxiangyu/FedMoE-main/data/OLI/OLI.npz --output OLI.json