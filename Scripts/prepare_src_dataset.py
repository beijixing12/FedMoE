"""Utility script to merge train/validation/test splits into the SRC input format.

The raw data files are expected to follow the layout showcased below::

    user_id,school_id,seq_len;question_seq,num_attempt_seq,correctness_seq,
    use_time_first_seq,use_time_seq,num_hint_seq,mask_seq,learning_goals
    4029
    7
    19
    10267,10248,10247,...
    ...

Each record therefore contains three scalar fields (``user_id``, ``school_id``
and ``seq_len``) followed by eight sequence fields. The script merges the three
splits, optionally remaps question identifiers to a contiguous range, and
creates an ``npz`` archive that can be consumed by :class:`KTScripts.DataLoader.KTDataset`.

Example usage::

    python Scripts/prepare_src_dataset.py \
        --train ./train.txt \
        --valid ./valid.txt \
        --test ./test.txt \
        --output-dir ./data/custom_dataset \
        --dataset-name custom_dataset

The command above produces ``./data/custom_dataset/custom_dataset.npz`` and a
JSON file containing the identifier mapping used during preprocessing.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass



@dataclass
class Record:
    """Container for a single student's interaction sequence."""

    user_id: int
    school_id: int
    seq_len: int
    question_seq: np.ndarray
    num_attempt_seq: np.ndarray
    correctness_seq: np.ndarray
    use_time_first_seq: np.ndarray
    use_time_seq: np.ndarray
    num_hint_seq: np.ndarray
    mask_seq: np.ndarray
    learning_goals: np.ndarray
    split_id: int


def _parse_numeric_sequence(raw: str, dtype) -> np.ndarray:
    """Parse a comma separated numeric sequence into a numpy array."""

    raw = raw.strip()
    if not raw:
        return np.array([], dtype=dtype)
    values = np.fromstring(raw, sep=",", dtype=dtype)
    if values.ndim != 1:
        values = values.reshape(-1)
    return values

def _normalise_path(path: str) -> str:
    """Return a filesystem path with common copy/paste artefacts removed."""

    path = os.path.expanduser(path.strip())
    if os.path.isfile(path):
        return path

    trimmed = path.rstrip(",;")
    if trimmed != path and os.path.isfile(trimmed):
        return trimmed

    return path

def _parse_split(path: str, split_name: str) -> List[Record]:
    """Read a split file and return a list of :class:`Record` instances."""

    if split_name not in SPLIT_TO_ID:
        raise ValueError(f"Unknown split '{split_name}'. Expected one of {list(SPLIT_TO_ID)}.")
    
    path = _normalise_path(path)

    with open(path, "r", encoding="utf-8") as file:
        header = file.readline().strip()
        if ";" not in header:
            raise ValueError(
                f"The header of '{path}' does not contain the expected ';' separator."
            )
        meta_fields, seq_fields = header.split(";", maxsplit=1)
        record_size = len(meta_fields.split(",")) + len(seq_fields.split(","))
        lines = [line.strip() for line in file if line.strip()]

    if len(lines) % record_size != 0:
        raise ValueError(
            f"The number of non-empty lines in '{path}' ({len(lines)}) is not a multiple of "
            f"the expected record size ({record_size})."
        )

    records: List[Record] = []
    for start in range(0, len(lines), record_size):
        chunk = lines[start : start + record_size]
        user_id = int(chunk[0])
        school_id = int(chunk[1])
        seq_len = int(chunk[2])
        question_seq = _parse_numeric_sequence(chunk[3], np.int64)
        num_attempt_seq = _parse_numeric_sequence(chunk[4], np.int64)
        correctness_seq = _parse_numeric_sequence(chunk[5], np.float32)
        use_time_first_seq = _parse_numeric_sequence(chunk[6], np.float32)
        use_time_seq = _parse_numeric_sequence(chunk[7], np.float32)
        num_hint_seq = _parse_numeric_sequence(chunk[8], np.int64)
        mask_seq = _parse_numeric_sequence(chunk[9], np.int64).astype(np.bool_, copy=False)
        learning_goals = _parse_numeric_sequence(chunk[10], np.int64)

        expected_fields = {
            "question_seq": question_seq,
            "num_attempt_seq": num_attempt_seq,
            "correctness_seq": correctness_seq,
            "use_time_first_seq": use_time_first_seq,
            "use_time_seq": use_time_seq,
            "num_hint_seq": num_hint_seq,
            "mask_seq": mask_seq,
        }
        for field_name, values in expected_fields.items():
            if len(values) != seq_len:
                raise ValueError(
                    f"{field_name} length {len(values)} does not match seq_len {seq_len} "
                    f"for user {user_id} in '{path}'."
                )

        records.append(
            Record(
                user_id=user_id,
                school_id=school_id,
                seq_len=seq_len,
                question_seq=question_seq.astype(np.int64, copy=False),
                num_attempt_seq=num_attempt_seq.astype(np.int64, copy=False),
                correctness_seq=correctness_seq.astype(np.float32, copy=False),
                use_time_first_seq=use_time_first_seq.astype(np.float32, copy=False),
                use_time_seq=use_time_seq.astype(np.float32, copy=False),
                num_hint_seq=num_hint_seq.astype(np.int64, copy=False),
                mask_seq=mask_seq.astype(np.bool_, copy=False),
                learning_goals=learning_goals.astype(np.int64, copy=False),
                split_id=SPLIT_TO_ID[split_name],
            )
        )

    return records


def _build_skill_mapping(records: Sequence[Record]) -> dict[int, int]:
    """Create a contiguous mapping for the question identifiers."""

    unique_skills = set()
    for record in records:
        unique_skills.update(record.question_seq.tolist())
        unique_skills.update(record.learning_goals.tolist())

    sorted_skills = sorted(unique_skills)
    return {skill_id: idx for idx, skill_id in enumerate(sorted_skills)}


def _remap_records(records: Iterable[Record], mapping: dict[int, int]) -> None:
    """In-place remapping of question identifiers using ``mapping``."""

    for record in records:
        record.question_seq = np.asarray([mapping[q] for q in record.question_seq], dtype=np.int64)
        if record.learning_goals.size:
            record.learning_goals = np.asarray(
                [mapping[g] for g in record.learning_goals if g in mapping], dtype=np.int64
            )


def _records_to_arrays(records: Sequence[Record]) -> dict[str, np.ndarray]:
    """Convert the list of records into numpy arrays ready for persistence."""

    def _object_array(getter):
        return np.array([getter(rec) for rec in records], dtype=object)

    arrays: dict[str, np.ndarray] = {
        "skill": _object_array(lambda rec: rec.question_seq),
        "num_attempt": _object_array(lambda rec: rec.num_attempt_seq),
        "y": _object_array(lambda rec: rec.correctness_seq.astype(np.float32, copy=False)),
        "use_time_first": _object_array(lambda rec: rec.use_time_first_seq),
        "use_time": _object_array(lambda rec: rec.use_time_seq),
        "num_hint": _object_array(lambda rec: rec.num_hint_seq),
        "mask": _object_array(lambda rec: rec.mask_seq.astype(np.bool_, copy=False)),
        "learning_goals": _object_array(lambda rec: rec.learning_goals),
        "real_len": np.array([rec.seq_len for rec in records], dtype=np.int32),
        "user_id": np.array([rec.user_id for rec in records], dtype=np.int64),
        "school_id": np.array([rec.school_id for rec in records], dtype=np.int64),
        "split": np.array([rec.split_id for rec in records], dtype=np.int8),
    }

    split_indices: dict[int, list[int]] = {split_id: [] for split_id in SPLIT_TO_ID.values()}
    for idx, rec in enumerate(records):
        split_indices[rec.split_id].append(idx)

    for split_name, split_id in SPLIT_TO_ID.items():
        arrays[f"{split_name}_indices"] = np.array(split_indices[split_id], dtype=np.int64)

    return arrays


def _save_npz(arrays: dict[str, np.ndarray], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def _save_metadata(mapping: dict[int, int], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metadata = {
        "skill_id_mapping": mapping,
        "split_labels": {name: split_id for name, split_id in SPLIT_TO_ID.items()},
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def merge_splits(
    train_path: str,
    valid_path: str,
    test_path: str,
    output_dir: str,
    dataset_name: str,
    remap_skills: bool = True,
) -> None:
    """Merge the three dataset splits and persist them in SRC format."""

    split_files = {
        "train": train_path,
        "valid": valid_path,
        "test": test_path,
    }

    all_records: List[Record] = []
    for split_name, split_path in split_files.items():
        if split_path:
            all_records.extend(_parse_split(split_path, split_name))

    if remap_skills:
        mapping = _build_skill_mapping(all_records)
        _remap_records(all_records, mapping)
    else:
        mapping: dict[int, int] = {}
        for record in all_records:
            for skill in record.question_seq:
                mapping[int(skill)] = int(skill)
            for goal in record.learning_goals:
                mapping[int(goal)] = int(goal)

    arrays = _records_to_arrays(all_records)
    dataset_path = os.path.join(output_dir, dataset_name, f"{dataset_name}.npz")
    mapping_path = os.path.join(output_dir, dataset_name, f"{dataset_name}_metadata.json")

    _save_npz(arrays, dataset_path)
    _save_metadata(mapping, mapping_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge KT dataset splits for SRC training.")
    parser.add_argument("--train", required=True, help="Path to the training split file.")
    parser.add_argument("--valid", required=True, help="Path to the validation split file.")
    parser.add_argument("--test", required=True, help="Path to the test split file.")
    parser.add_argument(
        "--output-dir",
        default="../data/assist2009",
        help="Directory where the SRC-formatted dataset will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Name of the dataset; also used as the filename inside the output directory.",
    )
    parser.add_argument(
        "--no-remap",
        action="store_true",
        help="Disable remapping of question identifiers to a contiguous range.",
    )
    args = parser.parse_args()

    merge_splits(
        train_path=args.train,
        valid_path=args.valid,
        test_path=args.test,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        remap_skills=not args.no_remap,
    )


if __name__ == "__main__":

    SPLIT_TO_ID = {"train": 0, "valid": 1, "test": 2}
    main()

    # python prepare_src_dataset.py --train /home/zengxiangyu/SRC-py/data/assist2009/train.txt, --valid /home/zengxiangyu/SRC-py/data/assist2009/valid.txt, --test  /home/zengxiangyu/SRC-py/data/assist2009/test.txt, --dataset-name assist2009