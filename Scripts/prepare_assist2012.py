from __future__ import annotations

"""Prepare the ASSISTments 2012 problem log export.

This script mirrors the preprocessing pipeline implemented in
``prepare_assist2009.py`` but tweaks the column defaults and output naming so
that the "problem log" style export (which labels concepts through the
``skill`` column and orders records by ``problem_log_id``) can be converted
without modifying the original helper.

Running the script will generate two files next to the raw CSV (or inside the
directory passed with ``--output-dir``):

* ``assist2012.npz``
* ``knowledge_concept_mapping_assist2012.csv``

Both artefacts follow the exact structure produced by the original
ASSISTments 2009 preprocessor so downstream tooling can consume them without
changes.
"""

import argparse
import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _parse_multi_value(cell: object) -> List[int]:
    if cell is None:
        return []
    if isinstance(cell, (int, float)):
        if pd.isna(cell):
            return []
        return [int(cell)]

    parts: List[str]
    if isinstance(cell, str):
        stripped = cell.strip()
        if not stripped or stripped.lower() == "nan":
            return []
        parts = [p for p in stripped.replace(" ", "").split("~~") if p]
    else:
        parts = [str(cell)] if cell is not None else []

    ids: List[int] = []
    for part in parts:
        try:
            parsed = int(float(part))
        except ValueError:
            continue
        ids.append(parsed)
    return ids


def _parse_multi_name(cell: object) -> List[str]:
    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text or text.lower() == "nan":
        return []
    return [p.strip() for p in text.split("~~") if p.strip()]


@dataclass
class SequenceExample:
    user_id: int
    school_id: int
    question_seq: List[int]
    skill_seq: List[int]
    attempt_seq: List[int]
    correctness_seq: List[int]
    first_response_seq: List[int]
    response_seq: List[int]
    hint_seq: List[int]
    learning_goals: List[int]

    @property
    def seq_len(self) -> int:
        return len(self.skill_seq)


def _coerce_to_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_raw_dataset(path: Path, *, order_column: str, encoding: str | None = None) -> pd.DataFrame:
    read_kwargs = {"low_memory": False}
    tried_encodings: List[str] = []

    if encoding:
        tried_encodings.append(encoding)
        df = pd.read_csv(path, encoding=encoding, **read_kwargs)
    else:
        candidates = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        last_error: UnicodeDecodeError | None = None
        for candidate in candidates:
            tried_encodings.append(candidate)
            try:
                df = pd.read_csv(path, encoding=candidate, **read_kwargs)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            else:
                if candidate != "utf-8":
                    print(
                        "Warning: fell back to encoding '%s' when reading %s" %
                        (candidate, path)
                    )
                break
        else:
            raise last_error if last_error is not None else UnicodeDecodeError(
                "unknown",
                b"",
                0,
                1,
                f"Unable to decode {path} with tried encodings: {', '.join(tried_encodings)}",
            )

    if order_column not in df.columns:
        raise ValueError(f"Column '{order_column}' not found in {path}")
    df = df.sort_values(by=["user_id", order_column])
    return df


def build_examples(
    df: pd.DataFrame,
    *,
    question_column: str,
    attempt_column: str,
    correctness_column: str,
    first_time_column: str,
    response_time_column: str,
    hint_column: str,
    skill_id_column: str,
    skill_name_column: str,
) -> Tuple[List[SequenceExample], Dict[int, str]]:
    concept_mapping: "OrderedDict[int, str]" = OrderedDict()
    examples: List[SequenceExample] = []

    required_columns = [
        question_column,
        attempt_column,
        correctness_column,
        first_time_column,
        hint_column,
        skill_id_column,
        skill_name_column,
    ]
    if response_time_column in df.columns:
        required_columns.append(response_time_column)
    else:
        response_time_column = first_time_column

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for user_id, group in df.groupby("user_id"):
        school_values = group.get("school_id")
        if school_values is not None:
            school_id = int(next((v for v in school_values if not pd.isna(v)), -1))
        else:
            school_id = -1

        question_seq: List[int] = []
        skill_seq: List[int] = []
        attempt_seq: List[int] = []
        correctness_seq: List[int] = []
        first_response_seq: List[int] = []
        response_seq: List[int] = []
        hint_seq: List[int] = []
        learning_goals: List[int] = []
        seen: set[int] = set()

        for row in group.itertuples(index=False):
            sid_cell = getattr(row, skill_id_column)
            sname_cell = getattr(row, skill_name_column)
            sid_list = _parse_multi_value(sid_cell)
            sname_list = _parse_multi_name(sname_cell)

            if sname_list and sid_list and len(sid_list) != len(sname_list):
                sname_list = sname_list[: len(sid_list)]

            for idx, sid in enumerate(sid_list):
                candidate = ""
                if idx < len(sname_list):
                    candidate = sname_list[idx].strip()

                if sid not in concept_mapping:
                    concept_mapping[sid] = candidate
                else:
                    existing = concept_mapping[sid]
                    if candidate:
                        if existing and existing != candidate:
                            raise ValueError(
                                f"Conflicting names for skill id {sid}: '{existing}' vs '{candidate}'"
                            )
                        if not existing:
                            concept_mapping[sid] = candidate

            if not sid_list:
                continue

            primary_sid = sid_list[0]

            question_seq.append(_coerce_to_int(getattr(row, question_column), -1))
            attempt_seq.append(_coerce_to_int(getattr(row, attempt_column), 0))
            correctness_seq.append(_coerce_to_int(getattr(row, correctness_column), 0))
            first_response_seq.append(_coerce_to_int(getattr(row, first_time_column), 0))
            response_seq.append(_coerce_to_int(getattr(row, response_time_column), 0))
            hint_seq.append(_coerce_to_int(getattr(row, hint_column), 0))
            skill_seq.append(primary_sid)

            if primary_sid not in seen:
                learning_goals.append(primary_sid)
                seen.add(primary_sid)

        if not question_seq:
            continue

        examples.append(
            SequenceExample(
                user_id=int(user_id),
                school_id=school_id,
                question_seq=question_seq,
                skill_seq=skill_seq,
                attempt_seq=attempt_seq,
                correctness_seq=correctness_seq,
                first_response_seq=first_response_seq,
                response_seq=response_seq,
                hint_seq=hint_seq,
                learning_goals=learning_goals,
            )
        )

    return examples, concept_mapping


def reindex_concepts(
    examples: Sequence[SequenceExample],
    concept_mapping: "OrderedDict[int, str]",
) -> Tuple[Sequence[SequenceExample], "OrderedDict[int, str]", List[int]]:
    old_to_new: Dict[int, int] = {}
    new_mapping: "OrderedDict[int, str]" = OrderedDict()
    missing_name_ids: List[int] = []

    for new_id, (old_id, name) in enumerate(concept_mapping.items()):
        old_to_new[old_id] = new_id
        new_mapping[new_id] = name
        if not name:
            missing_name_ids.append(new_id)

    for example in examples:
        example.skill_seq = [old_to_new[sid] for sid in example.skill_seq]
        example.learning_goals = [old_to_new[sid] for sid in example.learning_goals]

    return examples, new_mapping, missing_name_ids


def drop_missing_skill_names(df: pd.DataFrame, *, skill_name_column: str) -> Tuple[pd.DataFrame, int]:
    if skill_name_column not in df.columns:
        return df, 0

    column = df[skill_name_column]
    missing_mask = column.isna()

    column_as_str = column.astype(str).str.strip()
    lowered = column_as_str.str.lower()
    missing_mask |= column_as_str.eq("")
    missing_mask |= lowered.eq("nan")
    missing_mask |= lowered.eq("none")

    dropped = int(missing_mask.sum())
    if not dropped:
        return df, 0

    cleaned = df.loc[~missing_mask].copy()
    return cleaned, dropped


def handle_missing_values(
    df: pd.DataFrame,
    *,
    strategy: str,
    drop_columns: Sequence[str],
    fill_defaults: Dict[str, object],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if strategy not in {"impute", "drop"}:
        raise ValueError(f"Unsupported missing value strategy: {strategy}")

    working = df
    drop_mask = pd.Series(False, index=df.index)
    drop_reason: Dict[str, int] = {}
    imputed: Dict[str, int] = {}

    for column in drop_columns:
        if column not in df.columns:
            continue
        missing_mask = df[column].isna()
        count = int(missing_mask.sum())
        if count:
            drop_reason[column] = drop_reason.get(column, 0) + count
            drop_mask |= missing_mask

    if drop_mask.any():
        working = df.loc[~drop_mask].copy()

    for column, default in fill_defaults.items():
        if column not in working.columns:
            continue
        missing_mask = working[column].isna()
        count = int(missing_mask.sum())
        if not count:
            continue
        if strategy == "drop":
            drop_reason[column] = drop_reason.get(column, 0) + count
            working = working.loc[~missing_mask].copy()
        else:
            working.loc[missing_mask, column] = default
            imputed[column] = imputed.get(column, 0) + count

    return working, {
        "dropped_rows": int(df.shape[0] - working.shape[0]),
        "dropped_by_column": drop_reason,
        "imputed_by_column": imputed,
    }


def _to_object_array(examples: Sequence[SequenceExample], getter, dtype: np.dtype) -> np.ndarray:
    return np.array([np.asarray(getter(example), dtype=dtype) for example in examples], dtype=object)


def write_npz(
    path: Path,
    examples: Sequence[SequenceExample],
    concept_mapping: Dict[int, str],
) -> None:
    data = {
        "user_id": np.array([example.user_id for example in examples], dtype=np.int32),
        "school_id": np.array([example.school_id for example in examples], dtype=np.int32),
        "question_id": _to_object_array(examples, lambda ex: ex.question_seq, np.int32),
        "skill": _to_object_array(examples, lambda ex: ex.skill_seq, np.int32),
        "attempt": _to_object_array(examples, lambda ex: ex.attempt_seq, np.int32),
        "y": _to_object_array(examples, lambda ex: ex.correctness_seq, np.int8),
        "first_response_time": _to_object_array(examples, lambda ex: ex.first_response_seq, np.int32),
        "response_time": _to_object_array(examples, lambda ex: ex.response_seq, np.int32),
        "hint_count": _to_object_array(examples, lambda ex: ex.hint_seq, np.int32),
        "mask": _to_object_array(examples, lambda ex: [1] * ex.seq_len, np.int8),
        "learning_goals": _to_object_array(examples, lambda ex: ex.learning_goals, np.int32),
        "real_len": np.array([example.seq_len for example in examples], dtype=np.int32),
        "concept_ids": np.array(list(concept_mapping.keys()), dtype=np.int32),
        "concept_names": np.array(list(concept_mapping.values()), dtype=object),
    }

    np.savez_compressed(path, **data)


def write_mapping(path: Path, mapping: Dict[int, str]) -> None:
    with path.open("w", encoding="utf8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["concept_id", "concept_name"])
        for concept_id, name in mapping.items():
            writer.writerow([concept_id, name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the ASSISTments 2012 problem log dataset",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw ASSISTments problem log CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the processed files will be stored (defaults to the CSV directory)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="assist2012",
        help="Base name used for the generated .npz file",
    )
    parser.add_argument(
        "--order-column",
        type=str,
        default="problem_log_id",
        help="Column used to sort a learner's history",
    )
    parser.add_argument("--question-column", type=str, default="assistment_id")
    parser.add_argument("--attempt-column", type=str, default="attempt_count")
    parser.add_argument("--correctness-column", type=str, default="correct")
    parser.add_argument("--first-time-column", type=str, default="ms_first_response")
    parser.add_argument("--response-time-column", type=str, default="ms_first_response")
    parser.add_argument("--hint-column", type=str, default="hint_count")
    parser.add_argument("--skill-id-column", type=str, default="skill_id")
    parser.add_argument("--skill-name-column", type=str, default="skill")
    parser.add_argument(
        "--missing-strategy",
        choices=("impute", "drop"),
        default="impute",
        help=(
            "How to handle missing values in interaction columns. 'impute' replaces them with safe defaults "
            "while 'drop' removes affected interactions before sequence construction."
        ),
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Encoding of the raw CSV file. When omitted, the script will try several common options",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional custom path for the concept-id mapping CSV",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="If set, create a JSON file with dataset statistics for quick inspection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    if args.output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset(
        input_path,
        order_column=args.order_column,
        encoding=args.encoding,
    )
    cleaned_df, missing_report = handle_missing_values(
        raw_df,
        strategy=args.missing_strategy,
        drop_columns=("user_id", args.order_column),
        fill_defaults={
            args.question_column: -1,
            args.attempt_column: 0,
            args.correctness_column: 0,
            args.first_time_column: 0,
            args.response_time_column: 0,
            args.hint_column: 0,
        },
    )

    cleaned_df, dropped_missing_skill_names = drop_missing_skill_names(
        cleaned_df,
        skill_name_column=args.skill_name_column,
    )
    examples, mapping = build_examples(
        cleaned_df,
        question_column=args.question_column,
        attempt_column=args.attempt_column,
        correctness_column=args.correctness_column,
        first_time_column=args.first_time_column,
        response_time_column=args.response_time_column,
        hint_column=args.hint_column,
        skill_id_column=args.skill_id_column,
        skill_name_column=args.skill_name_column,
    )

    examples, reindexed_mapping, missing_name_ids = reindex_concepts(examples, mapping)

    dataset_path = output_dir / f"{args.dataset_name}.npz"
    mapping_path = args.mapping_file
    if mapping_path is None:
        mapping_path = output_dir / "knowledge_concept_mapping_assist2012.csv"
    elif not mapping_path.is_absolute():
        mapping_path = output_dir / mapping_path

    write_npz(dataset_path, examples, reindexed_mapping)
    write_mapping(mapping_path, reindexed_mapping)

    print(f"Saved processed dataset to {dataset_path}")
    print(f"Saved concept mapping to {mapping_path}")
    if missing_report["dropped_rows"]:
        drop_details = [
            f"{col}: {count}"
            for col, count in sorted(missing_report["dropped_by_column"].items())
            if count
        ]
        if drop_details:
            print(
                "Dropped %d rows due to missing values (%s)"
                % (missing_report["dropped_rows"], ", ".join(drop_details))
            )
        else:
            print(
                "Dropped %d rows due to missing values"
                % missing_report["dropped_rows"]
            )
    if missing_report["imputed_by_column"]:
        print(
            "Imputed missing values in: %s"
            % ", ".join(
                f"{col} ({count})"
                for col, count in sorted(missing_report["imputed_by_column"].items())
            )
        )

    if dropped_missing_skill_names:
        print(
            "Dropped %d rows due to missing skill names"
            % dropped_missing_skill_names
        )

    if missing_name_ids:
        formatted = ", ".join(str(new_id) for new_id in missing_name_ids)
        print(
            "Warning: %d concepts lack names in the source data: %s"
            % (len(missing_name_ids), formatted)
        )

    if args.write_summary:
        total_interactions = sum(example.seq_len for example in examples)
        summary = {
            "num_users": len(examples),
            "concepts": len(reindexed_mapping),
            "total_interactions": int(total_interactions),
            "max_seq_len": max((example.seq_len for example in examples), default=0),
            "mean_seq_len": (
                float(total_interactions) / len(examples)
                if examples
                else 0.0
            ),
            "skipped_interactions": int(len(raw_df) - total_interactions),
            "missing_value_report": missing_report,
            "output_file": str(dataset_path),
            "dropped_missing_skill_names": dropped_missing_skill_names,
            "concepts_missing_names": [
                {"id": new_id}
                for new_id in missing_name_ids
            ],
        }
        summary_path = output_dir / "dataset_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
        print(f"Saved dataset summary to {summary_path}")


if __name__ == "__main__":
    main()