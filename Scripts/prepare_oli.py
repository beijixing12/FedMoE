from __future__ import annotations

"""Prepare the OLI Statics dataset for the SRC knowledge tracing pipeline.

The OLI (Open Learning Initiative) statics dataset ships as a transaction log
with separate columns describing the student identifier, problem/step names,
first-attempt correctness and several knowledge component annotations.  This
script mirrors the preprocessing flow used for the ASSISTments exports so that
the generated artefacts (``.npz`` interaction archive plus the concept mapping
CSV) match the format consumed by the downstream simulators and agents in this
repository.

By default the converter reads the ``KC (Single-KC)`` column, assigns synthetic
integer concept identifiers, and produces the following files next to the raw
input (or inside ``--output-dir`` when supplied):

* ``oli.npz``
* ``knowledge_concept_mapping_oli.csv``

Both outputs are structurally identical to those produced by
``prepare_assist2009.py`` / ``prepare_assist2012.py``, allowing the new dataset
to be dropped into existing training scripts without further changes.
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from prepare_assist2009 import (
    build_examples,
    drop_missing_skill_names,
    reindex_concepts,
    write_mapping,
    write_npz,
)

_CANONICAL_NPZ_KEYS = {
    "user_id",
    "school_id",
    "question_id",
    "skill",
    "attempt",
    "y",
    "first_response_time",
    "response_time",
    "hint_count",
    "mask",
    "learning_goals",
    "real_len",
    "concept_ids",
    "concept_names",
}

_LEGACY_ALIAS_MAP = {
    "u": "user_id",
    "i": "skill",
    "y": "y",
    "t": "real_len",
}


def _try_read_csv(path: Path, *, delimiter: str, encoding: str | None) -> pd.DataFrame:
    """Attempt to read the raw export trying multiple encodings."""

    read_kwargs = {"low_memory": False, "sep": delimiter}
    tried_encodings: List[str] = []

    if encoding:
        tried_encodings.append(encoding)
        return pd.read_csv(path, encoding=encoding, **read_kwargs)

    candidates = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_error: Exception | None = None
    for candidate in candidates:
        tried_encodings.append(candidate)
        try:
            df = pd.read_csv(path, encoding=candidate, **read_kwargs)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
            continue
        else:
            if candidate != "utf-8":
                print(
                    "Warning: fell back to encoding '%s' when reading %s"
                    % (candidate, path)
                )
            return df

    if last_error is None:
        last_error = UnicodeDecodeError(
            "unknown",
            b"",
            0,
            1,
            f"Unable to decode {path} with tried encodings: {', '.join(tried_encodings)}",
        )
    raise last_error


def _load_raw_dataset(path: Path, *, encoding: str | None) -> pd.DataFrame:
    """Read the OLI export, automatically inferring the delimiter."""

    delimiters = ["\t", ",", ";"]
    last_error: Exception | None = None
    for delimiter in delimiters:
        try:
            df = _try_read_csv(path, delimiter=delimiter, encoding=encoding)
        except Exception as exc:  # noqa: BLE001 - propagate last failure below
            last_error = exc
            continue
        if df.shape[1] <= 1:
            last_error = ValueError(
                f"Failed to parse {path} using delimiter '{delimiter}'"
            )
            continue
        return df

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read dataset from {path}")


def _resolve_column_name(
    columns: Iterable[str],
    target: str,
    *,
    aliases: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Return the column name that matches ``target`` (case-/alias-insensitive)."""

    if target in columns:
        return target

    lowered = target.lower()
    for name in columns:
        if name.lower() == lowered:
            return name

    synonym_map = {
        "anon student id": ["anon_student_id", "student id", "studentid", "user id"],
        "problem name": ["problem", "problemname"],
        "step name": ["step", "stepname"],
        "problem view": ["attempt", "attempt count", "problem_view"],
        "first attempt": ["first_attempt", "first outcome", "first result"],
        "step duration (sec)": ["step duration", "duration (sec)", "duration"],
        "hints": ["hint count", "hint_count", "num_hints"],
        "first transaction time": ["first transaction", "first timestamp", "first_time"],
        "step start time": ["start time", "timestamp", "time"],
        "kc (f2011)": [
            "kc", "kc f2011", "skill", "kc_f2011", "kc(f2011)", "knowledge component",
        ],
        "kc (single-kc)": ["kc single", "skill single", "kc_single", "single kc"],
        "kc (unique-step)": ["kc unique", "kc_unique", "unique-step"],
    }

    search_list = list(aliases or []) + synonym_map.get(lowered, [])
    for alias in search_list:
        for name in columns:
            if name.lower() == alias.lower():
                return name
    return None


def _coerce_int(value: object, *, default: int = 0) -> int:
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
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


def _coerce_float(value: object, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (float, np.floating, int, np.integer)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    lowered = text.lower()
    if lowered in {".", "nan", "null", "none"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default

_PLACEHOLDER_CONCEPT_NAMES = {
    "nan",
    "none",
    "null",
    ".",
    "",
    "noskill",
    "single-kc",
    "single kc",
}


def _normalise_concept_names(cell: object) -> List[str]:
    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []
    text = str(cell).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.replace(";", "~~").split("~~")]
    cleaned: List[str] = []
    seen: set[str] = set()
    for part in parts:
        lowered = part.lower()
        if lowered in _PLACEHOLDER_CONCEPT_NAMES:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(part)
    return cleaned

def _ensure_canonical_npz_schema(path: Path) -> None:
    """Make sure the written NPZ archive exposes the standard column names."""

    with np.load(path, allow_pickle=True) as artefact:
        available = set(artefact.files)
        missing = _CANONICAL_NPZ_KEYS - available

        if missing and any(alias in available for alias in _LEGACY_ALIAS_MAP):
            payload = {name: artefact[name] for name in artefact.files}
            renamed = False
            for alias, canonical in _LEGACY_ALIAS_MAP.items():
                if alias in payload and canonical not in payload:
                    payload[canonical] = payload.pop(alias)
                    renamed = True
            if renamed:
                np.savez_compressed(path, **payload)
                available = set(payload.keys())
                missing = _CANONICAL_NPZ_KEYS - available

        if missing:
            raise ValueError(
                "Processed dataset %s is missing canonical arrays: %s (found: %s)"
                % (path, ", ".join(sorted(missing)), ", ".join(sorted(available)))
            )

_MISSING_TEXT_TOKENS = {
    "",
    "nan",
    "none",
    "null",
    ".",
}


def _is_missing_text(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip().lower()
    return text in _MISSING_TEXT_TOKENS


def _extract_order_value(
    timestamp: pd.Timestamp | float | int | None,
    fallback: int,
) -> int:
    if isinstance(timestamp, pd.Timestamp):
        if pd.isna(timestamp):
            return fallback
        return int(timestamp.value)
    if timestamp is None:
        return fallback
    if isinstance(timestamp, (int, np.integer)):
        return int(timestamp)
    if isinstance(timestamp, (float, np.floating)):
        return int(timestamp)
    return fallback


def build_interaction_frame(
    raw_df: pd.DataFrame,
    *,
    concept_column: str,
) -> Tuple[pd.DataFrame, OrderedDict[int, str], int]:
    columns = raw_df.columns

    student_col = _resolve_column_name(columns, "Anon Student Id")
    problem_col = _resolve_column_name(columns, "Problem Name")
    step_col = _resolve_column_name(columns, "Step Name")
    attempt_col = _resolve_column_name(columns, "Problem View")
    first_attempt_col = _resolve_column_name(columns, "First Attempt")
    hint_col = _resolve_column_name(columns, "Hints")
    duration_col = _resolve_column_name(columns, "Step Duration (sec)")
    first_tx_col = _resolve_column_name(columns, "First Transaction Time")
    start_time_col = _resolve_column_name(columns, "Step Start Time")

    if student_col is None:
        raise ValueError("Column 'Anon Student Id' not found in dataset")
    if problem_col is None or step_col is None:
        raise ValueError("Problem/step columns are required to derive exercise ids")
    if first_attempt_col is None:
        raise ValueError("Column 'First Attempt' not found in dataset")
    if concept_column not in columns:
        resolved = _resolve_column_name(columns, concept_column)
        if resolved is None:
            raise ValueError(f"Concept column '{concept_column}' not found in dataset")
        concept_column = resolved

    timestamps = None
    if first_tx_col is not None and first_tx_col in raw_df.columns:
        timestamps = pd.to_datetime(raw_df[first_tx_col], errors="coerce")
    elif start_time_col is not None and start_time_col in raw_df.columns:
        timestamps = pd.to_datetime(raw_df[start_time_col], errors="coerce")

    user_mapping: Dict[str, int] = {}
    question_mapping: Dict[Tuple[str, str], int] = {}
    concept_mapping: "OrderedDict[int, str]" = OrderedDict()
    name_to_id: Dict[str, int] = {}

    records: List[Dict[str, object]] = []
    skipped = 0

    for idx, row in raw_df.iterrows():
        student_cell = row.get(student_col)
        student_key: Optional[str] = None
        if not _is_missing_text(student_cell):
            student_key = str(student_cell).strip()
        if not student_key:
            skipped += 1
            continue
        user_id = user_mapping.setdefault(student_key, len(user_mapping))
        concept_names = _normalise_concept_names(row.get(concept_column))
        if not concept_names:
            skipped += 1
            continue

        skill_ids: List[int] = []
        for name in concept_names:
            concept_id = name_to_id.get(name)
            if concept_id is None:
                concept_id = len(concept_mapping)
                concept_mapping[concept_id] = name
                name_to_id[name] = concept_id
            skill_ids.append(concept_id)
        if _is_missing_text(row.get(problem_col)) or _is_missing_text(row.get(step_col)):
            skipped += 1
            continue
        problem_key = str(row[problem_col]).strip()
        step_key = str(row[step_col]).strip()
        question_key = (problem_key, step_key)
        question_id = question_mapping.setdefault(question_key, len(question_mapping))

        attempt_value = _coerce_int(row.get(attempt_col), default=1) if attempt_col else 1
        hint_value = _coerce_int(row.get(hint_col), default=0) if hint_col else 0
        duration_value = _coerce_float(row.get(duration_col), default=0.0) if duration_col else 0.0
        ms_duration = int(round(duration_value * 1000))

        attempt_cell = row[first_attempt_col]
        if _is_missing_text(attempt_cell):
            skipped += 1
            continue
        attempt_label = str(attempt_cell).strip().lower()
        correct_value = 1 if attempt_label in {"correct", "1", "true", "t"} else 0

        timestamp_value = None
        if timestamps is not None:
            timestamp_value = timestamps.iloc[idx]
        order_value = _extract_order_value(timestamp_value, fallback=idx)

        records.append(
            {
                "user_id": user_id,
                "school_id": -1,
                "interaction_order": order_value,
                "question_id": question_id,
                "attempt_count": attempt_value,
                "correct": correct_value,
                "ms_first_response": ms_duration,
                "ms_response_time": ms_duration,
                "hint_count": hint_value,
                "skill_id": "~~".join(str(sid) for sid in skill_ids),
                "skill": "~~".join(concept_names),
            }
        )

    if not records:
        raise ValueError("No valid interactions were found after filtering")

    frame = pd.DataFrame.from_records(records)
    frame.sort_values(by=["user_id", "interaction_order", "question_id"], inplace=True)
    frame["interaction_order"] = frame.groupby("user_id").cumcount()
    return frame, concept_mapping, skipped


def summarise_dataset(
    examples: Sequence,
    *,
    raw_rows: int,
    dataset_path: Path,
    concept_mapping: OrderedDict[int, str],
) -> Dict[str, object]:
    total_interactions = sum(example.seq_len for example in examples)
    summary = {
        "num_users": len(examples),
        "num_concepts": len(concept_mapping),
        "total_interactions": int(total_interactions),
        "max_seq_len": max((example.seq_len for example in examples), default=0),
        "mean_seq_len": (
            float(total_interactions) / len(examples) if examples else 0.0
        ),
        "skipped_interactions": int(raw_rows - total_interactions),
        "output_file": str(dataset_path),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the OLI statics dataset")
    parser.add_argument("--input", type=Path, required=True, help="Path to the raw OLI CSV/TSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the processed files will be stored (defaults to the CSV directory)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="oli",
        help="Base name used for the generated .npz file",
    )
    parser.add_argument(
        "--concept-column",
        type=str,
        default="KC (F2011)",
        help=(
            "Column containing the knowledge component labels. The default uses "
            "the OLI Statics F2011 skill model; override this if your export uses "
            "a different knowledge component column."
        ),
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Encoding of the raw file. When omitted, several common encodings are tried automatically",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="If set, write a JSON file with basic dataset statistics",
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

    raw_df = _load_raw_dataset(input_path, encoding=args.encoding)
    concept_column = args.concept_column
    resolved_name = concept_column
    if concept_column not in raw_df.columns:
        resolved = _resolve_column_name(raw_df.columns, concept_column)
        if resolved is None:
            fallback_columns = [
                "KC (F2011)",
                "KC (Unique-step)",
                "KC (Single-KC)",
            ]
            for candidate in fallback_columns:
                if candidate == concept_column:
                    continue
                resolved = _resolve_column_name(raw_df.columns, candidate)
                if resolved is not None:
                    resolved_name = resolved
                    print(
                        "Warning: concept column '%s' not found; falling back to '%s'"
                        % (concept_column, candidate)
                    )
                    break
            else:
                raise ValueError(
                    f"Concept column '{concept_column}' not found in dataset. "
                    "Use --concept-column to point to a valid knowledge component column."
                )
        else:
            resolved_name = resolved
    else:
        resolved_name = concept_column


    interaction_df, _, skipped = build_interaction_frame(
        raw_df,
        concept_column=resolved_name,
    )

    if skipped:
        print(f"Skipped {skipped} rows due to missing identifiers or concepts")

    cleaned_df, dropped_missing_skill_names = drop_missing_skill_names(
        interaction_df,
        skill_name_column="skill",
    )

    if dropped_missing_skill_names:
        print(
            "Dropped %d rows due to empty or placeholder skill names"
            % dropped_missing_skill_names
        )

    examples, mapping = build_examples(
        cleaned_df,
        question_column="question_id",
        attempt_column="attempt_count",
        correctness_column="correct",
        first_time_column="ms_first_response",
        response_time_column="ms_response_time",
        hint_column="hint_count",
        skill_id_column="skill_id",
        skill_name_column="skill",
    )

    examples, reindexed_mapping, missing_name_ids = reindex_concepts(examples, mapping)

    dataset_path = output_dir / f"{args.dataset_name}.npz"
    mapping_path = output_dir / f"knowledge_concept_mapping_{args.dataset_name}.csv"

    write_npz(dataset_path, examples, reindexed_mapping)
    _ensure_canonical_npz_schema(dataset_path)
    write_mapping(mapping_path, reindexed_mapping)

    print(f"Saved processed dataset to {dataset_path}")
    print(f"Saved concept mapping to {mapping_path}")

    if missing_name_ids:
        formatted = ", ".join(str(idx) for idx in missing_name_ids)
        print(
            "Warning: %d concepts lack names in the source data: %s"
            % (len(missing_name_ids), formatted)
        )

    if args.write_summary:
        summary = summarise_dataset(
            examples,
            raw_rows=raw_rows,
            dataset_path=dataset_path,
            concept_mapping=reindexed_mapping,
        )
        summary_path = output_dir / "dataset_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
        print(f"Saved dataset summary to {summary_path}")


if __name__ == "__main__":
    main()