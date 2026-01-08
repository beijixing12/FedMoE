from __future__ import annotations

"""Prepare the ASSISTments 2017 (ITEST) problem log export.

The ASSISTments 2017 dataset ships with a compact schema compared to the
classic 2009/2012 exports.  This script mirrors the preprocessing pipeline used
for those releases and reconstructs the missing columns (e.g. synthetic skill
identifiers, attempt counts, hint counters) so that the generated artefacts are
fully compatible with the downstream tooling in this repository.

Running the script will generate two files next to the raw TSV/CSV (or inside
the directory passed with ``--output-dir``):

* ``assist2017.npz``
* ``knowledge_concept_mapping_assist2017.csv``

Both artefacts follow the exact structure produced by ``prepare_assist2009.py``
and ``prepare_assist2012.py`` so consumers can swap between datasets without any
code changes.
"""

import argparse
import csv
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

def _try_read_csv(path: Path, *, delimiter: str, encoding: str | None) -> pd.DataFrame:
    """Attempt to read a CSV/TSV file trying multiple encodings."""

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
                    "Warning: fell back to encoding '%s' when reading %s" %
                    (candidate, path)
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


def _resolve_column_name(columns: Iterable[str], target: str) -> Optional[str]:
    """Return the column name that matches ``target`` (case- and alias-insensitive)."""

    if target in columns:
        return target

    lowered = target.lower()
    for name in columns:
        if name.lower() == lowered:
            return name

    synonym_map = {
        "starttime": ["start_time", "start time", "timestamp", "time"],
    }
    aliases = synonym_map.get(lowered, [])
    for alias in aliases:
        for name in columns:
            if name.lower() == alias:
                return name

    return None

def load_raw_dataset(
    path: Path,
    *,
    order_column: str,
    encoding: str | None = None,
    delimiter: str = "\t",
) -> pd.DataFrame:
    if delimiter in {"auto", "AUTO", ""}:
        delimiter_candidates = ["\t", ",", ";"]
    else:
        delimiter_candidates = [delimiter]
        for fallback in ("\t", ","):
            if fallback not in delimiter_candidates:
                delimiter_candidates.append(fallback)

    last_error: Exception | None = None
    for current_delimiter in delimiter_candidates:
        try:
            df = _try_read_csv(path, delimiter=current_delimiter, encoding=encoding)
        except Exception as exc:  # noqa: BLE001 - propagate after trying fallbacks
            last_error = exc
            continue

        resolved_name = _resolve_column_name(df.columns, order_column)
        if resolved_name is None:
            last_error = ValueError(
                f"Column '{order_column}' not found in {path} when using delimiter '{current_delimiter}'"
            )
            continue

        normalised = df.copy()
        if resolved_name != order_column:
            normalised = normalised.rename(columns={resolved_name: order_column})

        if "user_id" not in normalised.columns:
            user_column = _resolve_column_name(normalised.columns, "user_id")
            if user_column is None and "ITEST_id" in normalised.columns:
                user_column = "ITEST_id"
            if user_column is None:
                last_error = ValueError(
                    "The dataset must contain a 'user_id' or 'ITEST_id' column"
                )
                continue
            if user_column != "user_id":
                normalised = normalised.rename(columns={user_column: "user_id"})

        if "school_id" not in normalised.columns:
            normalised["school_id"] = -1

        return normalised.sort_values(by=["user_id", order_column])

    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to read dataset {path}")




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


def _load_graph_concepts(graph_path: Path) -> List[str]:
    with graph_path.open("r", encoding="utf8") as handle:
        data = json.load(handle)

    edges: Iterable[Dict[str, object]] = data.get("prerequisite_edges", [])  # type: ignore[assignment]
    concepts: "OrderedDict[str, None]" = OrderedDict()
    for edge in edges:
        for key in ("head", "tail"):
            name = str(edge.get(key, "")).strip()
            if not name:
                continue
            if name not in concepts:
                concepts[name] = None

    return list(concepts)


def _resolve_prerequisite_graph(path: Optional[Path]) -> Optional[Path]:
    if path is not None:
        candidate = path.expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Prerequisite graph not found: {candidate!s}")

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "prerequisites_graph.json",
        Path.cwd() / "data" / "prerequisites_graph.json",
        Path.cwd() / "data" / "assist09" / "prerequisites_graph.json",
        script_dir / "prerequisites_graph.json",
        script_dir / "data" / "assist09" / "prerequisites_graph.json",
        script_dir.parent / "data" / "assist09" / "prerequisites_graph.json",
    ]

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved.is_file():
            return resolved

    return None


def align_concepts_with_graph(
    examples: Sequence[SequenceExample],
    concept_mapping: "OrderedDict[int, str]",
    graph_concepts: Sequence[str],
) -> Tuple[Sequence[SequenceExample], "OrderedDict[int, str]", List[int], Dict[str, object]]:
    name_to_graph_index: Dict[str, int] = {}
    lowered_to_graph_index: Dict[str, int] = {}
    for idx, name in enumerate(graph_concepts):
        name_to_graph_index[name] = idx
        lowered_to_graph_index[name.lower()] = idx

    aligned_names: List[str] = list(graph_concepts)
    extra_name_to_id: Dict[str, int] = {}
    skill_to_new: Dict[int, int] = {}
    missing_name_ids: List[int] = []
    unmapped_counts: Counter[str] = Counter()
    matched = 0

    def _register_extra(name: str) -> int:
        if name in extra_name_to_id:
            return extra_name_to_id[name]
        new_id = len(aligned_names)
        extra_name_to_id[name] = new_id
        aligned_names.append(name)
        return new_id

    placeholder_index = 0

    for old_id, raw_name in concept_mapping.items():
        name = raw_name.strip()
        if name:
            lookup = name_to_graph_index.get(name)
            if lookup is None:
                lookup = lowered_to_graph_index.get(name.lower())
            if lookup is not None:
                skill_to_new[old_id] = lookup
                matched += 1
                continue
            unmapped_counts[name] += 1
            new_id = _register_extra(name)
        else:
            placeholder = f"__missing_skill_name_{placeholder_index}"
            placeholder_index += 1
            new_id = _register_extra(placeholder)
            missing_name_ids.append(new_id)
        skill_to_new[old_id] = new_id

    for example in examples:
        example.skill_seq = [skill_to_new[sid] for sid in example.skill_seq]
        example.learning_goals = [skill_to_new[sid] for sid in example.learning_goals]

    ordered_mapping: "OrderedDict[int, str]" = OrderedDict(
        (idx, name) for idx, name in enumerate(aligned_names)
    )

    report = {
        "matched": matched,
        "introduced": len(aligned_names) - len(graph_concepts),
        "unmapped_counts": unmapped_counts,
    }

    return examples, ordered_mapping, missing_name_ids, report


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
    missing_mask |= lowered.eq("noskill")
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
            working[column] = default
            continue
        missing_mask = working[column].isna()
        count = int(missing_mask.sum())
        if not count:
            continue
        imputed[column] = imputed.get(column, 0) + count
        working.loc[missing_mask, column] = default

    report = {
        "dropped_rows": int(drop_mask.sum()),
        "dropped_by_column": drop_reason,
        "imputed_by_column": imputed,
    }
    return working, report

def _to_object_array(
    examples: Sequence[SequenceExample],
    getter,
    dtype: np.dtype,
) -> np.ndarray:
    return np.array([np.asarray(getter(example), dtype=dtype) for example in examples], dtype=object)

def write_npz(path: Path, examples: Sequence[SequenceExample], concepts: Dict[int, str]) -> None:
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
        "concept_ids": np.array(list(concepts.keys()), dtype=np.int32),
        "concept_names": np.array(list(concepts.values()), dtype=object),
    }

    np.savez_compressed(path, **data)


def write_mapping(path: Path, mapping: Dict[int, str]) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["concept_id", "concept_name"])
        for idx, name in mapping.items():
            writer.writerow([idx, name])


def ensure_assist2017_columns(df: pd.DataFrame, *, skill_name_column: str) -> pd.DataFrame:
    working = df.copy()

    if "attempt_count" not in working.columns:
        working["attempt_count"] = 1

    if "hint_count" not in working.columns:
        working["hint_count"] = 0

    if "assistmentId" in working.columns and "assistment_id" not in working.columns:
        working = working.rename(columns={"assistmentId": "assistment_id"})

    if "problemId" in working.columns and "problem_id" not in working.columns:
        working = working.rename(columns={"problemId": "problem_id"})

    if "correct" in working.columns:
        working["correct"] = working["correct"].apply(_coerce_to_int, args=(0,))

    if "assistment_id" in working.columns:
        working["assistment_id"] = working["assistment_id"].apply(_coerce_to_int, args=(-1,))

    if "problem_id" in working.columns:
        working["problem_id"] = working["problem_id"].apply(_coerce_to_int, args=(-1,))

    if "school_id" not in working.columns:
        working["school_id"] = -1

    if "startTime" in working.columns:
        working["startTime"] = working["startTime"].apply(_coerce_to_int, args=(0,))

    if "skill_id" not in working.columns:
        skill_series = working.get(skill_name_column)
        if skill_series is None:
            raise ValueError(
                "The dataset must contain a skill column to derive numeric identifiers"
            )
        encoded, uniques = pd.factorize(skill_series.astype(str).str.strip())
        encoded = encoded.astype(np.int64)
        missing_mask = encoded == -1
        if missing_mask.any():
            # Defer handling missing names to the regular pipeline.
            encoded = encoded.astype(object)
            encoded[missing_mask] = np.nan
            working["skill_id"] = encoded
        else:
            working["skill_id"] = encoded

    return working


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to the raw ASSISTments 2017 file", default="/home/zengxiangyu/SRC-py/Scripts/assist2017-cv-train-1.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store the processed dataset (defaults to the input directory)",
    )
    parser.add_argument("--dataset-name", type=str, default="assist2017")
    parser.add_argument(
        "--order-column",
        type=str,
        default="startTime",
        help="Column used to sort a learner's history",
    )
    parser.add_argument("--question-column", type=str, default="assistment_id")
    parser.add_argument("--attempt-column", type=str, default="attempt_count")
    parser.add_argument("--correctness-column", type=str, default="correct")
    parser.add_argument("--first-time-column", type=str, default="startTime")
    parser.add_argument("--response-time-column", type=str, default="startTime")
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
        help="Encoding of the raw file. When omitted, the script will try several common options",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="\t",
        help="Field delimiter used in the raw file (default: tab)",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional custom path for the concept-id mapping CSV",
    )
    parser.add_argument(
        "--prerequisite-graph",
        type=Path,
        default=None,
        help=(
            "Path to a prerequisite graph JSON used to align skill identifiers. "
            "When omitted, the script will search common locations for "
            "'prerequisites_graph.json'."
        ),
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
        delimiter=args.delimiter,
    )

    raw_df = ensure_assist2017_columns(raw_df, skill_name_column=args.skill_name_column)

    cleaned_df, missing_report = handle_missing_values(
        raw_df,
        strategy=args.missing_strategy,
        drop_columns=("user_id", args.order_column),
        fill_defaults={
            args.question_column: -1,
            args.attempt_column: 1,
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

    if args.skill_id_column not in cleaned_df.columns:
        raise ValueError(
            f"Column '{args.skill_id_column}' is required after preprocessing."
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

    graph_path = _resolve_prerequisite_graph(args.prerequisite_graph)
    alignment_report: Dict[str, object] | None = None
    graph_concepts: List[str] | None = None
    if graph_path is not None:
        try:
            graph_concepts = _load_graph_concepts(graph_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(
                "Warning: failed to read prerequisite graph %s (%s). Falling back to sequential remapping." %
                (graph_path, exc),
            )
            graph_concepts = None

    if graph_concepts:
        examples, reindexed_mapping, missing_name_ids, alignment_report = align_concepts_with_graph(
            examples,
            mapping,
            graph_concepts,
        )
    else:
        examples, reindexed_mapping, missing_name_ids = reindex_concepts(examples, mapping)

    dataset_path = output_dir / f"{args.dataset_name}.npz"
    mapping_path = args.mapping_file
    if mapping_path is None:
        mapping_path = output_dir / "knowledge_concept_mapping_assist2017.csv"
    elif not mapping_path.is_absolute():
        mapping_path = output_dir / mapping_path

    write_npz(dataset_path, examples, reindexed_mapping)
    write_mapping(mapping_path, reindexed_mapping)

    print(f"Saved processed dataset to {dataset_path}")
    print(f"Saved concept mapping to {mapping_path}")
    if graph_concepts is not None and graph_path is not None and alignment_report is not None:
        print(
            "Aligned skills to prerequisite graph %s (matched %d concepts, introduced %d new concepts)."
            % (
                graph_path,
                alignment_report.get("matched", 0),
                alignment_report.get("introduced", 0),
            )
        )
        unmapped_counts: Counter[str] = alignment_report.get("unmapped_counts", Counter())  # type: ignore[assignment]
        if unmapped_counts:
            top_unmapped = ", ".join(
                f"{name} ({count})" for name, count in unmapped_counts.most_common(5)
            )
            print(
                "Warning: %d concept names were not present in the graph (top entries: %s)"
                % (sum(unmapped_counts.values()), top_unmapped)
            )
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
        if graph_concepts is not None and graph_path is not None and alignment_report is not None:
            unmapped_counts: Counter[str] = alignment_report.get("unmapped_counts", Counter())  # type: ignore[assignment]
            summary["graph_alignment"] = {
                "graph_path": str(graph_path),
                "matched": alignment_report.get("matched", 0),
                "introduced": alignment_report.get("introduced", 0),
                "unmapped_names": {name: int(count) for name, count in unmapped_counts.items()},
            }
        summary_path = output_dir / "dataset_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
        print(f"Saved dataset summary to {summary_path}")


if __name__ == "__main__":
    main()