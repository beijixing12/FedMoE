from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def _resolve_npz_key(
    available: Iterable[str], primary: str, alternatives: Sequence[str]
) -> Optional[str]:
    """Return the first matching key from ``available``."""

    keys = list(available)
    if primary in keys:
        return primary
    for candidate in alternatives:
        if candidate in keys:
            return candidate
    return None


def _safe_int(value: object) -> Optional[int]:
    """Best-effort conversion to ``int`` returning ``None`` on failure."""

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalise_sequence(values: np.ndarray, length: int) -> np.ndarray:
    """Truncate or pad ``values`` to ``length`` entries."""

    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.size >= length:
        return array[:length]
    padded = np.empty(length, dtype=array.dtype)
    padded[: array.size] = array
    padded[array.size :] = -1
    return padded


def _normalise_concept_label(
    raw: object,
    concept_labels: Dict[int, str],
    known_concepts: MutableMapping[str, None],
    *,
    allow_new: bool = True,
) -> Optional[str]:
    maybe_id = _safe_int(raw)
    if maybe_id is not None and maybe_id in concept_labels:
        label = concept_labels[maybe_id]
        if allow_new or label in known_concepts:
            known_concepts.setdefault(label, None)
            return label
        return label if label in known_concepts else None

    label = str(raw)
    if allow_new or label in known_concepts:
        known_concepts.setdefault(label, None)
        return label
    return label if label in known_concepts else None


def build_triplet_graph(
    npz_path: Path,
    *,
    require_exercises: bool = False,
) -> Tuple[
    Dict[int, str],
    OrderedDict[str, None],
    OrderedDict[str, None],
    OrderedDict[str, None],
    Dict[Tuple[str, str, str], int],
    int,
]:
    """Aggregate learner interactions from ``npz_path`` into triplets.

    Returns a tuple containing (concept label lookup, ordered concepts,
    ordered exercises, ordered learners, triplet counts, skipped count).
    """

    with np.load(npz_path, allow_pickle=True) as data:
        files = data.files

        skill_key = _resolve_npz_key(files, "skill", ("skills", "skill_seq"))
        if skill_key is None:
            raise KeyError(
                "The dataset archive is missing a 'skill' column required to build the graph."
            )

        user_key = _resolve_npz_key(
            files,
            "user_id",
            ("student_id", "anon_student_id", "Anon Student Id", "user", "learner_id"),
        )
        if user_key is None:
            raise KeyError(
                "The dataset archive is missing a 'user_id' column required to build the graph."
            )

        exercise_key = _resolve_npz_key(
            files,
            "question_id",
            ("problem_id", "exercise_id", "item_id", "problemId", "question_id_seq"),
        )
        if exercise_key is None and require_exercises:
            raise KeyError(
                "The dataset archive does not contain an exercise identifier column. "
                "Re-run without --require-exercises if this is expected."
            )

        length_key = _resolve_npz_key(files, "real_len", ("length", "seq_len", "sequence_length"))

        concept_ids = None
        concept_names = None
        if "concept_ids" in files:
            concept_ids = np.asarray(data["concept_ids"], dtype=np.int64)
        if "concept_names" in files:
            concept_names = [str(name) for name in data["concept_names"]]

        raw_skills = data[skill_key]
        raw_users = data[user_key]
        raw_exercises = data[exercise_key] if exercise_key else None
        raw_lengths = data[length_key] if length_key else None

    concept_labels: Dict[int, str] = {}
    if concept_ids is not None and concept_names and len(concept_names) == len(concept_ids):
        for cid, name in zip(concept_ids, concept_names):
            concept_labels[int(cid)] = name
    elif concept_ids is not None:
        for cid in concept_ids:
            concept_labels[int(cid)] = str(int(cid))

    concepts_order: OrderedDict[str, None] = OrderedDict()
    exercises_order: OrderedDict[str, None] = OrderedDict()
    learners_order: OrderedDict[str, None] = OrderedDict()

    triplet_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    skipped = 0

    for label in concept_labels.values():
        concepts_order.setdefault(label, None)
    skills = [np.asarray(seq, dtype=object) if seq is not None else None for seq in raw_skills]
    exercises = (
        [np.asarray(seq, dtype=object) if seq is not None else None for seq in raw_exercises]
        if raw_exercises is not None
        else None
    )
    lengths = (
        np.asarray(raw_lengths, dtype=np.int64)
        if raw_lengths is not None
        else None
    )

    for seq_idx, skill_seq in enumerate(skills):
        if skill_seq is None:
            skipped += 1
            continue

        learner_raw = _safe_int(raw_users[seq_idx])
        if learner_raw is None:
            skipped += 1
            continue
        learner_label = str(learner_raw)
        learners_order.setdefault(learner_label, None)

        seq_len = int(lengths[seq_idx]) if lengths is not None else len(skill_seq)
        if seq_len <= 0:
            skipped += 1
            continue

        norm_skills = _normalise_sequence(skill_seq, seq_len)
        norm_exercises = None
        if exercises is not None:
            exercise_seq = exercises[seq_idx]
            if exercise_seq is not None:
                norm_exercises = _normalise_sequence(exercise_seq, seq_len)

        for pos in range(seq_len):
            concept_id = _safe_int(norm_skills[pos])
            if concept_id is None or concept_id < 0:
                continue

            concept_label = concept_labels.get(concept_id)
            if concept_label is None:
                concept_label = str(concept_id)
                concept_labels[concept_id] = concept_label
            concepts_order.setdefault(concept_label, None)

            exercise_label = None
            if norm_exercises is not None:
                exercise_id = _safe_int(norm_exercises[pos])
                if exercise_id is not None and exercise_id >= 0:
                    exercise_label = str(exercise_id)
                    exercises_order.setdefault(exercise_label, None)

            if exercise_label is None:
                if require_exercises:
                    continue
                exercise_label = "__missing_exercise__"
                exercises_order.setdefault(exercise_label, None)

            triplet_counts[(concept_label, exercise_label, learner_label)] += 1

    if not triplet_counts:
        raise ValueError(
            "No valid learner–exercise–concept interactions found. Verify the dataset formatting."
        )

    return (
        concept_labels,
        concepts_order,
        exercises_order,
        learners_order,
        triplet_counts,
        skipped,
    )


def load_similarity_edges(
    path: Optional[Path],
    *,
    concept_labels: Dict[int, str],
    concepts_order: MutableMapping[str, None],
) -> Tuple[Sequence[Dict[str, object]], int]:
    skipped = 0
    if path is None:
        return [], skipped

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw_edges = data.get("similarity_edges", data.get("edges", []))
    edges = []
    for entry in raw_edges:
        try:
            raw_u = entry["u"]
            raw_v = entry["v"]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Similarity edge missing key: {exc.args[0]}") from exc

        u_label = _normalise_concept_label(
            raw_u, concept_labels, concepts_order, allow_new=False
        )
        v_label = _normalise_concept_label(
            raw_v, concept_labels, concepts_order, allow_new=False
        )

        if u_label is None or v_label is None:
            skipped += 1
            continue

        confidence = entry.get("confidence")
        weight = entry.get("weight", confidence if confidence is not None else 1.0)
        edge_record: Dict[str, object] = {"u": u_label, "v": v_label, "weight": float(weight)}
        if confidence is not None:
            edge_record["confidence"] = float(confidence)
        edges.append(edge_record)
    return edges, skipped


def dump_graph(
    output_path: Path,
    concepts: Sequence[str],
    exercises: Sequence[str],
    learners: Sequence[str],
    triplet_counts: Dict[Tuple[str, str, str], int],
    similarity_edges: Sequence[Dict[str, object]],
    *,
    indent: Optional[int] = 2,
) -> None:
    triplets = [
        {"concept": c, "exercise": e, "learner": l, "weight": w}
        for (c, e, l), w in sorted(triplet_counts.items())
    ]

    payload = {
        "concepts": list(concepts),
        "exercises": list(exercises),
        "learners": list(learners),
        "triplets": triplets,
    }
    if similarity_edges:
        payload["similarity_edges"] = list(similarity_edges)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build heterogeneous graphs from SRC datasets.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the processed dataset (.npz)")
    parser.add_argument(
        "--similarity",
        type=Path,
        help="Optional JSON containing concept similarity edges to merge into the output graph.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON file. Defaults to <dataset>_hetero_graph.json alongside the input dataset.",
    )
    parser.add_argument(
        "--require-exercises",
        action="store_true",
        help="Skip interactions without exercise identifiers instead of using a placeholder.",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Disable JSON pretty-printing to reduce file size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = (
        args.output
        if args.output is not None
        else args.input.with_name(f"{args.input.stem}_hetero_graph.json")
    )

    (
        concept_labels,
        concepts_order,
        exercises_order,
        learners_order,
        triplet_counts,
        skipped,
    ) = build_triplet_graph(args.input, require_exercises=args.require_exercises)

    similarity_edges, skipped_similarity = load_similarity_edges(
        args.similarity, concept_labels=concept_labels, concepts_order=concepts_order
    )

    dump_graph(
        output_path,
        concepts_order.keys(),
        exercises_order.keys(),
        learners_order.keys(),
        triplet_counts,
        similarity_edges,
        indent=None if args.no_pretty else 2,
    )

    print(f"Saved heterogeneous graph to {output_path}")
    print(f"Concepts: {len(concepts_order)}, Exercises: {len(exercises_order)}, Learners: {len(learners_order)}")
    print(f"Triplets: {len(triplet_counts)}")
    if skipped:
        print(f"Skipped sequences due to missing data: {skipped}")
    if similarity_edges:
        print(f"Merged {len(similarity_edges)} similarity edges")

    if skipped_similarity:
        print(
            f"Skipped {skipped_similarity} similarity edges referencing unknown concepts"
        )

if __name__ == "__main__":
    main()

# python build_hetero_graph.py --input /home/zengxiangyu/SRC-py/data/assist09/assist09.npz --similarity /home/zengxiangyu/SRC-py/data/assist09/similarity_graph.json
# python build_hetero_graph.py --input /home/zengxiangyu/SRC-py/data/assist12/assist12.npz --similarity /home/zengxiangyu/SRC-py/data/assist12/similarity_graph.json
# python build_hetero_graph.py --input /home/zengxiangyu/SRC-py/data/OLI/OLI.npz --similarity /home/zengxiangyu/SRC-py/data/OLI/similarity_edges.json