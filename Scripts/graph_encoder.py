from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import warnings
import torch
from torch import Tensor, nn
import math
from Scripts.prerequisite_dgcn import (
    PrerequisiteDGCN,
    PrerequisiteGraph,
    build_proximities,
    load_prerequisite_graph,
)

import numpy as np

@dataclass(frozen=True)
class SimilarityGraph:
    """In-memory representation of an undirected similarity graph."""

    concepts: List[str]
    adjacency: Tensor

    @property
    def device(self) -> torch.device:
        return self.adjacency.device

    @property
    def size(self) -> int:
        return self.adjacency.size(0)

@dataclass(frozen=True)
class TripletHeteroGraph:
    """Learner–exercise–concept heterogeneous graph."""

    concepts: List[str]
    exercises: List[str]
    learners: List[str]
    concept_exercise: Tensor
    concept_learner: Tensor

    @property
    def num_concepts(self) -> int:
        return self.concept_exercise.size(0)

    @property
    def num_exercises(self) -> int:
        return self.concept_exercise.size(1)

    @property
    def num_learners(self) -> int:
        return self.concept_learner.size(1)


def _expand_prerequisite_graph(
    graph: PrerequisiteGraph, target_size: int
) -> PrerequisiteGraph:
    """Pad a prerequisite graph so it matches ``target_size`` nodes."""

    if graph.size > target_size:
        raise ValueError(
            "Prerequisite graph has more nodes than skill_num: "
            f"{graph.size} > {target_size}"
        )
    if graph.size == target_size:
        return graph

    padding = target_size - graph.size
    warnings.warn(
        "Prerequisite graph size does not match skill_num; "
        f"padding with {padding} isolated nodes.",
        RuntimeWarning,
    )
    padded_adjacency = graph.adjacency.new_zeros((target_size, target_size))
    padded_adjacency[: graph.size, : graph.size] = graph.adjacency
    padded_concepts = graph.concepts + [
        f"__missing_prereq_{idx}" for idx in range(padding)
    ]
    return PrerequisiteGraph(padded_concepts, padded_adjacency)


def _expand_similarity_graph(
    graph: SimilarityGraph, target_size: int
) -> SimilarityGraph:
    """Pad a similarity graph so it matches ``target_size`` nodes."""

    if graph.size > target_size:
        raise ValueError(
            "Similarity graph has more nodes than skill_num: "
            f"{graph.size} > {target_size}"
        )
    if graph.size == target_size:
        return graph

    padding = target_size - graph.size
    warnings.warn(
        "Similarity graph size does not match skill_num; "
        f"padding with {padding} isolated nodes.",
        RuntimeWarning,
    )
    padded_adjacency = graph.adjacency.new_zeros((target_size, target_size))
    padded_adjacency[: graph.size, : graph.size] = graph.adjacency
    padded_concepts = graph.concepts + [
        f"__missing_similarity_{idx}" for idx in range(padding)
    ]
    return SimilarityGraph(padded_concepts, padded_adjacency)

def _expand_triplet_graph(
    graph: TripletHeteroGraph, target_size: int
) -> TripletHeteroGraph:
    """Pad a heterogeneous graph to ``target_size`` concepts."""

    if graph.num_concepts > target_size:
        raise ValueError(
            "Triplet hetero graph has more nodes than skill_num: "
            f"{graph.num_concepts} > {target_size}"
        )
    if graph.num_concepts == target_size:
        return graph

    padding = target_size - graph.num_concepts
    warnings.warn(
        "Triplet hetero graph size does not match skill_num; "
        f"padding with {padding} isolated nodes.",
        RuntimeWarning,
    )

    padded_concepts = graph.concepts + [
        f"__missing_triplet_{idx}" for idx in range(padding)
    ]
    concept_exercise = graph.concept_exercise
    concept_learner = graph.concept_learner

    if concept_exercise.size(0) != graph.num_concepts:
        raise ValueError("concept_exercise has inconsistent first dimension")
    if concept_learner.size(0) != graph.num_concepts:
        raise ValueError("concept_learner has inconsistent first dimension")

    padded_concept_exercise = concept_exercise.new_zeros(
        (target_size, concept_exercise.size(1))
    )
    padded_concept_exercise[: graph.num_concepts] = concept_exercise

    padded_concept_learner = concept_learner.new_zeros(
        (target_size, concept_learner.size(1))
    )
    padded_concept_learner[: graph.num_concepts] = concept_learner

    return TripletHeteroGraph(
        padded_concepts,
        graph.exercises,
        graph.learners,
        padded_concept_exercise,
        padded_concept_learner,
    )

def load_similarity_graph(
    json_path: Path | str,
    *,
    confidence_threshold: float = 0.0,
    normalise_confidence: bool = True,
) -> SimilarityGraph:
    """Load a similarity graph from ``json_path``.

    The JSON file is expected to contain a ``"similarity_edges"`` array
    where each entry provides ``u`` (first concept), ``v`` (second
    concept), ``relation`` (kept for completeness), ``confidence`` and an
    optional ``weight``.  Edges are treated as undirected and stored in a
    dense adjacency matrix.
    """

    graph_path = Path(json_path)
    with graph_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    edges: Iterable[Mapping[str, object]] = data.get("similarity_edges", [])

    concepts: MutableMapping[str, int] = {}
    filtered_edges: List[Tuple[str, str, float]] = []
    max_confidence = 0.0

    for edge in edges:
        head = str(edge["u"])
        tail = str(edge["v"])
        confidence = float(edge.get("confidence", 1.0))
        if confidence < confidence_threshold:
            continue
        weight = float(edge.get("weight", confidence))
        filtered_edges.append((head, tail, weight))
        max_confidence = max(max_confidence, weight)
        for concept in (head, tail):
            if concept not in concepts:
                concepts[concept] = len(concepts)

    if not concepts:
        raise ValueError(f"No similarity edges found in {graph_path!s}.")

    scale = 1.0 / max_confidence if normalise_confidence and max_confidence else 1.0
    num_nodes = len(concepts)
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for head, tail, weight in filtered_edges:
        i = concepts[head]
        j = concepts[tail]
        scaled_weight = weight * scale
        adjacency[i, j] += scaled_weight
        adjacency[j, i] += scaled_weight

    ordered_concepts: List[str] = [""] * num_nodes
    for name, index in concepts.items():
        ordered_concepts[index] = name

    return SimilarityGraph(ordered_concepts, adjacency)

def _resolve_index(
    value: object,
    names: Sequence[str],
    mapping: MutableMapping[str, int],
) -> int:
    if isinstance(value, int):
        if 0 <= value < len(names):
            return value
        raise ValueError(f"Index {value} out of range for sequence of length {len(names)}")
    key = str(value)
    if key in mapping:
        return mapping[key]
    if key not in names:
        mapping[key] = len(names) + len(mapping)
        return mapping[key]
    return names.index(key)


def _load_triplet_graph_from_json(json_path: Path) -> TripletHeteroGraph:
    """Load a learner–exercise–concept heterogeneous graph from JSON."""

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    concepts: List[str] = list(map(str, data.get("concepts", [])))
    exercises: List[str] = list(map(str, data.get("exercises", [])))
    learners: List[str] = list(map(str, data.get("learners", [])))

    concept_aliases: MutableMapping[str, int] = {}
    exercise_aliases: MutableMapping[str, int] = {}
    learner_aliases: MutableMapping[str, int] = {}

    triplets: Iterable[Mapping[str, object]] = data.get("triplets", data.get("interactions", []))
    if not triplets:
        raise ValueError(f"No triplets found in {graph_path!s}.")

    # Build adjacency lists
    concept_exercise_edges: List[Tuple[int, int, float]] = []
    concept_learner_edges: List[Tuple[int, int, float]] = []

    for triplet in triplets:
        try:
            concept_value = triplet["concept"]
            exercise_value = triplet["exercise"]
            learner_value = triplet["learner"]
        except KeyError as exc:
            raise KeyError(f"Triplet missing required key: {exc.args[0]}") from exc

        concept_idx = _resolve_index(concept_value, concepts, concept_aliases)
        exercise_idx = _resolve_index(exercise_value, exercises, exercise_aliases)
        learner_idx = _resolve_index(learner_value, learners, learner_aliases)

        weight = float(triplet.get("weight", 1.0))
        concept_exercise_edges.append((concept_idx, exercise_idx, weight))
        concept_learner_edges.append((concept_idx, learner_idx, weight))

    if concept_aliases:
        for name, idx in concept_aliases.items():
            if idx >= len(concepts):
                concepts.append(name)
    if exercise_aliases:
        for name, idx in exercise_aliases.items():
            if idx >= len(exercises):
                exercises.append(name)
    if learner_aliases:
        for name, idx in learner_aliases.items():
            if idx >= len(learners):
                learners.append(name)

    num_concepts = len(concepts)
    num_exercises = len(exercises)
    num_learners = len(learners)

    concept_exercise = torch.zeros((num_concepts, num_exercises), dtype=torch.float32)
    concept_learner = torch.zeros((num_concepts, num_learners), dtype=torch.float32)

    for concept_idx, exercise_idx, weight in concept_exercise_edges:
        if exercise_idx >= num_exercises:
            raise IndexError("Exercise index exceeded resolved range")
        concept_exercise[concept_idx, exercise_idx] += weight

    for concept_idx, learner_idx, weight in concept_learner_edges:
        if learner_idx >= num_learners:
            raise IndexError("Learner index exceeded resolved range")
        concept_learner[concept_idx, learner_idx] += weight

    return TripletHeteroGraph(
        concepts,
        exercises,
        learners,
        concept_exercise,
        concept_learner,
    )

def _resolve_npz_key(
    available: Iterable[str],
    primary: str,
    alternatives: Sequence[str],
) -> Optional[str]:
    candidates = list(available)
    if primary in candidates:
        return primary
    for candidate in alternatives:
        if candidate in candidates:
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


def _load_triplet_graph_from_npz(npz_path: Path) -> TripletHeteroGraph:
    """Construct a heterogeneous graph directly from a dataset NPZ archive."""

    with np.load(npz_path, allow_pickle=True) as data:
        files = data.files

        skill_key = _resolve_npz_key(files, "skill", ("skills", "skill_seq"))
        if skill_key is None:
            raise KeyError(
                "The dataset archive does not contain a 'skill' column required"
                " to build the heterogeneous graph."
            )

        user_key = _resolve_npz_key(
            files,
            "user_id",
            ("student_id", "anon_student_id", "Anon Student Id", "user"),
        )
        if user_key is None:
            raise KeyError(
                "The dataset archive does not contain a 'user_id' column required"
                " to build the heterogeneous graph."
            )

        exercise_key = _resolve_npz_key(
            files,
            "question_id",
            ("problem_id", "exercise_id", "item_id"),
        )

        length_key = _resolve_npz_key(files, "real_len", ("length", "seq_len"))

        concept_ids = None
        concept_names = None
        if "concept_ids" in files:
            concept_ids = np.asarray(data["concept_ids"], dtype=np.int64)
        if "concept_names" in files:
            concept_names = [str(name) for name in data["concept_names"]]

        raw_skills = data[skill_key]
        skills = [np.asarray(seq).copy() if seq is not None else None for seq in raw_skills]
        users = np.asarray(data[user_key], dtype=np.int64).copy()
        if exercise_key is not None:
            raw_exercises = data[exercise_key]
            exercises = [np.asarray(seq).copy() if seq is not None else None for seq in raw_exercises]
        else:
            exercises = None
        real_lens = (
            np.asarray(data[length_key], dtype=np.int64).copy() if length_key else None
        )

    if concept_ids is None:
        unique_concepts = set()
        for seq in skills:
            if seq is None:
                continue
            for raw in np.asarray(seq).reshape(-1):
                concept_val = _safe_int(raw)
                if concept_val is None or concept_val < 0:
                    continue
                unique_concepts.add(concept_val)
        ordered_concepts = sorted(unique_concepts)
    else:
        ordered_concepts = [int(x) for x in concept_ids]

    if concept_names and len(concept_names) == len(ordered_concepts):
        concept_labels = list(concept_names)
    else:
        concept_labels = [str(cid) for cid in ordered_concepts]

    concept_index = {cid: idx for idx, cid in enumerate(ordered_concepts)}

    exercise_index: MutableMapping[int, int] = {}
    learner_index: MutableMapping[int, int] = {}
    concept_exercise_edges: List[Tuple[int, int]] = []
    concept_learner_edges: List[Tuple[int, int]] = []

    for seq_idx, skill_seq in enumerate(skills):
        if skill_seq is None:
            continue
        seq_len = int(real_lens[seq_idx]) if real_lens is not None else len(skill_seq)
        if seq_len <= 0:
            continue

        skill_values = _normalise_sequence(skill_seq, seq_len)
        if exercises is not None:
            exercise_entry = exercises[seq_idx]
            exercise_values = (
                _normalise_sequence(exercise_entry, seq_len)
                if exercise_entry is not None
                else None
            )
        else:
            exercise_values = None

        learner_raw = _safe_int(users[seq_idx])
        if learner_raw is None:
            continue
        learner_idx = learner_index.setdefault(learner_raw, len(learner_index))

        for pos in range(seq_len):
            concept_raw = _safe_int(skill_values[pos])
            if concept_raw is None or concept_raw < 0:
                continue
            concept_pos = concept_index.get(concept_raw)
            if concept_pos is None:
                continue
            concept_learner_edges.append((concept_pos, learner_idx))
            if exercise_values is None:
                continue
            exercise_raw = _safe_int(exercise_values[pos])
            if exercise_raw is None or exercise_raw < 0:
                continue
            exercise_idx = exercise_index.setdefault(exercise_raw, len(exercise_index))
            concept_exercise_edges.append((concept_pos, exercise_idx))

    num_concepts = len(concept_index)
    num_exercises = len(exercise_index)
    num_learners = len(learner_index)

    concept_exercise = torch.zeros((num_concepts, num_exercises), dtype=torch.float32)
    concept_learner = torch.zeros((num_concepts, num_learners), dtype=torch.float32)

    for concept_idx, exercise_idx in concept_exercise_edges:
        concept_exercise[concept_idx, exercise_idx] += 1.0

    for concept_idx, learner_idx in concept_learner_edges:
        concept_learner[concept_idx, learner_idx] += 1.0

    exercises: List[str] = [""] * num_exercises
    for raw_id, idx in exercise_index.items():
        exercises[idx] = str(raw_id)

    learners: List[str] = [""] * num_learners
    for raw_id, idx in learner_index.items():
        learners[idx] = str(raw_id)

    return TripletHeteroGraph(
        [str(label) for label in concept_labels],
        exercises,
        learners,
        concept_exercise,
        concept_learner,
    )


def load_triplet_graph(path: Path | str) -> TripletHeteroGraph:
    """Load a learner–exercise–concept heterogeneous graph."""

    graph_path = Path(path)
    suffix = graph_path.suffix.lower()
    if suffix == ".npz":
        return _load_triplet_graph_from_npz(graph_path)
    if suffix == ".json":
        warnings.warn(
            "Loading heterogeneous graphs from JSON is deprecated; "
            "please provide the dataset NPZ to build the triplet graph.",
            DeprecationWarning,
        )
        return _load_triplet_graph_from_json(graph_path)
    raise ValueError(f"Unsupported heterogeneous graph format: {graph_path.suffix}")

def _symmetric_normalise(matrix: Tensor) -> Tensor:
    degree = matrix.sum(dim=1)
    inv_sqrt = torch.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = degree[mask].pow(-0.5)
    return inv_sqrt.unsqueeze(1) * matrix * inv_sqrt.unsqueeze(0)


class LightGCNEncoder(nn.Module):
    """LightGCN propagation on an undirected similarity graph."""

    def __init__(
        self,
        adjacency: Tensor,
        num_layers: int,
        *,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError("adjacency must be a square matrix")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")

        if add_self_loops:
            identity = torch.eye(adjacency.size(0), dtype=adjacency.dtype, device=adjacency.device)
            adjacency = adjacency + identity

        self.num_layers = num_layers
        self.register_buffer("normalised_adjacency", _symmetric_normalise(adjacency))

    def forward(self, features: Tensor) -> Tensor:
        if features.dim() != 2:
            raise ValueError("features must have shape (num_nodes, feature_dim)")
        if features.size(0) != self.normalised_adjacency.size(0):
            raise ValueError("Mismatch between feature nodes and adjacency size")

        propagation = features
        accumulated = features
        normalised = self.normalised_adjacency.to(features.device, dtype=features.dtype)

        for _ in range(self.num_layers):
            propagation = normalised @ propagation
            accumulated = accumulated + propagation

        return accumulated / (self.num_layers + 1)

class MetaPathHeteroEncoder(nn.Module):
    """Meta-path based propagation on heterogeneous concept graphs."""

    def __init__(
        self,
        meta_path_adjacencies: Sequence[Tensor],
        num_layers: int,
        *,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if not meta_path_adjacencies:
            raise ValueError("meta_path_adjacencies must not be empty")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")

        self.num_layers = num_layers
        normalised: List[Tensor] = []
        for adjacency in meta_path_adjacencies:
            if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
                raise ValueError("Meta-path adjacency must be a square matrix")
            matrix = adjacency.to(dtype=torch.float32)
            if add_self_loops:
                identity = torch.eye(matrix.size(0), dtype=matrix.dtype)
                matrix = matrix + identity
            normalised.append(_symmetric_normalise(matrix))

        for idx, matrix in enumerate(normalised):
            self.register_buffer(f"_meta_path_{idx}", matrix)

        self._num_paths = len(normalised)
        self.path_logits = nn.Parameter(torch.zeros(self._num_paths))

    def _meta_path_matrices(self) -> List[Tensor]:
        return [getattr(self, f"_meta_path_{idx}") for idx in range(self._num_paths)]

    def forward(self, features: Tensor) -> Tensor:
        if features.dim() != 2:
            raise ValueError("features must have shape (num_nodes, feature_dim)")

        matrices = [
            matrix.to(features.device, dtype=features.dtype)
            for matrix in self._meta_path_matrices()
        ]
        weights = torch.softmax(self.path_logits, dim=0)
        aggregated = torch.zeros_like(features)

        for weight, adjacency in zip(weights, matrices):
            propagation = features
            accumulated = features
            for _ in range(self.num_layers):
                propagation = adjacency @ propagation
                accumulated = accumulated + propagation
            accumulated = accumulated / (self.num_layers + 1)
            aggregated = aggregated + weight * accumulated

        return aggregated

class GraphFusionEncoder(nn.Module):
    """Fuse DGCN and LightGCN representations into skill embeddings."""

    def __init__(
        self,
        skill_num: int,
        embedding_dim: int,
        *,
        prerequisite_graph: PrerequisiteGraph,
        similarity_graph: SimilarityGraph,
        hetero_graph: Optional[TripletHeteroGraph] = None,
        dgcn_layers: int = 2,
        lightgcn_layers: int = 2,
        fusion_weight: float = 0.5,
        dgcn_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        prerequisite_graph = _expand_prerequisite_graph(prerequisite_graph, skill_num)
        similarity_graph = _expand_similarity_graph(similarity_graph, skill_num)
        if not (0.0 <= fusion_weight <= 1.0):
            raise ValueError("fusion_weight must be between 0 and 1")

        if hetero_graph is not None:
            hetero_graph = _expand_triplet_graph(hetero_graph, skill_num)

        self.embedding = nn.Embedding(skill_num, embedding_dim)

        proximities = build_proximities(prerequisite_graph)
        self._dgcn_proximity_count = len(proximities)
        for idx, matrix in enumerate(proximities):
            self.register_buffer(f"_dgcn_proximity_{idx}", matrix)

        self.dgcn = PrerequisiteDGCN(
            num_features=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=dgcn_layers,
            dropout=dgcn_dropout,
        )
        base_similarity = similarity_graph.adjacency.to(dtype=torch.float32)
        meta_path_adjacencies: List[Tensor] = []

        if hetero_graph is not None:
            concept_exercise = hetero_graph.concept_exercise.to(dtype=torch.float32)
            concept_learner = hetero_graph.concept_learner.to(dtype=torch.float32)

            def _append_if_nonzero(matrix: Tensor) -> None:
                if matrix.numel() == 0:
                    return
                if torch.count_nonzero(matrix) == 0:
                    return
                meta_path_adjacencies.append(matrix)

            if concept_exercise.numel() and concept_learner.numel():
                learner_exercise = concept_learner.t() @ concept_exercise
                exercise_learner = concept_exercise.t() @ concept_learner
                exercise_concept = concept_exercise.t()

                # L–E–C–C'–E–L meta-path incorporating similarity edges.
                temp = concept_learner @ learner_exercise
                temp = temp @ exercise_concept
                temp = temp @ base_similarity
                temp = temp @ concept_exercise
                temp = temp @ exercise_learner
                path_leccel = temp @ concept_learner.t()
                _append_if_nonzero(path_leccel)

                # L–E–C–E–L meta-path projected back to concepts.
                temp = concept_learner @ learner_exercise
                temp = temp @ exercise_concept
                temp = temp @ concept_exercise
                temp = temp @ exercise_learner
                path_lecel = temp @ concept_learner.t()
                _append_if_nonzero(path_lecel)

            # E–C–E meta-path projected back to concepts.
            if concept_exercise.numel():
                exercise_adj = concept_exercise.t() @ concept_exercise
                path_ece = concept_exercise @ exercise_adj @ concept_exercise.t()
                _append_if_nonzero(path_ece)

        # C–C' meta-path (similarity graph) is always included.
        meta_path_adjacencies.append(base_similarity)

        self.hetero_encoder = MetaPathHeteroEncoder(
            meta_path_adjacencies,
            lightgcn_layers,
        )

        self.fusion_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        nn.init.zeros_(self.fusion_gate.weight)
        if 0.0 < fusion_weight < 1.0:
            bias_value = math.log(fusion_weight / (1.0 - fusion_weight))
            self.fusion_gate.bias.data.fill_(bias_value)
        else:
            nn.init.zeros_(self.fusion_gate.bias)

    def _dgcn_proximities(self) -> List[Tensor]:
        return [
            getattr(self, f"_dgcn_proximity_{idx}")
            for idx in range(self._dgcn_proximity_count)
        ]

    def forward(self) -> Tensor:
        base_features = self.embedding.weight
        proximities = [
            matrix.to(base_features.device, dtype=base_features.dtype)
            for matrix in self._dgcn_proximities()
        ]
        dgcn_embeddings = self.dgcn(base_features, proximities)
        hetero_embeddings = self.hetero_encoder(base_features)
        gate_input = torch.cat([dgcn_embeddings, hetero_embeddings], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(gate_input))
        return gate * dgcn_embeddings + (1.0 - gate) * hetero_embeddings


__all__ = [
    "GraphFusionEncoder",
    "LightGCNEncoder",
    "SimilarityGraph",
    "TripletHeteroGraph",
    "load_prerequisite_graph",
    "load_similarity_graph",
    "load_triplet_graph",
]
