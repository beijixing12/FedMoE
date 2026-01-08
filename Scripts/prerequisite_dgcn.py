from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PrerequisiteGraph:
    """In-memory representation of a directed prerequisite graph.

    Attributes:
        concepts: A list of concept names ordered according to the matrix
            representation.
        adjacency: A dense adjacency matrix where ``adjacency[i, j]`` stores
            the confidence-weighted strength of the relation
            ``concepts[i] -> concepts[j]``.
    """

    concepts: List[str]
    adjacency: Tensor

    @property
    def device(self) -> torch.device:
        return self.adjacency.device

    @property
    def size(self) -> int:
        return self.adjacency.size(0)


def load_prerequisite_graph(
    json_path: Path | str | None,
    *,
    confidence_threshold: float = 0.0,
    normalise_confidence: bool = True,
) -> PrerequisiteGraph:
    """Load prerequisite relations from ``json_path``.

    The JSON file is expected to contain a ``"prerequisite_edges"`` array where
    each entry exposes ``head`` (source concept), ``tail`` (destination
    concept), ``relation`` (kept for completeness), ``confidence`` and an
    optional ``weight``.  Confidence scores below ``confidence_threshold`` are
    ignored.  Edge strengths are stored as floating point numbers where the
    highest confidence edge inside the file is mapped to ``1.0`` if
    ``normalise_confidence`` is true.
    """
    if json_path is None:
        raise FileNotFoundError(
            "No prerequisite graph path was provided. Ensure the graph files are present "
            "or pass --prerequisite_graph to specify their location explicitly."
        )
    graph_path = Path(json_path)
    with graph_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    edges: Iterable[Mapping[str, object]] = data.get("prerequisite_edges", [])

    concepts: MutableMapping[str, int] = {}
    filtered_edges: List[Tuple[str, str, float]] = []
    max_confidence = 0.0

    for edge in edges:
        head = str(edge["head"])
        tail = str(edge["tail"])
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
        raise ValueError(f"No prerequisite edges found in {graph_path!s}.")

    scale = 1.0 / max_confidence if normalise_confidence and max_confidence else 1.0
    num_nodes = len(concepts)
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for head, tail, weight in filtered_edges:
        i = concepts[head]
        j = concepts[tail]
        adjacency[i, j] = weight * scale

    ordered_concepts: List[str] = [""] * num_nodes
    for name, index in concepts.items():
        ordered_concepts[index] = name

    return PrerequisiteGraph(ordered_concepts, adjacency)


def _degree_inv_sqrt(values: Tensor) -> Tensor:
    inv_sqrt = torch.zeros_like(values)
    mask = values > 0
    inv_sqrt[mask] = values[mask].pow(-0.5)
    return inv_sqrt


def _degree_inv(values: Tensor) -> Tensor:
    inv = torch.zeros_like(values)
    mask = values > 0
    inv[mask] = values[mask].reciprocal()
    return inv


def first_order_proximity(adjacency: Tensor) -> Tensor:
    """Return the symmetrised first-order proximity matrix ``A_F``."""

    return 0.5 * (adjacency + adjacency.t())


def second_order_in_proximity(adjacency: Tensor) -> Tensor:
    """Return the second-order in-degree proximity ``A_Sin``."""

    out_degree = adjacency.sum(dim=1)
    weighting = torch.diag(_degree_inv(out_degree))
    return adjacency.t() @ weighting @ adjacency


def second_order_out_proximity(adjacency: Tensor) -> Tensor:
    """Return the second-order out-degree proximity ``A_Sout``."""

    in_degree = adjacency.sum(dim=0)
    weighting = torch.diag(_degree_inv(in_degree))
    return adjacency @ weighting @ adjacency.t()


def _normalise_with_self_loops(matrix: Tensor) -> Tensor:
    """Add self-loops and perform symmetric normalisation."""

    identity = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    matrix = matrix + identity
    degree = matrix.sum(dim=1)
    inv_sqrt = _degree_inv_sqrt(degree)
    inv_sqrt = inv_sqrt.unsqueeze(0) * inv_sqrt.unsqueeze(1)
    return inv_sqrt * matrix


class DGCNLayer(nn.Module):
    """One layer of the Directed Graph Convolutional Network."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        concat: bool = True,
        dropout: float = 0.3,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.concat = concat
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        x: Tensor,
        proximities: Sequence[Tensor],
    ) -> Tensor:
        if len(proximities) != 3:
            raise ValueError("DGCNLayer expects exactly three proximity matrices")

        normalised = [
            _normalise_with_self_loops(matrix.to(x.device, dtype=x.dtype))
            for matrix in proximities
        ]

        support = x @ self.weight
        z_f, z_sin, z_sout = (matrix @ support for matrix in normalised)

        if self.concat:
            output = torch.cat(
                (z_f, self.alpha * z_sin, self.beta * z_sout), dim=-1
            )
        else:
            output = z_f + self.alpha * z_sin + self.beta * z_sout

        return self.dropout(output)


class PrerequisiteDGCN(nn.Module):
    """Stacked DGCN layers operating on prerequisite graphs."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int = 2,
        *,
        activation: Optional[nn.Module] = nn.ReLU(),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_size = num_features
        for _ in range(num_layers):
            layer = DGCNLayer(
                in_features=in_size,
                out_features=hidden_size,
                concat=True,
                dropout=dropout,
            )
            self.layers.append(layer)
            in_size = hidden_size * 3
        self.final_linear = nn.Linear(in_size, hidden_size)
        self.activation = activation

    def forward(self, x: Tensor, proximities: Sequence[Tensor]) -> Tensor:
        for layer in self.layers:
            x = layer(x, proximities)
            if self.activation is not None:
                x = self.activation(x)
        return self.final_linear(x)


def build_prerequisite_features(
    graph: PrerequisiteGraph,
    *,
    initial_features: Optional[Tensor] = None,
) -> Tensor:
    """Return node features for the prerequisite graph.

    If ``initial_features`` is ``None``, identity features are used."""

    if initial_features is None:
        return torch.eye(graph.size, dtype=torch.float32, device=graph.device)

    if initial_features.size(0) != graph.size:
        raise ValueError(
            "initial_features must have shape (num_nodes, num_features). "
            f"Expected {graph.size} nodes, got {initial_features.size(0)}."
        )
    return initial_features.to(graph.device)


def build_proximities(graph: PrerequisiteGraph) -> Tuple[Tensor, Tensor, Tensor]:
    adjacency = graph.adjacency
    return (
        first_order_proximity(adjacency),
        second_order_in_proximity(adjacency),
        second_order_out_proximity(adjacency),
    )


__all__ = [
    "PrerequisiteGraph",
    "PrerequisiteDGCN",
    "build_prerequisite_features",
    "build_proximities",
    "first_order_proximity",
    "second_order_in_proximity",
    "second_order_out_proximity",
    "load_prerequisite_graph",
]