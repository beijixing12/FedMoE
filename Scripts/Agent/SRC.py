from pathlib import Path
from typing import Optional, Union
import torch
from torch import nn
from Scripts.Agent.mamba_sequence import MambaSequenceModel, MambaState
from KTScripts.BackModels import MLP, Transformer
from Scripts.graph_encoder import (
    GraphFusionEncoder,
    load_prerequisite_graph,
    load_similarity_graph,
    load_triplet_graph,
)
import warnings
def _resolve_graph_path(
    provided: Optional[Union[Path, str]],
    base_dir: Path,
    filename: str,
    data_dir: Optional[Union[Path, str]] = None,
    dataset: Optional[str] = None,
) -> Path:
    """Resolve a graph resource path with sensible fallbacks."""

    if provided is not None:
        path = Path(provided)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if path.is_file():
            return path
        if path.exists():
            raise FileNotFoundError(
                f"Expected a file for {filename!r}, but got directory: {path!s}"
            )

    def _normalise_root(root: Union[Path, str]) -> Path:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = (base_dir / root_path).resolve()
        else:
            root_path = root_path.resolve()
        return root_path

    def _add_root(collection, root):
        root_path = _normalise_root(root)
        if root_path not in collection:
            collection.append(root_path)

    candidate_roots: list[Path] = []

    if data_dir is not None:
        _add_root(candidate_roots, data_dir)

    _add_root(candidate_roots, base_dir)
    _add_root(candidate_roots, base_dir / "data")
    _add_root(candidate_roots, base_dir.parent)
    _add_root(candidate_roots, base_dir.parent / "data")



    if base_dir.name != "1LPRSRC":
        vendored_root = base_dir / "1LPRSRC"
        _add_root(candidate_roots, vendored_root)
        _add_root(candidate_roots, vendored_root / "data")

    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add_candidate(path: Path) -> None:
        if path not in seen:
            seen.add(path)
            candidates.append(path)
    dataset_hints: list[str] = []
    if dataset:
        dataset_hints.append(dataset)
        dataset_lower = dataset.lower()
        if dataset_lower not in dataset_hints:
            dataset_hints.append(dataset_lower)
        if dataset_lower.startswith("assist") and dataset_lower[len("assist") :].isdigit():
            # Many datasets are stored with four-digit year suffixes.
            year_suffix = dataset_lower[len("assist") :]
            long_form = f"assist20{year_suffix}"
            if long_form not in dataset_hints:
                dataset_hints.append(long_form)
        if dataset_lower.startswith("assist20") and dataset_lower[len("assist20") :].isdigit():
            # Support both "assist09" and "assist2009" style folder names.
            short_suffix = dataset_lower[len("assist20") :]
            short_form = f"assist{short_suffix}"
            if short_form not in dataset_hints:
                dataset_hints.append(short_form)
    dataset_hint_tokens = {hint.lower() for hint in dataset_hints}
    for root in candidate_roots:
        _add_candidate(root / filename)
        _add_candidate(root / "graphs" / filename)
        for hint in dataset_hints:
            _add_candidate(root / hint / filename)
            _add_candidate(root / hint / "graphs" / filename)
            
        try:
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                child_name = child.name
                child_token = child_name.lower()
                if dataset_hint_tokens:
                    matched = any(
                        token == child_token or token in child_token
                        for token in dataset_hint_tokens
                    )
                    if not matched:
                        continue
                _add_candidate(child / filename)
                _add_candidate(child / "graphs" / filename)
        except OSError:
            # The root might not exist or be inaccessible; skip gracefully.
            pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f" - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Unable to locate {filename!r}. Checked the following locations relative to"
        f" {base_dir}:\n{searched if searched else ' (no candidates generated)'}"
    )

class SRC(nn.Module):
    def __init__(
        self,
        skill_num,
        input_size,
        weight_size,
        hidden_size,
        dropout,
        allow_repeat=False,
        with_kt=False,
        *,
        prerequisite_graph_path: Optional[Union[Path, str]] = None,
        similarity_graph_path: Optional[Union[Path, str]] = None,
        dataset_name: Optional[str] = None,
        data_dir: Optional[Union[Path, str]] = None,
        hetero_graph_path: Optional[Union[Path, str]] = None,
        dgcn_layers=2,
        lightgcn_layers=2,
        fusion_weight=0.5,
        mamba_kwargs: Optional[dict] = None,
        num_experts: int = 1,
        zpd_tau: float = 0.5,
        num_exercises: Optional[int] = None,
    ):
        super().__init__()
        base_dir = Path(__file__).resolve().parents[2]
        prerequisite_graph_file = _resolve_graph_path(
            prerequisite_graph_path,
            base_dir,
            "prerequisites_graph.json",
            data_dir=data_dir,
            dataset=dataset_name,
        )

        similarity_graph_file = _resolve_graph_path(
            similarity_graph_path,
            base_dir,
            "similarity_graph.json",
            data_dir=data_dir,
            dataset=dataset_name,
        )
        hetero_graph_file = None
        if hetero_graph_path is not None:
            candidate = Path(hetero_graph_path)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            if candidate.is_file():
                hetero_graph_file = candidate
            else:
                warnings.warn(
                    f"Provided hetero graph path {hetero_graph_path!r} is not a file",
                    RuntimeWarning,
                )
        elif dataset_name:
            dataset_hint = dataset_name if dataset_name.endswith(".npz") else f"{dataset_name}.npz"
            try:
                hetero_graph_file = _resolve_graph_path(
                    None,
                    base_dir,
                    dataset_hint,
                    data_dir=data_dir,
                    dataset=dataset_name,
                )
            except FileNotFoundError:
                hetero_graph_file = None
                warnings.warn(
                    "Unable to locate dataset archive for heterogeneous graph construction; "
                    "falling back to similarity graph only.",
                    RuntimeWarning,
                )

        # prerequisite_graph_file = '/home/zengxiangyu/SRC-py/data/assist09/prerequisites_graph.json'
        # similarity_graph_file = '/home/zengxiangyu/SRC-py/data/assist09/similarity_graph.json'
        # hetero_graph_file = '/home/zengxiangyu/SRC-py/data/assist09/hetero_graph.json'

        # prerequisite_graph_file = '/home/zengxiangyu/SRC-py/data/assist12/prerequisites_graph.json'
        # similarity_graph_file = '/home/zengxiangyu/SRC-py/data/assist12/similarity_graph.json'
        # hetero_graph_file = '/home/zengxiangyu/SRC-py/data/assist12/assist12_hetero_graph.json'

        prerequisite_graph_file = '/home/zengxiangyu/SRC-py/data/OLI/prerequisites_graph.json'
        similarity_graph_file = '/home/zengxiangyu/SRC-py/data/OLI/similarity_graph.json'
        hetero_graph_file = '/home/zengxiangyu/SRC-py/data/OLI/OLI_hetero_graph.json'


        # prerequisite_graph_file = '/home/zengxiangyu/SRC-py/data/assist17/prerequisites_graph_dense.json'
        # similarity_graph_file = '/home/zengxiangyu/SRC-py/data/assist17/similarity_graph_dense.json'

        prerequisite_graph = load_prerequisite_graph(prerequisite_graph_file)
        similarity_graph = load_similarity_graph(similarity_graph_file)

        hetero_graph = None
        if hetero_graph_file is not None:
            try:
                hetero_graph = load_triplet_graph(hetero_graph_file)
            except FileNotFoundError:
                warnings.warn(
                    f"Heterogeneous graph not found at {hetero_graph_file}; "
                    "falling back to similarity-only propagation.",
                    RuntimeWarning,
                )
            except ValueError as exc:
                warnings.warn(
                    f"Failed to load heterogeneous graph ({exc}); using similarity graph only.",
                    RuntimeWarning,
                )

            except KeyError as exc:
                warnings.warn(
                    f"Heterogeneous graph archive {hetero_graph_file} missing required column ({exc}); "
                    "falling back to similarity graph only.",
                    RuntimeWarning,
                )

        self.graph_encoder = GraphFusionEncoder(
            skill_num=skill_num,
            embedding_dim=input_size,
            prerequisite_graph=prerequisite_graph,
            similarity_graph=similarity_graph,
            hetero_graph=hetero_graph,
            dgcn_layers=dgcn_layers,
            lightgcn_layers=lightgcn_layers,
            fusion_weight=fusion_weight,
            dgcn_dropout=dropout,
        )
        self.l1 = nn.Linear(input_size + 1, input_size)
        self.l2 = nn.Linear(input_size, hidden_size)
        if mamba_kwargs is None:
            mamba_kwargs = {}
        default_mamba = {
            "d_state": hidden_size,
            "d_conv": 4,
            "expand": 2,
            "dropout": dropout,
        }
        default_mamba.update(mamba_kwargs)

        self.state_encoder = MambaSequenceModel(
            input_size,
            hidden_size,
            **dict(default_mamba),
        )
        self.path_encoder = Transformer(hidden_size, hidden_size, 0.0, head=1, b=1, transformer_mask=False)
        self.W1_list = nn.ModuleList(
            [nn.Linear(hidden_size, weight_size, bias=False) for _ in range(num_experts)]
        )  # blending encoder
        self.W2_list = nn.ModuleList(
            [nn.Linear(hidden_size, weight_size, bias=False) for _ in range(num_experts)]
        )  # blending decoder
        self.vt_list = nn.ModuleList(
            [nn.Linear(weight_size, 1, bias=False) for _ in range(num_experts)]
        )  # scaling sum of enc and dec by v.T
        self.decoders = nn.ModuleList(
            [
                MambaSequenceModel(
                    hidden_size,
                    hidden_size,
                    **dict(default_mamba),
                )
                for _ in range(num_experts)
            ]
        )
        self.state_proj_list = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)]
        )
        if with_kt:
            self.ktRnn = MambaSequenceModel(
                hidden_size,
                hidden_size,
                **dict(default_mamba),
            )
            self.ktMlp = MLP(hidden_size, [hidden_size // 2, hidden_size // 4, 1], dropout=dropout)
            self.kt_exercise_proj = nn.Linear(hidden_size * 2, 1)
            self.exercise_embedding = None
            if num_exercises is not None:
                self.exercise_embedding = nn.Embedding(num_exercises, hidden_size)
        self.allow_repeat = allow_repeat
        self.withKt = with_kt
        self.skill_num = skill_num
        self.num_experts = num_experts
        self.num_exercises = num_exercises
        self.prototypes = nn.Parameter(torch.randn(num_experts, hidden_size))
        self.router_head = nn.Linear(hidden_size * 2, 1)
        self.zpd_tau = zpd_tau
        self._graph_embeddings = None
        self._latest_state_output = None
        self._latest_states = None

    def forward(self, targets, initial_logs, initial_log_scores, origin_path, n, expert_id=0):
        """Alias for :meth:`construct` so the module can be invoked directly."""
        return self.construct(targets, initial_logs, initial_log_scores, origin_path, n, expert_id=expert_id)

    def begin_episode(self, targets, initial_logs, initial_log_scores, expert_id=0):
        # targets: (B, K), where K is the num of targets in this batch
        targets = self.l2(self._embed_indices(targets).mean(dim=1, keepdim=True))  # (B, 1, H)
        batch_size = targets.size(0)
        if initial_logs is not None:
            self.step(initial_logs, initial_log_scores, None)
        if expert_id is None:
            raise ValueError("expert_id must be provided; call route_expert() after micro-loop.")
        resolved_expert_id = self._resolve_expert_id(expert_id, batch_size)
        decoder, _, _, _, state_proj = self._get_expert_modules(resolved_expert_id)
        decoder_state = decoder.init_state(batch_size, targets.device, targets.dtype)
        expert_state = None
        if self._latest_state_output is not None:
            expert_state = state_proj(self._latest_state_output[:, -1:, :])
        return targets, decoder_state, expert_state, resolved_expert_id

    def step(self, x, score, states):
        x_embed = self._embed_indices(x)
        if score is None:
            score = torch.zeros_like(x_embed[..., 0], dtype=x_embed.dtype, device=x_embed.device)
        score = score.to(x_embed.dtype)
        target_shape = x_embed.shape[:-1] + (1,)
        score_dims = score.dim()
        target_dims = len(target_shape)
        if score_dims < target_dims:
            for _ in range(target_dims - score_dims):
                score = score.unsqueeze(-1)
        elif score_dims > target_dims:
            score = score.reshape(target_shape)
        elif score.shape != target_shape:
            score = score.reshape(target_shape)
        x = self.l1(torch.cat((x_embed, score), -1))
        if not isinstance(states, MambaState):
            states = self.state_encoder.init_state(x.shape[0], x.device, x.dtype)
        state_output, states = self.state_encoder(x, states)
        self._latest_state_output = state_output
        self._latest_states = states
        return states

    def step_with_output(self, x, score, states):
        x_embed = self._embed_indices(x)
        if score is None:
            score = torch.zeros_like(x_embed[..., 0], dtype=x_embed.dtype, device=x_embed.device)
        score = score.to(x_embed.dtype)
        target_shape = x_embed.shape[:-1] + (1,)
        score_dims = score.dim()
        target_dims = len(target_shape)
        if score_dims < target_dims:
            for _ in range(target_dims - score_dims):
                score = score.unsqueeze(-1)
        elif score_dims > target_dims:
            score = score.reshape(target_shape)
        elif score.shape != target_shape:
            score = score.reshape(target_shape)
        x = self.l1(torch.cat((x_embed, score), -1))
        if not isinstance(states, MambaState):
            states = self.state_encoder.init_state(x.shape[0], x.device, x.dtype)
        state_output, states = self.state_encoder(x, states)
        self._latest_state_output = state_output
        self._latest_states = states
        return state_output, states

    def construct(self, targets, initial_logs, initial_log_scores, origin_path, n, expert_id=0):
        self._refresh_graph_embeddings()
        try:
            targets, states, expert_state, resolved_expert_id = self.begin_episode(
                targets, initial_logs, initial_log_scores, expert_id=expert_id
            )
            decoder, W1, W2, vt, _ = self._get_expert_modules(resolved_expert_id)
            expert_context = targets
            if expert_state is not None:
                expert_context = expert_context + expert_state
            inputs = self.l2(self._embed_indices(origin_path))
            encoder_states = inputs
            encoder_states = self.path_encoder(encoder_states)
            encoder_states = encoder_states + inputs
            blend1 = W1(encoder_states + encoder_states.mean(dim=1, keepdim=True) + expert_context)  # (B, L, W)
            decoder_input = torch.zeros_like(inputs[:, 0:1])  # (B, 1, I)
            probs, paths = [], []
            selecting_s = []
            a1 = torch.arange(inputs.shape[0], device=inputs.device)
            selected = torch.zeros_like(inputs[:, :, 0], dtype=torch.bool)
            minimum_fill = torch.full_like(inputs[:, :, 0], -1e9, dtype=inputs.dtype)
            hidden_states = []
            for i in range(n):
                hidden, states = decoder(decoder_input, states)
                if self.withKt and i > 0:
                    hidden_states.append(hidden)
                # Compute blended representation at each decoder time step
                blend2 = W2(hidden)  # (B, 1, W)
                blend_sum = blend1 + blend2  # (B, L, W)
                out = vt(blend_sum).squeeze(-1)  # (B, L)
                if not self.allow_repeat:
                    out = torch.where(selected, minimum_fill, out)
                    out = torch.softmax(out, dim=-1)
                    if self.training:
                        selecting = torch.multinomial(out, 1).squeeze(-1)
                    else:
                        selecting = torch.argmax(out, dim=1)
                    selected[a1, selecting] = True
                else:
                    out = torch.softmax(out, dim=-1)
                    selecting = torch.multinomial(out, 1).squeeze(-1)
                selecting_s.append(selecting)
                path = origin_path[a1, selecting]
                decoder_input = encoder_states[a1, selecting].unsqueeze(1)
                out = out[a1, selecting]
                paths.append(path)
                probs.append(out)
            probs = torch.stack(probs, 1)
            paths = torch.stack(paths, 1)  # (B, n)
            selecting_s = torch.stack(selecting_s, 1)
            if self.withKt and self.training:
                hidden_states.append(hidden)
                hidden_states = torch.cat(hidden_states, dim=1)
                batch, steps, width = hidden_states.shape
                kt_logits = self.ktMlp(hidden_states.view(batch * steps, width))
                kt_output = torch.sigmoid(kt_logits).view(batch, steps, -1)
                result = [paths, probs, selecting_s, kt_output]
                return result
            return paths, probs, selecting_s
        finally:
            self._graph_embeddings = None

    def backup(self, targets, initial_logs, initial_log_scores, origin_path, selecting_s, expert_id=0):
        self._refresh_graph_embeddings()
        try:
            targets, states, expert_state, resolved_expert_id = self.begin_episode(
                targets, initial_logs, initial_log_scores, expert_id=expert_id
            )
            decoder, W1, W2, vt, _ = self._get_expert_modules(resolved_expert_id)
            expert_context = targets
            if expert_state is not None:
                expert_context = expert_context + expert_state
            inputs = self.l2(self._embed_indices(origin_path))
            encoder_states = inputs
            encoder_states = self.path_encoder(encoder_states)
            encoder_states = encoder_states + inputs
            blend1 = W1(encoder_states + encoder_states.mean(dim=1, keepdim=True) + expert_context)  # (B, L, W)
            batch_indices = torch.arange(encoder_states.shape[0], device=encoder_states.device).unsqueeze(1)
            selecting_states = encoder_states[batch_indices, selecting_s]
            selecting_states = torch.cat((torch.zeros_like(selecting_states[:, 0:1]), selecting_states[:, :-1]), 1)
            hidden_states, _ = decoder(selecting_states, states)
            blend2 = W2(hidden_states)  # (B, n, W)
            blend_sum = blend1.unsqueeze(1) + blend2.unsqueeze(2)  # (B, n, L, W)
            out = vt(blend_sum).squeeze(-1)  # (B, n, L)
            # Masking probabilities according to output order
            mask = selecting_s.unsqueeze(1).repeat(1, selecting_s.shape[-1], 1)  # (B, n, n)
            mask = torch.tril(mask + 1, -1).view(-1, mask.shape[-1])
            out = out.view(-1, out.shape[-1])
            out = torch.cat((torch.zeros_like(out[:, 0:1]), out), -1)
            row_indices = torch.arange(out.shape[0], device=out.device).unsqueeze(1)
            out[row_indices, mask] = -1e9
            out = out[:, 1:].view(origin_path.shape[0], -1, origin_path.shape[1])

            out = torch.softmax(out, dim=-1)
            probs = torch.gather(out, 2, selecting_s.unsqueeze(-1)).squeeze(-1)
            return probs
        finally:
            self._graph_embeddings = None

    def _refresh_graph_embeddings(self):
        self._graph_embeddings = self.graph_encoder()
        return self._graph_embeddings

    def _get_graph_embeddings(self):
        if self._graph_embeddings is None:
            return self._refresh_graph_embeddings()
        return self._graph_embeddings

    def _get_expert_modules(self, expert_id):
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(f"expert_id must be in [0, {self.num_experts - 1}], got {expert_id}")
        return (
            self.decoders[expert_id],
            self.W1_list[expert_id],
            self.W2_list[expert_id],
            self.vt_list[expert_id],
            self.state_proj_list[expert_id],
        )

    def _resolve_expert_id(self, expert_id, batch_size):
        if expert_id is not None:
            self._get_expert_modules(expert_id)
            return expert_id
        if self._latest_state_output is None:
            return 0
        if batch_size != 1:
            raise ValueError("expert_id=None requires batch_size=1 for routing.")
        h_t = self._latest_state_output[:, -1, :]
        prototypes = self.prototypes.unsqueeze(0)
        h_t = h_t.unsqueeze(1).expand(-1, prototypes.shape[1], -1)
        router_input = torch.cat((h_t, prototypes), dim=-1)
        logits = self.router_head(router_input).squeeze(-1)
        probs = torch.sigmoid(logits)
        distances = torch.abs(probs - self.zpd_tau)
        return int(torch.argmin(distances, dim=1).item())

    def route_expert(self, batch_size=1):
        return self._resolve_expert_id(None, batch_size)

    def route_expert_with_state(self, state_output):
        if state_output.dim() == 3:
            state_output = state_output[:, -1, :]
        if state_output.dim() != 2:
            raise ValueError("state_output must be (B, H) or (B, T, H).")
        prototypes = self.prototypes.unsqueeze(0)
        h_t = state_output.unsqueeze(1).expand(-1, prototypes.shape[1], -1)
        router_input = torch.cat((h_t, prototypes), dim=-1)
        logits = self.router_head(router_input).squeeze(-1)
        probs = torch.sigmoid(logits)
        distances = torch.abs(probs - self.zpd_tau)
        return torch.argmin(distances, dim=1)

    def predict_exercise_correctness(self, exercise_ids):
        if not self.withKt:
            raise RuntimeError("KT prediction requested but with_kt=False")
        if self._latest_state_output is None:
            raise RuntimeError("KT prediction requested before any state update")
        h_t = self._latest_state_output[:, -1, :]
        exercise_embeddings = self._get_exercise_embeddings(exercise_ids)
        if exercise_embeddings.dim() == 2:
            exercise_embeddings = exercise_embeddings.unsqueeze(1)
        h_t = h_t.unsqueeze(1).expand(-1, exercise_embeddings.shape[1], -1)
        kt_input = torch.cat((h_t, exercise_embeddings), dim=-1)
        logits = self.kt_exercise_proj(kt_input).squeeze(-1)
        return torch.sigmoid(logits)

    def predict_exercise_correctness_with_state(self, state_output, exercise_ids):
        if not self.withKt:
            raise RuntimeError("KT prediction requested but with_kt=False")
        if state_output.dim() == 3:
            state_output = state_output[:, -1, :]
        exercise_embeddings = self._get_exercise_embeddings(exercise_ids)
        if exercise_embeddings.dim() == 2:
            exercise_embeddings = exercise_embeddings.unsqueeze(1)
        h_t = state_output.unsqueeze(1).expand(-1, exercise_embeddings.shape[1], -1)
        kt_input = torch.cat((h_t, exercise_embeddings), dim=-1)
        logits = self.kt_exercise_proj(kt_input).squeeze(-1)
        return torch.sigmoid(logits)

    def _get_exercise_embeddings(self, exercise_ids):
        if self.exercise_embedding is not None:
            return self.exercise_embedding(exercise_ids)
        return self._embed_indices(exercise_ids)

    def freeze_expert_params(self):
        for module in (
            self.decoders,
            self.W1_list,
            self.W2_list,
            self.vt_list,
            self.state_proj_list,
        ):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_expert_params(self):
        for module in (
            self.decoders,
            self.W1_list,
            self.W2_list,
            self.vt_list,
            self.state_proj_list,
        ):
            for param in module.parameters():
                param.requires_grad = True

    def get_shared_state_dict(self):
        expert_prefixes = (
            "decoders.",
            "W1_list.",
            "W2_list.",
            "vt_list.",
            "state_proj_list.",
        )
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith(expert_prefixes)
        }

    def load_shared_state_dict(self, shared_state):
        current = self.state_dict()
        current.update(shared_state)
        self.load_state_dict(current)

    def _embed_indices(self, indices):
        embeddings = self._get_graph_embeddings()
        return embeddings[indices]
