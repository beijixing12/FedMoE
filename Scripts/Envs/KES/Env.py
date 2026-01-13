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
import json
import torch

from pathlib import Path

from .utils import load_d_agent, episode_reward


class KESEnv():
    def __init__(
        self,
        dataset,
        model_name='DKT',
        dataset_name='assist09',
        concept_exercise_map=None,
        mastery_threshold=0.8,
        skill_num_override=None,
    ):
        self.skill_num = dataset.feats_num if skill_num_override is None else skill_num_override
        self.model = load_d_agent(model_name, dataset_name, self.skill_num)
        self.targets = None
        self.states = (None, None)
        self.initial_score = None
        self.exercise_feedback = self._build_exercise_feedback(dataset)
        (
            self.concept_exercise_map,
            self.exercise_error_rate,
            self.difficulty_bins,
            self.difficulty_counts,
        ) = self._load_concept_exercise_map(concept_exercise_map)
        self.mastery_threshold = mastery_threshold

    def exam(self, targets, states):
        with torch.no_grad():
            scores = []
            for i in range(targets.shape[1]):
                score, _ = self.model.learn_lstm(targets[:, i:i + 1], *states)  # (B, 1)
                scores.append(score)
            return torch.mean(torch.cat(scores, dim=1), dim=1)

    def begin_episode(self, targets, initial_logs):
        self.model = self.model.to(targets.device)
        self.targets = targets
        initial_score, initial_log_scores, states = self.begin_episode_(targets, initial_logs)
        self.initial_score = initial_score
        self.states = states
        return initial_log_scores

    def begin_episode_(self, targets, initial_logs=None):
        with torch.no_grad():
            states = (None, None)
            score = None
            if initial_logs is not None:
                score, states = self.model.learn_lstm(initial_logs)
            initial_score = self.exam(targets, states)
            return initial_score, score, states

    def n_step(self, learning_path, binary=False):
        with torch.no_grad():
            scores, states = self.model.learn_lstm(learning_path, *self.states)
        self.states = states
        if binary:
            scores = (scores > 0.5).float()
        return scores

    def end_episode(self, **kwargs):
        final_score, reward = self.end_episode_(self.initial_score, self.targets, *self.states)
        if 'score' in kwargs:
            return final_score, reward
        return reward

    def end_episode_(self, initial_score, targets, states1, states2):
        final_score = self.exam(targets, (states1, states2))
        reward = episode_reward(initial_score, final_score, 1).unsqueeze(-1)
        return final_score, reward

    def run_micro_loop(self, model, concept_ids, device=None):
        if self.concept_exercise_map is None:
            raise RuntimeError("Concept-exercise mapping required for micro-loop")
        if device is None:
            device = next(model.parameters()).device
        if model._latest_state_output is None or model._latest_states is None:
            raise RuntimeError("Model state required before running micro-loop")
        batch_size = concept_ids.shape[0]
        selections = torch.zeros_like(concept_ids)
        current_scores = torch.zeros_like(concept_ids, dtype=torch.float)
        routed_experts = torch.zeros_like(concept_ids)
        cached_inputs = model._latest_states.cached_inputs
        for idx in range(concept_ids.shape[1]):
            concept = concept_ids[:, idx]
            for b in range(batch_size):
                exercises = self._get_exercises_for_concept(concept[b].item())
                max_steps = max(len(exercises), 1) * 5
                state = None
                if cached_inputs is not None:
                    state = type(model._latest_states)(cached_inputs[b : b + 1].clone())
                state_output = model._latest_state_output[b : b + 1, -1, :].clone()
                mastery_scores = []
                for _ in range(max_steps):
                    exercise_ids = torch.tensor(exercises, device=device, dtype=torch.long)
                    probs = model.predict_exercise_correctness_with_state(
                        state_output, exercise_ids.unsqueeze(0)
                    ).squeeze(0)
                    choice = torch.argmin(torch.abs(probs - 0.5)).item()
                    chosen_exercise = exercise_ids[choice]
                    selections[b, idx] = chosen_exercise
                    feedback = self._sample_exercise_feedback(chosen_exercise.item(), device=device)
                    current_scores[b, idx] = feedback
                    mastery_scores.append(feedback.item())
                    state_output, state = model.step_with_output(
                        chosen_exercise.view(1, 1),
                        feedback.view(1, 1),
                        state,
                    )
                    if self._concept_mastered(torch.tensor(mastery_scores)):
                        break
                routed_experts[b, idx] = model.route_expert_with_state(state_output).item()
        return selections, current_scores, routed_experts

    def _concept_mastered(self, mastery_scores):
        return mastery_scores.mean().item() >= self.mastery_threshold

    def _get_exercises_for_concept(self, concept_id):
        exercises = self.concept_exercise_map.get(str(concept_id), [])
        if not exercises:
            raise KeyError(f"No exercises found for concept {concept_id}")
        return exercises

    @staticmethod
    def _build_exercise_feedback(dataset):
        totals = {}
        corrects = {}
        for skill, responses, mask in dataset.data:
            skill_seq = torch.as_tensor(skill).cpu().numpy()
            response_seq = torch.as_tensor(responses).cpu().numpy()
            if isinstance(mask, (bool, int, float)) or getattr(mask, "ndim", 0) == 0:
                valid = [bool(mask)] * len(skill_seq)
            else:
                valid = torch.as_tensor(mask).cpu().numpy().astype(bool)
            for exercise_id, answer, keep in zip(skill_seq, response_seq, valid):
                if not keep:
                    continue
                exercise = int(exercise_id)
                totals[exercise] = totals.get(exercise, 0) + 1
                corrects[exercise] = corrects.get(exercise, 0) + int(answer)
        feedback = {}
        for exercise, total in totals.items():
            feedback[exercise] = corrects.get(exercise, 0) / max(total, 1)
        return feedback

    def _sample_exercise_feedback(self, exercise_id, device=None):
        if device is None:
            device = torch.device("cpu")
        prob = self.exercise_feedback.get(exercise_id)
        if prob is None:
            prob = 0.5
        return torch.bernoulli(torch.tensor(prob, device=device, dtype=torch.float))

    @staticmethod
    def _load_concept_exercise_map(mapping_path):
        if mapping_path is None:
            return None, None, None, None
        path = Path(mapping_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "concept_exercise_map" in payload:
            return (
                payload.get("concept_exercise_map"),
                payload.get("exercise_error_rate"),
                payload.get("difficulty_bins"),
                payload.get("difficulty_counts"),
            )
        return payload, None, None, None

    def get_difficulty_payload(self):
        return {
            "exercise_error_rate": self.exercise_error_rate,
            "difficulty_bins": self.difficulty_bins,
            "difficulty_counts": self.difficulty_counts,
        }
