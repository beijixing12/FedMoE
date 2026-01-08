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
import os

import numpy as np
from typing import Iterable, Optional, Sequence, Tuple


def _resolve_npz_key(
    available: Iterable[str],
    primary: str,
    alternatives: Sequence[str],
) -> Optional[str]:
    """Return the first matching key present in ``available``.

    Historic preprocessing pipelines occasionally wrote different column names
    into the compressed ``.npz`` archives (for example ``skill_seq`` instead of
    ``skill``).  The current exporters use the canonical keys, but keeping a
    compatibility shim here avoids runtime crashes when users plug in older
    artefacts.  The helper prefers ``primary`` and otherwise scans the supplied
    ``alternatives``.
    """

    available_set = list(available)
    if primary in available_set:
        return primary
    for candidate in alternatives:
        if candidate in available_set:
            return candidate
    return None


class KTDataset:
    def __init__(self, data_folder, max_len=200):
        dataset_path = self._resolve_npz_path(data_folder)
        archive_name = os.path.basename(dataset_path)
        self.dataset_name = os.path.splitext(archive_name)[0]
        with np.load(dataset_path, allow_pickle=True) as data:
            files = set(data.files)
            skill_key = self._resolve_key(files, ['skill', 'skills', 'concept_ids', 'concept_id'])
            label_key = self._resolve_key(files, ['y', 'label', 'labels'])
            length_key = self._resolve_key(files, ['real_len', 'real_length', 'seq_len', 'length'])
            if skill_key is None or label_key is None or length_key is None:
                missing = []
                if skill_key is None:
                    missing.append("skill")
                if label_key is None:
                    missing.append("y")
                if length_key is None:
                    missing.append("real_len")
                raise KeyError(
                    f"Missing required field(s) {', '.join(missing)} in {archive_name}. "
                    f"Available fields: {sorted(files)}"
                )
            self.data = [data[skill_key], data[label_key], data[length_key]]
            if self.dataset_name == 'junyi':
                self.data[0] = data['problem'] - 1
        self.data[1] = [_.astype(np.float32) for _ in self.data[1]]
        try:
            self.feats_num = np.max(self.data[0]).item() + 1
        except ValueError:
            self.feats_num = np.max(np.concatenate(self.data[0])).item() + 1
        self.data = list(zip(*self.data))
        self.users_num = len(self.data)
        self.max_len = max_len
        self.mask = np.zeros(self.max_len, dtype=bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        skill, y, real_len = self.data[item]
        skill, y = skill[:self.max_len], y[:self.max_len]
        if len(skill) < self.max_len:
            skill, y = skill.copy(), y.copy()
            skill.resize((self.max_len,))
            y.resize((self.max_len,))
        mask = self.mask.copy()
        mask[:real_len] = True
        return skill, y, mask

    @staticmethod
    def _resolve_key(files, candidates):
        lower_to_original = {key.lower(): key for key in files}
        for candidate in candidates:
            if candidate in files:
                return candidate
            lower_candidate = candidate.lower()
            if lower_candidate in lower_to_original:
                return lower_to_original[lower_candidate]
        return None


    @staticmethod
    def _resolve_npz_path(data_folder):
        if os.path.isfile(data_folder):
            if data_folder.lower().endswith('.npz'):
                return data_folder
            raise FileNotFoundError(f"Expected an .npz file, got {data_folder}")

        if not os.path.isdir(data_folder):
            raise FileNotFoundError(f"Dataset path {data_folder} does not exist")

        folder_name = os.path.basename(os.path.normpath(data_folder))
        preferred = os.path.join(data_folder, folder_name + '.npz')
        if os.path.isfile(preferred):
            return preferred

        npz_files = sorted(
            f for f in os.listdir(data_folder)
            if f.lower().endswith('.npz') and os.path.isfile(os.path.join(data_folder, f))
        )
        if not npz_files:
            raise FileNotFoundError(
                f"No .npz files found under {data_folder}. "
                "Please provide a directory containing a dataset archive or the archive file itself."
            )
        if len(npz_files) > 1:
            # Prefer an archive that matches the folder name ignoring extensions.
            stem_to_file = {os.path.splitext(f)[0].lower(): f for f in npz_files}
            folder_stem = folder_name.lower()
            if folder_stem in stem_to_file:
                return os.path.join(data_folder, stem_to_file[folder_stem])
        return os.path.join(data_folder, npz_files[0])



class RecDataset(KTDataset):
    # For GRU4Rec, Predict the next item
    def __getitem__(self, item):
        skill, _, real_len = self.data[item]
        skill = skill[:self.max_len + 1]
        if len(skill) < self.max_len + 1:
            skill = skill.copy()
            skill.resize((self.max_len + 1,))
        skill = skill.astype(np.int64, copy=False)
        mask = self.mask.copy()
        mask[:real_len - 1] = True
        return skill[:-1], skill[1:], mask


class RetrievalDataset(KTDataset):
    def __init__(self, data_folder, r=5, train_test_split=0.8, max_len=200):
        super(RetrievalDataset, self).__init__(data_folder, max_len)
        self.es = Elasticsearch(hosts=['http://localhost:9200/']).options(
            request_timeout=20,
            retry_on_timeout=True,
            ignore_status=[400, 404]
        )
        self.safe_users = np.arange(int(len(self.data) * train_test_split))
        self.R = r
        self.index = f'{self.dataset_name}_train'
        self.safe_query = self.get_safe_query()

    def get_safe_query(self):
        safe_query = [[self.data[i][0][0], self.data[i][1][0]] for i in self.safe_users]
        safe_query = np.asarray(safe_query, dtype=np.int32)
        return safe_query

    def get_query(self, user, skills, index_range):
        safe_user = np.random.choice(self.safe_users, self.R + 1, replace=False)
        safe_user = safe_user[safe_user != user][:self.R]
        safe_query = self.safe_query[safe_user]
        query_s = []
        skills_str = ' '
        for _ in index_range:
            skills_str += f' {skills[_]}'
            query = [{'index': self.index},
                     {'size': self.R,
                      'query': {'bool': {'must': [{'term': {'skill': skills[_]}}, {'match': {'history': skills_str}}],
                                         'must_not': {'term': {'user': user}}}}}]
            query_s += query
        result = self.es.msearch(searches=query_s)['responses']
        r_his, r_skill_y, r_len = [], [], []
        for rs in result:  # seq_len
            skill_y, real_len = [], []
            rs = rs['hits']['hits']
            for r in rs:  # R
                r = r['_source']
                his = np.fromstring(r['history'], dtype=np.int32, sep=' ')
                his = np.stack((his, np.asarray(r['y'], dtype=np.int32)), axis=-1)
                if his.ndim == 1:
                    his = np.expand_dims(his, 0)
                his.resize((self.max_len, 2))
                r_his.append(his)
                skill_y.append(his[-1])
                real_len.append(len(skill_y))
            # If the quantity is less than R, fill it up
            for _ in range(self.R - len(rs)):
                his = safe_query[_:_ + 1].copy()
                his.resize((self.max_len, 2))
                r_his.append(his)
                skill_y.append(his[_])
                real_len.append(len(skill_y))
            r_skill_y.append(skill_y)  # (R, 2)
            r_len.append(real_len)
        r_his, r_skill_y, r_len = np.asarray(r_his, dtype=np.int32), np.asarray(r_skill_y, dtype=np.int32), np.asarray(
            r_len, dtype=np.int32)
        r_his.resize((self.max_len * self.R, self.max_len, 2))
        r_skill_y.resize((self.max_len, self.R, 2))
        r_len.resize((self.max_len, self.R))
        r_len[r_len < 1] = 1
        return r_his, r_skill_y, r_len

    def __getitem__(self, item):
        skill, y, real_len = self.data[item]
        skill, y = skill[:self.max_len], y[:self.max_len]
        r_his, r_skill_y, r_len = self.get_query(item, skill, range(real_len))
        if len(skill) < self.max_len:
            skill, y = skill.copy(), y.copy()
            skill.resize((self.max_len,))
            y.resize((self.max_len,))
        mask = self.mask.copy()
        mask[:real_len] = True
        return skill, r_his, r_skill_y, y, mask, r_len


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    folder = '../data/assist15'
    dataset = RetrievalDataset(folder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for batch in tqdm(dataloader):
        pass
