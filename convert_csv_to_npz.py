"""Utility to convert custom CSV-like sequences into SRC NPZ format."""
import argparse
import json
import os
from typing import Dict, List

import numpy as np


def _parse_sequence(line: str) -> List[int]:
    if not line:
        return []
    return [int(token) for token in line.split(',') if token]


def _build_skill_mapping(skill_sequences: List[List[int]]) -> Dict[int, int]:
    unique = sorted({token for seq in skill_sequences for token in seq if token != 0})
    mapping = {value: index + 1 for index, value in enumerate(unique)}
    mapping[0] = 0
    return mapping


def convert(input_path: str, output_dir: str, dataset_name: str, save_mapping: bool = True) -> None:
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError('输入文件为空。')
    header = lines[0]
    if ';' not in header:
        raise ValueError('首行缺少列定义分隔符 ";"，请确认数据格式。')
    blocks = lines[1:]
    expected_per_user = 3 + 7
    if len(blocks) % expected_per_user != 0:
        raise ValueError('数据行数量不能被每个样本的字段数整除，请检查是否存在缺失行。')

    skills, labels, real_lengths = [], [], []
    extra_fields = {
        'num_attempt_seq': [],
        'use_time_first_seq': [],
        'use_time_seq': [],
        'num_hint_seq': [],
        'mask_seq': [],
        'school_id': []
    }
    cursor = 0
    while cursor < len(blocks):
        user_id = int(blocks[cursor])
        school_id = int(blocks[cursor + 1])
        seq_len = int(blocks[cursor + 2])
        question_seq = _parse_sequence(blocks[cursor + 3])
        num_attempt_seq = _parse_sequence(blocks[cursor + 4])
        correctness_seq = _parse_sequence(blocks[cursor + 5])
        use_time_first_seq = _parse_sequence(blocks[cursor + 6])
        use_time_seq = _parse_sequence(blocks[cursor + 7])
        num_hint_seq = _parse_sequence(blocks[cursor + 8])
        mask_seq = _parse_sequence(blocks[cursor + 9])
        cursor += expected_per_user

        if len(question_seq) != len(correctness_seq):
            raise ValueError(f'用户 {user_id} 的题目序列与答题标签长度不一致。')

        trimmed_len = min(seq_len, len(question_seq))
        skills.append(question_seq[:trimmed_len])
        labels.append(correctness_seq[:trimmed_len])
        real_lengths.append(trimmed_len)

        extra_fields['num_attempt_seq'].append(num_attempt_seq[:trimmed_len])
        extra_fields['use_time_first_seq'].append(use_time_first_seq[:trimmed_len])
        extra_fields['use_time_seq'].append(use_time_seq[:trimmed_len])
        extra_fields['num_hint_seq'].append(num_hint_seq[:trimmed_len])
        extra_fields['mask_seq'].append(mask_seq[:trimmed_len])
        extra_fields['school_id'].append(school_id)

    max_len = max(len(seq) for seq in skills)
    skill_arr = np.zeros((len(skills), max_len), dtype=np.int32)
    label_arr = np.zeros((len(skills), max_len), dtype=np.float32)
    num_attempt_arr = np.zeros((len(skills), max_len), dtype=np.int32)
    use_time_first_arr = np.zeros((len(skills), max_len), dtype=np.int32)
    use_time_arr = np.zeros((len(skills), max_len), dtype=np.int32)
    num_hint_arr = np.zeros((len(skills), max_len), dtype=np.int32)
    mask_arr = np.zeros((len(skills), max_len), dtype=np.int32)

    mapping = _build_skill_mapping(skills)
    for idx, (skill_seq, label_seq) in enumerate(zip(skills, labels)):
        mapped_skill = [mapping[value] for value in skill_seq]
        length = len(mapped_skill)
        skill_arr[idx, :length] = mapped_skill
        label_arr[idx, :length] = label_seq
        num_attempt_arr[idx, :length] = extra_fields['num_attempt_seq'][idx][:length]
        use_time_first_arr[idx, :length] = extra_fields['use_time_first_seq'][idx][:length]
        use_time_arr[idx, :length] = extra_fields['use_time_seq'][idx][:length]
        num_hint_arr[idx, :length] = extra_fields['num_hint_seq'][idx][:length]
        mask_arr[idx, :length] = extra_fields['mask_seq'][idx][:length]

    real_len_arr = np.asarray(real_lengths, dtype=np.int32)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset_name}.npz')
    np.savez(
        output_path,
        skill=skill_arr,
        y=label_arr,
        real_len=real_len_arr,
        num_attempt=num_attempt_arr,
        use_time_first=use_time_first_arr,
        use_time=use_time_arr,
        num_hint=num_hint_arr,
        mask=mask_arr,
    )

    if save_mapping:
        mapping_path = os.path.join(output_dir, f'{dataset_name}_skill_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)

    print(f'成功保存数据到 {output_path}')
    if save_mapping:
        print(f'技能编号映射已写入 {mapping_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将自定义序列 CSV 转换为 SRC 所需的 NPZ 格式。')
    parser.add_argument('--input', required=True, help='原始 CSV 文件路径。')
    parser.add_argument('--output_dir', default='./data/custom', help='NPZ 输出目录。')
    parser.add_argument('--name', default='custom', help='保存的 npz 文件名（不含扩展名）。')
    parser.add_argument('--no-mapping', action='store_true', help='不额外保存技能映射 JSON 文件。')
    args = parser.parse_args()

    convert(args.input, args.output_dir, args.name, save_mapping=not args.no_mapping)