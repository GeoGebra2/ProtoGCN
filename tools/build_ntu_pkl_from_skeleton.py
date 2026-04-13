import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np


NTU60_XSUB_TRAIN = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
}

NTU120_XSUB_TRAIN = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
    83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
}

FILE_NAME_PATTERN = re.compile(r'^S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})$')


def parse_args():
    parser = argparse.ArgumentParser(description='Build NTU pickle annotations from raw skeleton files')
    parser.add_argument('--skeleton-dir', required=True, help='Directory containing NTU .skeleton files')
    parser.add_argument('--output-pkl', required=True, help='Output annotation pickle path')
    parser.add_argument('--dataset', choices=['ntu60', 'ntu120'], default='ntu60')
    parser.add_argument('--split', choices=['xview', 'xsub', 'xset', 'none'], default='xview')
    parser.add_argument('--label-source', choices=['subject', 'action'], default='subject')
    parser.add_argument('--subject-map-json', default=None, help='JSON path for subject->label mapping')
    parser.add_argument('--keep-subjects', type=int, nargs='*', default=None)
    parser.add_argument('--compact-labels', action='store_true')
    parser.add_argument('--num-person', type=int, default=2)
    parser.add_argument('--ext', default='.skeleton')
    parser.add_argument('--skip-bad-files', action='store_true')
    parser.add_argument('--pickle-protocol', type=int, default=4)
    return parser.parse_args()


def parse_name(stem):
    match = FILE_NAME_PATTERN.match(stem)
    if match is None:
        raise ValueError(f'Invalid NTU sample name: {stem}')
    setup, camera, subject, repeat, action = map(int, match.groups())
    return {
        'setup': setup,
        'camera': camera,
        'subject': subject,
        'repeat': repeat,
        'action': action
    }


def to_int_key_dict(mapping):
    return {int(k): int(v) for k, v in mapping.items()}


def read_subject_map(path):
    if path is None:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return to_int_key_dict(data)


def select_label(meta, label_source, subject_map):
    subject_id = meta['subject']
    action_id = meta['action']
    if subject_map is not None:
        if subject_id not in subject_map:
            raise ValueError(f'Subject {subject_id} not found in subject map')
        return int(subject_map[subject_id])
    if label_source == 'subject':
        return subject_id - 1
    return action_id - 1


def parse_skeleton_file(file_path, num_person):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    if len(lines) == 0:
        raise ValueError(f'Empty file: {file_path}')
    cursor = 0
    total_frames = int(lines[cursor])
    cursor += 1
    body_tracks = {}
    body_scores = {}

    for frame_idx in range(total_frames):
        body_count = int(lines[cursor])
        cursor += 1
        for _ in range(body_count):
            body_info = lines[cursor].split()
            cursor += 1
            body_id = body_info[0]
            joint_count = int(lines[cursor])
            cursor += 1
            if body_id not in body_tracks:
                body_tracks[body_id] = np.zeros((total_frames, joint_count, 3), dtype=np.float32)
                body_scores[body_id] = 0.0
            for joint_idx in range(joint_count):
                joint_data = lines[cursor].split()
                cursor += 1
                xyz = np.array([float(joint_data[0]), float(joint_data[1]), float(joint_data[2])], dtype=np.float32)
                body_tracks[body_id][frame_idx, joint_idx] = xyz
                body_scores[body_id] += float(np.linalg.norm(xyz))

    if len(body_tracks) == 0:
        keypoint = np.zeros((num_person, total_frames, 25, 3), dtype=np.float32)
        return keypoint, total_frames

    ranked_body_ids = sorted(body_tracks.keys(), key=lambda k: body_scores[k], reverse=True)
    selected_body_ids = ranked_body_ids[:num_person]
    keypoint_list = [body_tracks[body_id] for body_id in selected_body_ids]
    keypoint = np.stack(keypoint_list, axis=0).astype(np.float32)
    return keypoint, total_frames


def build_split(sample_metas, split_mode, dataset):
    if split_mode == 'none':
        ids = [x['id'] for x in sample_metas]
        return {'train': ids, 'val': ids}

    if split_mode == 'xview':
        train_ids = [x['id'] for x in sample_metas if x['camera'] in {2, 3}]
        val_ids = [x['id'] for x in sample_metas if x['camera'] not in {2, 3}]
        return {'xview_train': train_ids, 'xview_val': val_ids}

    if split_mode == 'xset':
        train_ids = [x['id'] for x in sample_metas if x['setup'] % 2 == 0]
        val_ids = [x['id'] for x in sample_metas if x['setup'] % 2 == 1]
        return {'xset_train': train_ids, 'xset_val': val_ids}

    if dataset == 'ntu60':
        train_subjects = NTU60_XSUB_TRAIN
    else:
        train_subjects = NTU120_XSUB_TRAIN
    train_ids = [x['id'] for x in sample_metas if x['subject'] in train_subjects]
    val_ids = [x['id'] for x in sample_metas if x['subject'] not in train_subjects]
    return {'xsub_train': train_ids, 'xsub_val': val_ids}


def main():
    args = parse_args()
    skeleton_dir = Path(args.skeleton_dir)
    ext = args.ext if args.ext.startswith('.') else f'.{args.ext}'
    files = sorted(list(skeleton_dir.rglob(f'*{ext}')))
    if len(files) == 0:
        raise ValueError(f'No files found in {skeleton_dir} with extension {ext}')

    subject_map = read_subject_map(args.subject_map_json)
    keep_subjects = set(args.keep_subjects) if args.keep_subjects is not None else None

    annotations = []
    sample_metas = []
    failed = []

    for file_path in files:
        sample_id = file_path.stem
        try:
            meta = parse_name(sample_id)
            if keep_subjects is not None and meta['subject'] not in keep_subjects:
                continue
            label = select_label(meta, args.label_source, subject_map)
            keypoint, total_frames = parse_skeleton_file(file_path, num_person=args.num_person)
            annotations.append({
                'frame_dir': sample_id,
                'label': int(label),
                'keypoint': keypoint,
                'total_frames': int(total_frames)
            })
            meta['id'] = sample_id
            sample_metas.append(meta)
        except Exception as exc:
            failed.append((str(file_path), str(exc)))
            if not args.skip_bad_files:
                raise

    if len(annotations) == 0:
        raise ValueError('No valid samples were parsed')

    if args.compact_labels:
        unique_labels = sorted({item['label'] for item in annotations})
        remap = {old: idx for idx, old in enumerate(unique_labels)}
        for item in annotations:
            item['label'] = remap[item['label']]

    split = build_split(sample_metas, split_mode=args.split, dataset=args.dataset)
    output = {
        'split': split,
        'annotations': annotations
    }

    output_path = Path(args.output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output, f, protocol=args.pickle_protocol)

    print(f'samples: {len(annotations)}')
    print(f'split keys: {list(split.keys())}')
    print(f'labels: {len(set(x["label"] for x in annotations))}')
    if len(failed) > 0:
        print(f'failed: {len(failed)}')
        for path, reason in failed[:20]:
            print(f'  {path} -> {reason}')


if __name__ == '__main__':
    main()
