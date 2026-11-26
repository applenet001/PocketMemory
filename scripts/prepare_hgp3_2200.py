"""Reproduce the HGP3-2200 training pool and its held-out evaluation splits."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EXPECTED = {
    'train': {'PerLTQA': 1000, 'proScript': 400, 'MobileMem': 800},
    'val': {'PerLTQA': 97, 'proScript': 54, 'MobileMem': 192},
    'test': {'PerLTQA': 84, 'proScript': 58, 'MobileMem': 192},
}
SEMANTIC_COUNTS = {'train': 500, 'val': 50, 'test': 50}


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(line) for line in (ROOT / path).read_text(encoding='utf-8-sig').split('\n') if line.strip()]


def rank(row, namespace):
    return hashlib.sha256(f'{namespace}:{SEED}:{row["id"]}'.encode()).hexdigest()


def train_rank(row):
    return rank(row, 'hgp3-2400')


def eval_rank(row):
    return rank(row, 'hgp3-2200-external-evaluation')


def normalize(text):
    return re.sub(r'[\s\ufeff]+', '', text).lower()


def candidate(id, source, group_id, text, label, annotation_source):
    return dict(id=id, source=source, group_id=group_id, text=text.strip(), label=label,
                working_state=None, split='train', annotation_source=annotation_source, reviewed=False)


def take_distinct(pool, count, seen, rank_key):
    selected = []
    for row in sorted(pool, key=rank_key):
        text = normalize(row['text'])
        if not text or text in seen:
            continue
        seen.add(text)
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError(f'Expected {count} distinct rows, found {len(selected)}')


def load_external():
    events, profiles = {}, {}
    for i, person in enumerate(read_json('data/external/PerLTQA/Dataset/zh/perltmem.json')):
        group = f'PerLTQA:{i}'
        events[group] = [
            candidate(f'{group}:{key}', 'PerLTQA', group, event['content'], 'episodic', 'mapped_event')
            for key, event in person.get('events', {}).items()
            if isinstance(event, dict) and isinstance(event.get('content'), str)
        ]
        profiles[group] = []
        description = person.get('profile_description')
        if isinstance(description, str) and description.strip():
            profiles[group].append(candidate(f'{group}:profile', 'PerLTQA', group, description,
                                             'semantic', 'mapped_profile'))
        for key, relation in person.get('social_relationship', {}).items():
            text = relation.get('Description') if isinstance(relation, dict) else relation
            if isinstance(text, str) and text.strip():
                row = candidate(f'{group}:relationship:{key}', 'PerLTQA', group, text,
                                'semantic', 'mapped_relationship')
                profiles[group].append(row)
    pro = []
    for row in read_jsonl('outputs/proscript-dialogue-600/synthetic.jsonl'):
        if row['review_required'] is False and row['auto_check']['pass'] is True and row['source_issue'] is False:
            group = row.get('group_id') or row.get('parent_id') or row['id']
            pro.append(candidate(row['id'], 'proScript', group, row['text'], 'procedural', 'synthetic_auto_verified'))
            pro[-1].update(split=row['split'], upstream_split=row['split'])
    return events, profiles, pro


def messages(session):
    return [row for row in session['messages'] if isinstance(row.get('content'), str) and row['content'].strip()]


def mobile_row(session, message, split):
    id = 'MobileMem:' + message['id']
    group = 'MobileMem:' + session['id']
    metadata = dict(session_id=session['id'], message_id=message['id'])
    metadata.update((key, message[key]) for key in ('role', 'name', 'timestamp') if key in message)
    if split == 'train':
        return dict(candidate(id, 'MobileMem', group, message['content'], None, 'pending'),
                    **metadata, annotation_status='pending')
    return dict(id=id, source='MobileMem', group_id=group, **metadata, text=message['content'].strip(),
                split=split, label=None, working_state=None, annotation_source='pending',
                annotation_status='pending', reviewed=False)


def mobile_session_options(sessions, count):
    options = [{} for _ in range(count + 1)]
    options[0][0] = []
    for session in sorted(sessions, key=lambda row: rank(row, 'hgp3-eval')):
        turns = len(messages(session))
        for n in range(count, 0, -1):
            for total, ids in options[n - 1].items():
                options[n].setdefault(total + turns, [*ids, session['id']])
    return options[count]


def choose_mobile_val(held_users):
    if [len(sessions) for sessions in held_users] != [23, 19]:
        raise ValueError('Unexpected MobileMem held-out sessions')
    totals = [sum(len(messages(session)) for session in sessions) for sessions in held_users]
    options_a = mobile_session_options(held_users[0], 12)
    options_b = mobile_session_options(held_users[1], 9)
    a, b = min(
        ((a, b) for a in options_a for b in options_b),
        key=lambda pair: (1000 * abs(sum(pair) - 192)
                          + abs(2 * pair[0] - totals[0]) + abs(2 * pair[1] - totals[1])),
    )
    return set(options_a[a] + options_b[b])


def choose_perltqa_val(people, profiles):
    if len(people) != 11:
        raise ValueError('Unexpected PerLTQA held-out people')
    total = sum(len(rows) for rows in people.values())
    best_score = float('inf')
    best = None
    for mask in range(1, (1 << len(people)) - 1):
        groups = {group for i, group in enumerate(people) if mask & (1 << i)}
        val_text = {text for group in groups for text in profiles[group]}
        test_text = {text for group in people.keys() - groups for text in profiles[group]}
        if min(len(val_text - test_text), len(test_text - val_text)) < 50:
            continue
        count = sum(len(people[group]) for group in groups)
        score = 100 * abs(2 * count - total) + abs(len(people) - 2 * len(groups))
        if score < best_score:
            best_score, best = score, groups
    if best is None:
        raise ValueError('Cannot allocate 50 distinct PerLTQA semantic rows per holdout split')
    return best


def verify(buckets):
    ids, groups, texts = set(), {}, {}
    for split, rows in buckets.items():
        counts = Counter(row['source'] for row in rows)
        if counts != EXPECTED[split]:
            raise ValueError(f'Unexpected {split} counts: {counts}')
        semantic = sum(row['source'] == 'PerLTQA' and row['label'] == 'semantic' for row in rows)
        if semantic != SEMANTIC_COUNTS[split]:
            raise ValueError(f'Unexpected {split} PerLTQA semantic count: {semantic}')
        for row in rows:
            text = normalize(row['text'])
            if not text or row['id'] in ids or row['split'] != split:
                raise ValueError(f'Invalid or duplicate row: {row["id"]}')
            ids.add(row['id'])
            if groups.setdefault(row['group_id'], split) != split or texts.setdefault(text, split) != split:
                raise ValueError(f'Text or group shared across splits: {row["id"]}')


def build_splits():
    per, profiles, pro = load_external()
    mobile = read_json('data/text/mobilemem_data.json')
    held = [
        session['id'] for user in mobile
        for session in sorted(user['sessions'], key=train_rank)[:(len(user['sessions']) + 4) // 5]
    ]
    held_ids = set(held)
    mobile_train = (
        mobile_row(session, message, 'train')
        for user in mobile for session in user['sessions'] if session['id'] not in held_ids
        for message in messages(session)
    )
    seen = set()
    # Recreate the initial pool before retaining its first 800 MobileMem turns.
    parent = sorted([
        *take_distinct((row for row in pro if row['split'] == 'train'), 400, seen, train_rank),
        *take_distinct((row for events in per.values() for row in events), 500, seen, train_rank),
        *take_distinct(mobile_train, 1000, seen, train_rank),
    ], key=train_rank)
    excluded_ids = {row['id'] for row in [r for r in parent if r['source'] == 'MobileMem'][800:]}
    train = [row for row in parent if row['id'] not in excluded_ids]
    train_groups = {row['group_id'] for row in train}
    used_texts = {normalize(row['text']) for row in train}
    held_users = [[session for session in user['sessions'] if session['id'] in held_ids] for user in mobile]
    mobile_val = choose_mobile_val(held_users)
    evaluation = {'val': [], 'test': []}
    for sessions in held_users:
        for session in sessions:
            split = 'val' if session['id'] in mobile_val else 'test'
            for message in messages(session):
                row = mobile_row(session, message, split)
                evaluation[split].append(row)
                used_texts.add(normalize(row['text']))
    per_eligible = {}
    for group, events in per.items():
        if group not in train_groups:
            rows = [row for row in events if row['text'] and normalize(row['text']) not in used_texts]
            if rows:
                per_eligible[group] = rows
    for row in pro:
        split = row['split']
        if split in evaluation and row['group_id'] not in train_groups and normalize(row['text']) not in used_texts:
            # Preserve the field order of the original evaluation files.
            fields = {key: value for key, value in row.items() if key not in ('annotation_source', 'reviewed')}
            evaluation[split].append(dict(fields, annotation_source=row['annotation_source'], reviewed=False))
    used_texts.update(normalize(row['text']) for rows in evaluation.values() for row in rows)
    used_texts.update(normalize(row['text']) for rows in per_eligible.values() for row in rows)
    semantic_texts = {
        group: {normalize(row['text']) for row in profiles[group]} - used_texts for group in per_eligible
    }
    per_val = choose_perltqa_val(per_eligible, semantic_texts)
    per_holdout = {'val': per_val, 'test': per_eligible.keys() - per_val}
    for group, rows in per_eligible.items():
        split = 'val' if group in per_val else 'test'
        evaluation[split].extend(dict(row, split=split) for row in rows)
    train.extend(take_distinct(
        (row for group in profiles.keys() & train_groups for row in profiles[group]),
        SEMANTIC_COUNTS['train'], used_texts, train_rank,
    ))
    for split, groups in per_holdout.items():
        pool = (row for group in groups for row in profiles[group] if normalize(row['text']) in semantic_texts[group])
        picked = take_distinct(pool, SEMANTIC_COUNTS[split], used_texts, eval_rank)
        evaluation[split].extend(dict(row, split=split) for row in picked)
    buckets = dict(train=sorted(train, key=train_rank),
                   **{split: sorted(rows, key=eval_rank) for split, rows in evaluation.items()})
    verify(buckets)
    return buckets, held


def carry_mobile_annotations(buckets, annotation_path):
    previous = read_json(annotation_path)
    result = {}
    for split, rows in buckets.items():
        old = {row['id']: row for row in previous[split] if row['source'] == 'MobileMem'}
        current = {row['id']: row for row in rows if row['source'] == 'MobileMem'}
        if old.keys() != current.keys():
            raise ValueError(f'MobileMem IDs changed in {split}; cannot safely carry over annotations')
        for id, row in current.items():
            if any(row[field] != old[id][field] for field in ('text', 'split', 'session_id', 'role')):
                raise ValueError(f'MobileMem evidence changed for {id}')
        result[split] = [old[row['id']] if row['source'] == 'MobileMem' else row for row in rows]
    return result


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as file:
        file.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_manifests(out, buckets, held, annotated=None):
    split_stats = {}
    for split, rows in (annotated or buckets).items():
        pending = sum(row.get('annotation_status') == 'pending' for row in rows)
        split_stats[split] = dict(
            total=len(rows), sources=EXPECTED[split], labeled=len(rows) - pending, pending=pending,
            perltqa_person_groups=sorted({row['group_id'] for row in rows if row['source'] == 'PerLTQA'}),
            perltqa_labels=dict(Counter(row['label'] for row in rows if row['source'] == 'PerLTQA')),
            mobilemem=dict(
                sessions=len({row['session_id'] for row in rows if row['source'] == 'MobileMem'}),
                turns=EXPECTED[split]['MobileMem'],
            ),
        )
    annotation_prompt = 'scripts/hgp-message-label-prompt.txt'
    write_json(out / 'manifest.json', dict(
        seed=SEED, **split_stats['train'], mobilemem_reserved_session_ids=held,
        perltqa_language='zh', annotation_prompt=annotation_prompt,
    ))
    write_json(out / 'evaluation-manifest.json', dict(
        seed=SEED, annotation_prompt=annotation_prompt,
        source_protocols={
            'PerLTQA': dict(language='zh', group_unit='person', labels={
                'semantic': ['profile_description', 'social_relationship'], 'episodic': ['events'],
            }),
            'proScript': dict(group_unit='scenario', annotation='synthetic_auto_verified'),
            'MobileMem': dict(group_unit='session', source_reservation='manifest.json'),
        },
        splits={split: split_stats[split] for split in ('val', 'test')},
    ))


def prepare(out, annotations=None):
    buckets, held = build_splits()
    annotated = carry_mobile_annotations(buckets, annotations) if annotations else None
    filenames = dict(train='candidates.jsonl', val='val.jsonl', test='test.jsonl')
    paths = [out / name for name in (*filenames.values(), 'manifest.json', 'evaluation-manifest.json')]
    if annotated is not None:
        paths.append(out / 'annotations-unified/dataset.json')
    if any(path.exists() for path in paths):
        raise FileExistsError(f'Output already prepared: {out}. Use a different --out directory.')
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in buckets.items():
        with (out / filenames[split]).open('x', encoding='utf-8', newline='\n') as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    write_manifests(out, buckets, held, annotated)
    if annotated is not None:
        destination = out / 'annotations-unified/dataset.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, annotated)
    print(json.dumps({split: len(rows) for split, rows in buckets.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/hgp3-train-2200-rebuilt')
    parser.add_argument('--annotations', type=Path, help='Carry existing MobileMem labels into the new dataset.')
    args = parser.parse_args()
    prepare(args.out, args.annotations)
