"""HGP three-class data loader. Working memory is a separate state-update task.

Sources (native annotations are hints, NOT unified HGP gold labels):
TriMEM: CSV Content + binary episodic/personal/general semantic annotations.
PerLTQA zh: profile_description/social_relationship -> semantic; events -> episodic.
proScript v1a: scenarios, events and dependency edges -> procedural candidate.
IMCS21: Inform-Drug_Recommendation/Medical_Advice/Precautions -> procedural candidate.
MobileMem: ONLY session messages; never profiles, graphs, QA or answer evidence.
Only MobileMem receives DeepSeek annotation. External labels retain native/mapped
provenance; proScript and IMCS21 mappings are weak supervision, not HGP gold labels.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['semantic', 'episodic', 'procedural']
# Engineering defaults, not a claim about TriMEM's annotation protocol.
QUOTAS = {'TriMEM': 600, 'PerLTQA': 900, 'proScript': 600, 'IMCS21': 300, 'MobileMem': 600}


def read(path):
    p = Path(path)
    return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()] if p.exists() else []


def write(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def candidates(root=ROOT):
    def emit(source, group, key, text, hint=None, context=''):
        text = str(text).strip()
        # Bounded contiguous evidence chunks, retaining the original offsets.
        if source != 'MobileMem' and len(text) > 3000:
            return  # Preserve the unit carrying the original annotation.
        size = 1000 if source == 'MobileMem' else max(1, len(text))
        for start in range(0, len(text), size):
            chunk = text[start:start + size].strip()
            if len(chunk) < 20:
                continue
            yield dict(id=f'{source}:{key}:{start}', source=source,
                       group_id=f'{source}:{group}', text=chunk, context=context,
                       source_offset=start, native_hint=hint,
                       chunked=len(text) > size)

    path = root / 'data/external/MemoType/TriMEM/TriMEM.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        for i, r in enumerate(csv.DictReader(f)):
            labels = []
            if int(r['Episodic Memory']):
                labels.append('episodic')
            if int(r['Personal Semantic Memory']) or int(r['General Semantic Memory']):
                labels.append('semantic')
            if len(labels) == 1:
                yield from emit('TriMEM', i, i, r['Content'], labels[0])
    people = json.loads((root / 'data/external/PerLTQA/Dataset/zh/perltmem.json').read_text(encoding='utf-8'))
    for i, person in enumerate(people):
        for section in ('profile_description', 'social_relationship', 'events'):
            values = person.get(section, {})
            if isinstance(values, str):
                values = {'description': values}
            if isinstance(values, list):
                values = dict(enumerate(values))
            for key, value in values.items():
                if isinstance(value, dict):
                    text = value.get('content', value.get('description', json.dumps(value, ensure_ascii=False)))
                else:
                    text = str(value)
                hint = 'episodic' if section == 'events' else 'semantic'
                yield from emit('PerLTQA', i, f'{i}:{section}:{key}', text, hint)
    for path in sorted((root / 'data/external/proScript/proscript_v1a').glob('*.jsonl')):
        for i, r in enumerate(read(path)):
            # Same scenario stays together even if present in upstream multiple splits.
            scenario = r['scenario'].strip()
            text = scenario + '\n' + r['flatten_output_for_script_generation']
            yield from emit('proScript', scenario.lower(), f'{path.stem}:{i}', text, 'procedural')
    # Never consume IMCS test_input duplicate or hidden evaluation labels.
    for upstream in ('train', 'dev'):
        path = root / f'data/external/IMCS21-original/dataset/{upstream}.json'
        for case, r in json.loads(path.read_text(encoding='utf-8')).items():
            previous = []
            for turn in r['dialogue']:
                text = turn['sentence']
                act = turn['dialogue_act']
                if act in {'Inform-Drug_Recommendation', 'Inform-Medical_Advice', 'Inform-Precautions'}:
                    yield from emit('IMCS21', case, f'{case}:{turn["sentence_id"]}', text, 'procedural', '\n'.join(previous[-2:])[-1000:])
                previous.append(text)
    path = root / 'data/text/mobilemem_data.json'
    for person in json.loads(path.read_text(encoding='utf-8')):
        for session in person['sessions']:
            previous = []
            for i, message in enumerate(session['messages']):
                text = message.get('content', '')
                if not isinstance(text, str):
                    continue
                yield from emit('MobileMem', session['id'], f'{session["id"]}:{i}', text,
                                context='\n'.join(previous[-2:])[-1000:])
                previous.append(text)


def normalized(text):
    return re.sub(r'\s+', '', text).casefold()


def model_input(row):
    """Same evidence and context seen by the teacher and the lightweight router."""
    context = row.get('context', '').strip()
    return f'Context (disambiguation only):\n{context}\nTarget memory:\n{row["text"]}' if context else row['text']


def load_training(path, expected=3000):
    rows = read(path)
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} usable memories, got {len(rows)}')
    ids, groups, texts = set(), {}, {}
    for r in rows:
        if r['id'] in ids or r.get('label') not in LABELS or not r['text'].strip():
            raise ValueError('Duplicate ID, invalid label or empty memory')
        ids.add(r['id'])
        if r['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Invalid split')
        for mapping, key in ((groups, r['group_id']), (texts, normalized(r['text']))):
            if mapping.setdefault(key, r['split']) != r['split']:
                raise ValueError('Group/text leakage')
        if r.get('uncertain') or r.get('excluded'):
            raise ValueError('Unresolved/excluded row')
        if r.get('review_required') and not r.get('reviewed'):
            raise ValueError('Required human review missing')
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != set(LABELS):
            raise ValueError(f'Missing classes in {split}')
    return rows
