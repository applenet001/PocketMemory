"""Prepare 3,000 memories; see hgp3_data.py for exact sources and mappings.

Default source counts: TriMEM=600, PerLTQA=900, proScript=600, IMCS21=300,
MobileMem=600. These are source quotas, not class-balance guarantees.
Split 80/10/10 by source. MobileMem holds sessions apart, not users, as requested.
PerLTQA person / IMCS case / proScript scenario groups held apart. A fixed
hashed group assignment is used BEFORE sampling; exact text duplicates are removed.
MobileMem reads messages only, not generated QA, profiles or graph annotations.
No API calls. External label mappings are weak supervision where noted.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from hgp3_data import ROOT, QUOTAS, candidates, digest, normalized, write


def prepare(out, seed=42):
    out = Path(out)
    if (out / 'candidates.jsonl').exists():
        raise ValueError('Output already prepared; choose a new directory to preserve annotations')
    pools = {(source, split): [] for source in QUOTAS for split in ('train', 'val', 'test')}
    seen = set()
    for r in candidates():
        key = normalized(r['text'])
        if key in seen:
            continue
        seen.add(key)
        bucket = int(digest(f'{seed}:{r["group_id"]}')[:8], 16) % 100
        r['split'] = 'train' if bucket < 80 else 'val' if bucket < 90 else 'test'
        if r['source'] != 'MobileMem':
            r.update(label=r['native_hint'], annotation_source='native' if r['source'] == 'TriMEM' else 'structure_mapping',
                     reviewed=False, uncertain=False,
                     weak_supervision=r['source'] in {'proScript', 'IMCS21'})
        pools[r['source'], r['split']].append(r)
    selected, reserve = [], []
    rng = random.Random(seed)
    availability = {}
    for source, quota in QUOTAS.items():
        sizes = [('train', quota * 8 // 10), ('val', quota // 10), ('test', quota // 10)]
        for split, n in sizes:
            pool = pools[source, split]
            rng.shuffle(pool)
            availability[f'{source}/{split}'] = len(pool)
            if len(pool) < n:
                raise ValueError(f'Insufficient candidates: {source}/{split}: {len(pool)} < {n}; change seed or quotas explicitly')
            selected.extend(pool[:n])
            reserve.extend(pool[n:])
    write(out / 'candidates.jsonl', selected)
    write(out / 'reserve.jsonl', reserve)
    summary = dict(seed=seed, total=len(selected), source_quotas=QUOTAS,
                   split_counts=dict(Counter(r['split'] for r in selected)),
                   initial_class_counts=dict(Counter(r.get('label', 'pending') for r in selected)),
                   available=availability, manual_fraction_mobilemem=0.2,
                   reference='https://arxiv.org/html/2610.11573v1#A2.SS3',
                   note='20% MobileMem audit plus flags; not 20% of all 3000. External mapped labels are not human gold.')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(ROOT / 'outputs/hgp3-3000'))
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    prepare(a.out, a.seed)
