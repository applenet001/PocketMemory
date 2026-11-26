"""Label MobileMem turns with DeepSeek and prepare a manual review queue."""

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hgp3_data import LABELS, ROOT, digest, read


SPLITS = {'train': 'candidates.jsonl', 'val': 'val.jsonl', 'test': 'test.jsonl'}
EXPECTED = {'train': 800, 'val': 192, 'test': 192}
PROMPT_PATH = ROOT / 'scripts/hgp-message-label-prompt.txt'
MODEL = 'deepseek-flash'


def atomic_write(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8', newline='\n')
    temporary.replace(path)


def api_key():
    key = os.getenv('DEEPSEEK_API_KEY', '').strip()
    if not key:
        env = ROOT / 'config/local.env'
        if env.exists():
            for line in env.read_text(encoding='utf-8-sig').splitlines():
                field, _, value = line.partition('=')
                if field.strip() == 'DEEPSEEK_API_KEY':
                    key = value.strip().strip('"').strip("'")
                    break
    if not key:
        raise ValueError('Missing DEEPSEEK_API_KEY')
    return key


def parse_labels(answer, count):
    choice = answer['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('Incomplete model response')
    items = json.loads(choice['message']['content'])['items']
    labels = {}
    for item in items:
        alias = str(item['id'])
        if alias in labels:
            raise ValueError('Duplicate response ID')
        label = item['label']
        if isinstance(label, str):
            label = label.strip().lower()
            if label == 'null':
                label = None
        if label not in LABELS + [None]:
            raise ValueError('Invalid main label')
        if type(item['working_state']) is not bool or type(item['uncertain']) is not bool:
            raise ValueError('Invalid working_state or uncertain value')
        if not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError('Missing annotation reason')
        labels[alias] = {field: item[field] for field in ('label', 'working_state', 'uncertain', 'reason')}
        labels[alias]['label'] = label
    if set(labels) != {str(i) for i in range(count)}:
        raise ValueError('Response IDs do not match request')
    return labels


def annotate(batch, prompt, key):
    payload = [dict(id=str(i), role=row.get('role'), target=row['text']) for i, row in enumerate(batch)]
    body = dict(model=MODEL, temperature=0, max_tokens=4096, thinking={'type': 'disabled'},
                response_format={'type': 'json_object'},
                messages=[dict(role='system', content=prompt),
                          dict(role='user', content=json.dumps(payload, ensure_ascii=False))])
    request = Request('https://api.deepseek.com/chat/completions', data=json.dumps(body).encode(),
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    usage = Counter()
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                answer = json.load(response)
            usage.update({k: v for k, v in answer.get('usage', {}).items() if type(v) is int})
            annotations = parse_labels(answer, len(batch))
            return dict(model=answer['model'], usage=dict(usage), items=[
                dict(id=row['id'], **annotations[str(i)], text_sha256=digest(row['text']))
                for i, row in enumerate(batch)
            ])
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f'DeepSeek returned HTTP {exc.code}') from None
        except ValueError as exc:
            if attempt == 2:
                raise RuntimeError(f'DeepSeek response validation failed: {exc}') from None
        except (URLError, TimeoutError, KeyError, TypeError, IndexError) as exc:
            if attempt == 2:
                raise RuntimeError(f'DeepSeek request or response failed: {type(exc).__name__}') from None
        time.sleep(2 ** attempt)


def summarize(destination, splits, annotations, usage):
    summary = {}
    dataset = {}
    if destination.exists():
        previous = json.loads(destination.read_text(encoding='utf-8'))
        existing_review = {
            row['id']: row['human_review'] for rows in previous.values()
            for row in rows if 'human_review' in row
        }
    else:
        existing_review = {}
    for split, rows in splits.items():
        counts = Counter()
        merged = []
        for row in rows:
            result = annotations.get(row['id'])
            if result is None:
                merged.append(row)
                continue
            fields = {k: result[k] for k in ('label', 'working_state', 'uncertain', 'reason')}
            item = row | fields | dict(
                annotation_source='deepseek', reviewed=False,
                annotation_status='needs_human_review' if result['uncertain'] else 'teacher_labeled',
            )
            counts['labeled'] += 1
            counts['uncertain'] += result['uncertain']
            if result['uncertain']:
                item['human_review'] = existing_review.get(row['id'], dict(
                    label=None, working_state=None, reviewed=False, reviewer='', note='',
                ))
            merged.append(item)
        dataset[split] = merged
        summary[split] = dict(total=EXPECTED[split], labeled=counts['labeled'], uncertain=counts['uncertain'])
    if all(counts['total'] == counts['labeled'] for counts in summary.values()):
        atomic_write(destination, json.dumps(dataset, ensure_ascii=False, indent=2) + '\n')
    report = dict(splits=summary, token_usage=dict(usage),
                  review_rows=sum(counts['uncertain'] for counts in summary.values()))
    print(json.dumps(report, ensure_ascii=False), flush=True)


def run(data, out, batch_size, workers, max_batches):
    splits = {split: read(data / filename) for split, filename in SPLITS.items()}
    rows = [row for items in splits.values() for row in items if row['source'] == 'MobileMem']
    if Counter(row['split'] for row in rows) != EXPECTED or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Unexpected MobileMem split counts or duplicate message IDs')
    prompt = PROMPT_PATH.read_text(encoding='utf-8')
    config = dict(model=MODEL, prompt_sha256=digest(prompt), batch_size=batch_size,
                  inputs={split: digest((data / filename).read_text(encoding='utf-8'))
                          for split, filename in SPLITS.items()})
    destination = out / 'dataset.json'
    checkpoint_path = out / '.checkpoint.json'
    if destination.exists() and not checkpoint_path.exists():
        raise FileExistsError(f'{destination} is complete. Use a new --out for a fresh run.')
    out.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if checkpoint['config'] != config:
            raise ValueError('Input, prompt, model or batch size changed. Choose a new output directory.')
    else:
        checkpoint = dict(config=config, batches={})

    def save_checkpoint():
        atomic_write(checkpoint_path, json.dumps(checkpoint, ensure_ascii=False) + '\n')

    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    annotations = {}
    usage = Counter()
    pending = []

    def collect(batch, result):
        if [(row['id'], digest(row['text'])) for row in batch] != [
            (item['id'], item['text_sha256']) for item in result['items']
        ]:
            raise ValueError('Batch cache does not match current input')
        annotations.update((item['id'], item) for item in result['items'])
        usage.update(result['usage'])

    for i, batch in enumerate(batches):
        batch_id = str(i)
        if batch_id in checkpoint['batches']:
            collect(batch, checkpoint['batches'][batch_id])
        else:
            pending.append((batch_id, batch))
    pending = pending[:max_batches]
    failures = []
    if pending:
        key = api_key()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = {executor.submit(annotate, batch, prompt, key): (batch_id, batch) for batch_id, batch in pending}
            for task in as_completed(tasks):
                batch_id, batch = tasks[task]
                try:
                    result = task.result()
                except RuntimeError as exc:
                    failures.append(f'{batch_id}: {exc}')
                    continue
                collect(batch, result)
                checkpoint['batches'][batch_id] = result
                save_checkpoint()
                if len(annotations) % (batch_size * 10) == 0 or len(annotations) == len(rows):
                    print(f'Annotated {len(annotations)}/{len(rows)}', flush=True)
    summarize(destination, splits, annotations, usage)
    if len(annotations) == len(rows):
        checkpoint_path.unlink(missing_ok=True)
    if failures:
        raise SystemExit(f'{len(failures)} batches failed. Rerun to resume. First failure: {failures[0]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'outputs/hgp3-train-2200')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/hgp3-train-2200/annotations-unified')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--max-batches', type=int)
    args = parser.parse_args()
    if min(args.batch_size, args.workers) < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error('Batch size, workers and maximum batches must be positive')
    run(args.data, args.out, args.batch_size, args.workers, args.max_batches)
