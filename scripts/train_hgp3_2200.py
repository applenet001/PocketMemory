"""Train and evaluate the HGP3-2200 router with a frozen Qwen encoder."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from hgp3_data import LABELS, ROOT, normalized


DATA = ROOT / 'outputs/hgp3-train-2200/annotations-unified/dataset.json'
DEFAULT_MODEL = (
    Path.home() / '.cache/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master'
)
ROUTES = [*LABELS, None]
SPLITS = ('train', 'val', 'test')


def load_rows(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if set(data) != set(SPLITS) or len(data['train']) != 2200 or not data['val'] or not data['test']:
        raise ValueError('Expected 2,200 training samples and nonempty validation and test splits')
    seen_ids, groups, texts = set(), {}, {}
    output = []
    for split in SPLITS:
        for row in data[split]:
            text = normalized(row['text'])
            if not text or row['id'] in seen_ids or row['split'] != split:
                raise ValueError(f'Empty, duplicate, or misassigned row: {row["id"]}')
            seen_ids.add(row['id'])
            if groups.setdefault(row['group_id'], split) != split or texts.setdefault(text, split) != split:
                raise ValueError(f'Split leakage for {row["id"]}')
            review = row.get('human_review') or {}
            reviewed = review.get('reviewed') is True
            if row.get('uncertain') and not reviewed:
                raise ValueError(f'Unresolved review: {row["id"]}')
            if row['source'] == 'MobileMem' and row.get('annotation_status') == 'pending':
                raise ValueError(f'Missing MobileMem annotation: {row["id"]}')
            annotation = review if reviewed else row
            label, working = annotation['label'], annotation['working_state']
            if label not in ROUTES or (working is not None and type(working) is not bool):
                raise ValueError(f'Invalid label: {row["id"]}')
            if row['source'] == 'MobileMem' and type(working) is not bool:
                raise ValueError(f'Missing MobileMem working_state: {row["id"]}')
            output.append(dict(row, label=label, working_state=working, reviewed=reviewed))
    return output


def encode(rows, args, data_sha256):
    import numpy as np
    import torch
    key = hashlib.sha256()
    for path in sorted(p for p in args.model.rglob('*') if p.is_file()):
        stat = path.stat()
        key.update(f'{path.relative_to(args.model)}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    encoder_key = f'{args.model.resolve()}:{key.hexdigest()}'
    cache_key = f'{data_sha256}:{encoder_key}'
    cache = args.out / 'embeddings.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            if archive['key'].item() == cache_key:
                return archive['features']
    features = None
    missing = list(range(len(rows)))
    if args.reuse_cache:
        prior_rows = load_rows(args.cache_source_data)
        prior_hash = hashlib.sha256(args.cache_source_data.read_bytes()).hexdigest()
        with np.load(args.reuse_cache, allow_pickle=False) as archive:
            if archive['key'].item() != f'{prior_hash}:{encoder_key}':
                raise ValueError('Prior embedding cache does not match the source data or encoder')
            prior_features = archive['features']
        if prior_features.ndim != 2 or len(prior_features) != len(prior_rows):
            raise ValueError('Prior embedding cache has an unexpected shape')
        prior_index = {(row['id'], row['text']): i for i, row in enumerate(prior_rows)}
        features = np.empty((len(rows), prior_features.shape[1]), dtype=np.float32)
        missing = []
        for i, row in enumerate(rows):
            old_index = prior_index.get((row['id'], row['text']))
            if old_index is None:
                missing.append(i)
            else:
                features[i] = prior_features[old_index]
        print(f'Reused {len(rows) - len(missing)} verified embeddings; {len(missing)} to encode', flush=True)
    if missing:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(str(args.model), local_files_only=True, device='cpu')
        encoder.eval()
        texts = [rows[i]['text'] for i in missing]
        lengths = [len(ids) for ids in encoder.tokenizer(texts, truncation=False, verbose=False)['input_ids']]
        if max(lengths) > encoder.max_seq_length:
            raise ValueError(f'Longest sample is {max(lengths)} tokens, above encoder limit {encoder.max_seq_length}')
        print(f'Encoding {len(texts)} rows on CPU; longest input {max(lengths)} tokens', flush=True)
        order = sorted(range(len(texts)), key=lengths.__getitem__)
        dimension = encoder.get_sentence_embedding_dimension()
        if features is None:
            features = np.empty((len(rows), dimension), dtype=np.float32)
        elif features.shape[1] != dimension:
            raise ValueError('Prior embedding dimension differs from the encoder')
        with torch.inference_mode():
            for start in range(0, len(order), 128):
                batch = order[start:start + 128]
                features[[missing[i] for i in batch]] = encoder.encode(
                    [texts[i] for i in batch], normalize_embeddings=True, batch_size=args.encode_batch_size,
                    show_progress_bar=False,
                )
                print(f'Encoded {start + len(batch)}/{len(texts)}', flush=True)
    temporary = cache.with_suffix('.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file, features=features, key=cache_key)
    temporary.replace(cache)
    return features


def fit_logistic(features, targets, train, validation, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    best = None
    for class_weight in (None, 'balanced'):
        for strength in (0.1, 1.0, 10.0, 100.0):
            model = LogisticRegression(C=strength, class_weight=class_weight, max_iter=1000)
            model.fit(features[train], targets[train])
            score = f1_score(targets[validation], model.predict(features[validation]),
                             labels=labels, average='macro', zero_division=0)
            if best is None or score > best[0]:
                best = (score, model)
    return best[1]


def set_deferral(probabilities, targets, target_agreement, minimum):
    import numpy as np

    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    order = np.argsort(-confidence, kind='stable')
    scores = confidence[order]
    cumulative = np.cumsum(correct[order])
    boundaries = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True]) + 1
    eligible = [n for n in boundaries if n >= minimum and cumulative[n - 1] / n >= target_agreement]
    if not eligible:
        return None
    return float(scores[max(eligible) - 1])


def evaluate_routes(probs, targets, threshold):
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix

    pred = probs.argmax(axis=1)
    labels = list(range(len(ROUTES)))
    accepted = probs.max(axis=1) >= threshold if threshold is not None else np.zeros(len(targets), dtype=bool)
    return dict(
        n=len(targets), accuracy=float((pred == targets).mean()), classification=classification_report(
            targets, pred, labels=labels, target_names=[*LABELS, 'no_primary_memory'],
            output_dict=True, zero_division=0,
        ),
        confusion_matrix=confusion_matrix(targets, pred, labels=labels).tolist(),
        accepted=int(accepted.sum()), coverage=float(accepted.mean()),
        accepted_agreement=float((pred[accepted] == targets[accepted]).mean()) if accepted.any() else None,
    )


def evaluate_route_splits(probabilities, targets, split, source, threshold):
    report = {}
    by_source = {}
    for name in SPLITS:
        selected = split == name
        report[f'all_{name}'] = evaluate_routes(probabilities[selected], targets[selected], threshold)
        by_source[name] = {}
        for origin in sorted(set(source[selected])):
            mask = selected & (source == origin)
            by_source[name][origin] = evaluate_routes(probabilities[mask], targets[mask], threshold)
        report[f'mobilemem_{name}'] = by_source[name]['MobileMem']
    report['by_source'] = by_source
    return report


def evaluate_working_splits(probabilities, targets, split, threshold):
    from sklearn.metrics import classification_report

    return {
        f'mobilemem_{name}': classification_report(
            targets[mask], probabilities[mask] >= threshold, labels=[0, 1],
            target_names=['false', 'true'], output_dict=True, zero_division=0,
        )
        for name in SPLITS if (mask := (split == name) & (targets >= 0)).any()
    }


def evaluate_checkpoint(args, rows, data_sha256):
    import numpy as np
    import torch
    from scipy.special import expit, softmax

    config = json.loads((args.out / 'config.json').read_text(encoding='utf-8'))
    if (config['data_sha256'] != data_sha256 or config['encoder'] != str(args.model.resolve())
            or config['route_labels'] != ROUTES):
        raise ValueError('Checkpoint dataset, encoder or labels do not match the evaluation inputs')
    checkpoint = torch.load(args.out / 'head.pt', map_location='cpu', weights_only=True)
    features = encode(rows, args, data_sha256)
    features = features.astype(np.float64)
    logits = features @ checkpoint['route.weight'].numpy().T + checkpoint['route.bias'].numpy()
    probabilities = softmax(logits / config['route_temperature'], axis=1)
    targets = np.asarray([ROUTES.index(row['label']) for row in rows])
    splits = np.asarray([row['split'] for row in rows])
    sources = np.asarray([row['source'] for row in rows])
    working = np.asarray([int(row['working_state']) if row['working_state'] is not None else -1 for row in rows])
    working_logits = features @ checkpoint['working.weight'].numpy().T + checkpoint['working.bias'].numpy()
    path = args.out / 'metrics.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['route'] = evaluate_route_splits(probabilities, targets, splits, sources, config['confidence_threshold'])
    report['working_state'] = evaluate_working_splits(expit(working_logits.ravel()), working, splits, config['working_threshold'])
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    print(f'Updated classifier metrics for train, val, and test in {path}', flush=True)


def train(args):
    import numpy as np
    import torch
    from scipy.special import expit, softmax
    from sklearn.metrics import f1_score, log_loss

    torch.set_num_threads(args.threads)
    rows = load_rows(args.data)
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if args.evaluate_only:
        evaluate_checkpoint(args, rows, data_sha256)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'head.pt').exists():
        raise FileExistsError(f'Classifier already exists at {args.out}. Choose another --out.')
    x = encode(rows, args, data_sha256)
    y = np.asarray([ROUTES.index(row['label']) for row in rows])
    w = np.asarray([int(row['working_state']) if row['working_state'] is not None else -1 for row in rows])
    source = np.asarray([row['source'] for row in rows])
    split = np.asarray([row['split'] for row in rows])
    mobile = source == 'MobileMem'
    val_mobile = (split == 'val') & mobile
    known_work = w >= 0
    routes = fit_logistic(x, y, split == 'train', val_mobile, list(range(len(ROUTES))))
    working = fit_logistic(x, w, (split == 'train') & known_work, val_mobile & known_work, [0, 1])
    logits = routes.decision_function(x)
    temperatures = np.geomspace(0.25, 4, 101)
    temperature = float(min(temperatures, key=lambda t: log_loss(
        y[val_mobile], softmax(logits[val_mobile] / t, axis=1), labels=range(len(ROUTES)),
    )))
    probabilities = softmax(logits / temperature, axis=1)
    threshold = set_deferral(probabilities[val_mobile], y[val_mobile], args.target_agreement, args.min_accepted)
    work_prob = expit(working.decision_function(x))
    work_threshold = float(max(np.linspace(0.05, 0.95, 91), key=lambda t: (
        f1_score(w[val_mobile], work_prob[val_mobile] >= t, average='macro', zero_division=0), -abs(t - 0.5),
    )))
    report = {
        'route': evaluate_route_splits(probabilities, y, split, source, threshold),
        'working_state': evaluate_working_splits(work_prob, w, split, work_threshold),
        'split_sizes': dict(Counter(split)),
        'working_state_supervision': dict(Counter(split[known_work])),
        'human_reviewed_mobilemem_test': sum(row['reviewed'] for row in rows if row['split'] == 'test' and row['source'] == 'MobileMem'),
        'label_provenance': 'Native, mapped, generated, and mostly DeepSeek labels; metrics measure agreement, not human-gold accuracy.',
    }
    config = dict(
        encoder=str(args.model.resolve()), architecture='frozen_encoder_two_linear_heads',
        route_labels=ROUTES, embedding_dimension=x.shape[1], normalize_embeddings=True,
        input_format='message text', data_sha256=data_sha256,
        route_temperature=temperature, confidence_threshold=threshold, working_threshold=work_threshold,
        validation_target=args.target_agreement, validation_min_accepted=args.min_accepted,
        route_regularization_c=routes.C, route_class_weight=routes.class_weight,
        working_regularization_c=working.C, working_class_weight=working.class_weight,
    )
    checkpoint = {}
    for prefix, fitted in [('route', routes), ('working', working)]:
        checkpoint[f'{prefix}.weight'] = torch.as_tensor(fitted.coef_, dtype=torch.float32)
        checkpoint[f'{prefix}.bias'] = torch.as_tensor(fitted.intercept_, dtype=torch.float32)
    for filename, content in [('config.json', config), ('metrics.json', report)]:
        (args.out / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    torch.save(checkpoint, args.out / 'head.pt')
    print(json.dumps(dict(
        mobilemem_test_route_macro_f1=report['route']['mobilemem_test']['classification']['macro avg']['f1-score'],
        mobilemem_test_working_macro_f1=report['working_state']['mobilemem_test']['macro avg']['f1-score'],
        validation_coverage=report['route']['mobilemem_val']['coverage'],
        test_coverage=report['route']['mobilemem_test']['coverage'],
        threshold=threshold,
    )), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/hgp3-train-2200/classifier')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--encode-batch-size', type=int, default=16)
    parser.add_argument('--reuse-cache', type=Path)
    parser.add_argument('--cache-source-data', type=Path)
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Evaluate the saved checkpoint on all splits without fitting new weights.')
    parser.add_argument('--target-agreement', type=float, default=0.90)
    parser.add_argument('--min-accepted', type=int, default=20)
    args = parser.parse_args()
    if min(args.threads, args.encode_batch_size, args.min_accepted) < 1 or not 0 < args.target_agreement <= 1:
        parser.error('Sizes must be positive and target agreement must be in (0, 1]')
    if bool(args.reuse_cache) != bool(args.cache_source_data):
        parser.error('--reuse-cache and --cache-source-data must be used together')
    train(args)
