"""Lightweight three-way HGP router: frozen local encoder + softmax linear head.

Data comes from prepare_hgp3.py and label_hgp3.py: TriMEM (600), PerLTQA zh
(900), proScript (600), IMCS21 (300), MobileMem (600). See hgp3_data.py for
native vs mapped supervision. Only MobileMem is DeepSeek-labeled; 20% stratified
human audit plus uncertain cases. Working memory is not a competing class.
Metrics on externally mapped labels are proxy agreement, not human-gold accuracy.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from hgp3_data import ROOT, LABELS, load_training, model_input, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'outputs/hgp3-3000/data.jsonl')
    p.add_argument('--model', type=Path, default=ROOT / 'models/qwen3-embedding-0.6b')
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/hgp3-model')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--encode-batch-size', type=int, default=4)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.encode_batch_size, a.threads) < 1:
        p.error('Training sizes must be positive')
    rows = load_training(a.data)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import classification_report, confusion_matrix
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformer(str(a.model), local_files_only=True, device='cpu')
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    texts = [model_input(r) for r in rows]
    overlong = [r['id'] for r, text in zip(rows, texts) if len(encoder.tokenizer.encode(text)) > encoder.max_seq_length]
    if overlong:
        raise ValueError(f'{len(overlong)} overlong memories; re-segment and review before training, no silent truncation: {overlong[:5]}')
    # No embedding cache reuse: avoid stale weights/text cache contamination.
    with torch.inference_mode():
        embeddings = encoder.encode(texts, batch_size=a.encode_batch_size,
                                    normalize_embeddings=True, show_progress_bar=True)
    x = torch.tensor(np.asarray(embeddings), dtype=torch.float32)
    y = torch.tensor([LABELS.index(r['label']) for r in rows], dtype=torch.long)
    idx = {s: torch.tensor([i for i, r in enumerate(rows) if r['split'] == s]) for s in ('train', 'val', 'test')}
    loader = DataLoader(TensorDataset(x[idx['train']], y[idx['train']]), batch_size=a.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(a.seed))
    counts = torch.bincount(y[idx['train']], minlength=3).float()
    weights = counts.sum() / (3 * counts)
    head = torch.nn.Linear(x.shape[1], len(LABELS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.005, weight_decay=0.01)
    best, best_state = float('inf'), None
    for epoch in range(a.epochs):
        head.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(bx), by, weight=weights)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.cross_entropy(head(x[idx['val']]), y[idx['val']]).item()
        if val_loss < best:
            best, best_state = val_loss, {k: v.detach().clone() for k, v in head.state_dict().items()}
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: validation loss={val_loss:.4f}', flush=True)
    head.load_state_dict(best_state)
    with torch.no_grad():
        probs = head(x).softmax(-1).numpy()
    truth = y.numpy()
    pred = probs.argmax(-1)

    def metrics(indices):
        indices = np.asarray(indices, dtype=int)
        if not len(indices):
            return {'n': 0}
        return dict(n=len(indices), report=classification_report(truth[indices], pred[indices], labels=[0, 1, 2],
                    target_names=LABELS, output_dict=True, zero_division=0),
                    confusion_matrix=confusion_matrix(truth[indices], pred[indices], labels=[0, 1, 2]).tolist())

    test = idx['test'].tolist()
    report = dict(test_proxy_agreement=metrics(test),
                  mobilemem_validation=metrics([i for i in idx['val'].tolist() if rows[i]['source'] == 'MobileMem']),
                  human_reviewed_test=metrics([i for i in test if rows[i].get('reviewed')]),
                  mobilemem_test=metrics([i for i in test if rows[i]['source'] == 'MobileMem']),
                  by_source={s: metrics([i for i in test if rows[i]['source'] == s]) for s in sorted({r['source'] for r in rows})},
                  split_sizes={s: len(v) for s, v in idx.items()},
                  sources=dict(Counter(r['source'] for r in rows)),
                  warning='MobileMem test may contain unaudited teacher labels; human_reviewed_test is a sampled subset, not a full gold test.')
    # Set a deferral threshold using validation only; expose coverage and agreement.
    validation = idx['val'].numpy()
    threshold = 1.0
    coverage = 0.0
    for t in np.linspace(0.34, 0.99, 66):
        selected = validation[probs[validation].max(1) >= t]
        if len(selected) >= min(20, len(validation)) and (pred[selected] == truth[selected]).mean() >= 0.9:
            threshold, coverage = float(t), len(selected) / len(validation)
            break
    torch.save(head.state_dict(), a.out / 'head.pt')
    config = dict(encoder=str(a.model.resolve()), labels=LABELS, architecture='frozen_encoder_linear_softmax',
                  embedding_dimension=x.shape[1], normalize_embeddings=True, max_seq_length=encoder.max_seq_length,
                  input_format='hgp3_data.model_input: optional context followed by target memory',
                  confidence_threshold=threshold, validation_coverage=coverage, seed=a.seed,
                  data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(), source_counts=report['sources'])
    for name, obj in [('config.json', config), ('metrics.json', report)]:
        (a.out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    write(a.out / 'test-predictions.jsonl', [dict(id=rows[i]['id'], label=rows[i]['label'],
          predicted=LABELS[pred[i]], probabilities=probs[i].tolist(), reviewed=rows[i].get('reviewed', False)) for i in test])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
