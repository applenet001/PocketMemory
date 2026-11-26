"""HGP confidence routing and per-user incremental updates of the local HGP3 heads."""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from threading import Lock

from label_hgp3_2200 import PROMPT_PATH, annotate, api_key
from train_hgp3_2200 import ROUTES, load_rows, normalized


@lru_cache(maxsize=2)
def local_encoder(model):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model, local_files_only=True, device='cpu')
    encoder.eval()
    return encoder, Lock()


def deepseek_route(message):
    result = annotate([message], PROMPT_PATH.read_text(encoding='utf-8'), api_key())
    return dict(result['items'][0], usage=result['usage'])


class AdaptiveRouter:
    def __init__(self, classifier, data, state, fallback, tau0=0.9, tau_min=0.5, beta=0.05,
                 batch_size=100, adaptive=True, embed=None):
        import numpy as np
        import torch

        self.classifier, self.data = Path(classifier), Path(data)
        self.state_path = Path(state) if state is not None else None
        if adaptive and self.state_path is None:
            raise ValueError('Adaptive routing requires a state path')
        self.fallback = fallback
        self.tau0, self.tau_min, self.beta = tau0, tau_min, beta
        self.batch_size, self.adaptive = batch_size, adaptive
        if not (0 < tau_min <= tau0 <= 1 and beta >= 0 and batch_size > 0):
            raise ValueError('Invalid HGP threshold or update batch')
        self.config = json.loads((self.classifier / 'config.json').read_text(encoding='utf-8'))
        head = self.classifier / 'head.pt'
        self.identity = hashlib.sha256(head.read_bytes()).hexdigest()
        if self.config['route_labels'] != ROUTES:
            raise ValueError('Classifier route labels do not match HGP3')
        self.head = torch.load(head, map_location='cpu', weights_only=True)
        self.state = dict(head_sha256=self.identity, iteration=0, pending=[], history=[])
        if self.state_path is not None and self.state_path.exists():
            self.state = torch.load(self.state_path, map_location='cpu', weights_only=True)
            if self.state['head_sha256'] != self.identity:
                raise ValueError('Adaptive state belongs to a different base classifier')
            self.head = self.state.get('weights', self.head)
        self.embed = embed or self._encode
        self.base_vectors = None
        self.cached_inputs = {}
        if adaptive:
            rows = load_rows(self.data)
            data_hash = hashlib.sha256(self.data.read_bytes()).hexdigest()
            if self.config['data_sha256'] != data_hash:
                raise ValueError('Incremental replay data does not match base training')
            with np.load(self.classifier / 'embeddings.npz', allow_pickle=False) as archive:
                if not archive['key'].item().startswith(data_hash + ':'):
                    raise ValueError('Incremental replay embeddings do not match base training')
                train = [i for i, row in enumerate(rows) if row['split'] == 'train']
                vectors = archive['features']
                self.base_vectors = vectors[train]
                self.cached_inputs = {
                    row['text']: vectors[i] for i, row in enumerate(rows) if row['source'] == 'MobileMem'
                }
            self.base_labels = [ROUTES.index(rows[i]['label']) for i in train]
            self.base_work = [rows[i]['working_state'] for i in train]
            self.known_text = {normalized(row['text']) for row in rows}
            self.heldout_ids = {
                row['id'].removeprefix('MobileMem:') for row in rows
                if row['source'] == 'MobileMem' and row['split'] != 'train'
            }

    def _encode(self, text):
        if text in self.cached_inputs:
            return self.cached_inputs[text]
        encoder, lock = local_encoder(self.config['encoder'])
        with lock:
            return encoder.encode(text, normalize_embeddings=True, show_progress_bar=False)

    @property
    def threshold(self):
        return max(self.tau_min, self.tau0 * math.exp(-self.beta * self.state['iteration']))

    def vector(self, text):
        import numpy as np

        vector = np.asarray(self.embed(text), dtype=np.float32)
        if vector.shape != (self.config['embedding_dimension'],) or not np.isfinite(vector).all():
            raise ValueError('Invalid router embedding')
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError('Router embedding has zero norm')
        return vector / norm

    def route(self, message, *, allow_uncertain=False):
        from scipy.special import expit, softmax

        vector = self.vector(message['text'])
        logits = self.head['route.weight'].numpy() @ vector + self.head['route.bias'].numpy()
        probs = softmax(logits / self.config['route_temperature'])
        work_logit = (self.head['working.weight'].numpy() @ vector + self.head['working.bias'].numpy()).item()
        result = dict(label=ROUTES[int(probs.argmax())],
                      working_state=bool(expit(work_logit) >= self.config['working_threshold']),
                      source='classifier', confidence=float(probs.max()), threshold=self.threshold,
                      uncertain=False)
        if result['confidence'] < result['threshold']:
            if self.fallback is None:
                if allow_uncertain:
                    result['uncertain'] = True
                    return result, vector
                raise RuntimeError('Low-confidence turn requires LLM fallback. Enable it explicitly for this run.')
            fallback = self.fallback(message)
            if (fallback['label'] not in ROUTES or type(fallback['working_state']) is not bool
                    or type(fallback['uncertain']) is not bool):
                raise ValueError('Invalid HGP fallback annotation')
            result.update(label=fallback['label'], working_state=fallback['working_state'],
                          uncertain=fallback['uncertain'], source='llm', usage=fallback.get('usage', {}))
        return result, vector

    def _save(self, state):
        import torch

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(self.state_path)
        self.state = state

    def observe(self, message, result, vector):
        if (not self.adaptive or result['source'] != 'llm' or result['uncertain']
                or message['id'].removeprefix('MobileMem:') in self.heldout_ids):
            return
        signature = normalized(message['text'])
        examples = self.state['history'] + self.state['pending']
        if (signature in self.known_text or any(row['signature'] == signature for row in examples)
                or (self.base_vectors @ vector).max() >= 0.90
                or any(row['vector'].numpy() @ vector >= 0.90 for row in examples)):
            return
        import torch

        row = dict(signature=signature, vector=torch.from_numpy(vector.copy()),
                   label=ROUTES.index(result['label']), working_state=result['working_state'])
        self._save(dict(self.state, pending=self.state['pending'] + [row]))
        if len(self.state['pending']) >= self.batch_size:
            self.update()

    def update(self):
        import numpy as np
        import torch
        from sklearn.linear_model import LogisticRegression

        examples = self.state['history'] + self.state['pending']
        if not self.adaptive or not self.state['pending']:
            return
        x = np.vstack([self.base_vectors, *(r['vector'].numpy() for r in examples)])
        routes = self.base_labels + [r['label'] for r in examples]
        working = self.base_work + [r['working_state'] for r in examples]
        observed = [i for i, label in enumerate(working) if label is not None]
        weights = {}
        for name, features, targets in (
            ('route', x, routes), ('working', x[observed], [int(working[i]) for i in observed]),
        ):
            head = LogisticRegression(C=self.config[f'{name}_regularization_c'],
                                      class_weight=self.config[f'{name}_class_weight'], max_iter=1000)
            head.fit(features, targets)
            weights[f'{name}.weight'] = torch.as_tensor(head.coef_, dtype=torch.float32)
            weights[f'{name}.bias'] = torch.as_tensor(head.intercept_, dtype=torch.float32)
        self._save(dict(self.state, iteration=self.state['iteration'] + 1, weights=weights,
                        pending=[], history=examples))
        self.head = weights
