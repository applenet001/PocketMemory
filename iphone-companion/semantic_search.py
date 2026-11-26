"""Rank one inbox's records using the local Qwen encoder."""
from functools import lru_cache
import sys

from classifier import ENCODER, ROOT


MAX_QUERY_LENGTH = 300
CHUNK_LENGTH = 1000
CHUNK_OVERLAP = 120
MIN_SIMILARITY = 0.45
RESULT_LIMIT = 20


def _encoder():
    scripts = str(ROOT / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from hgp3_router import local_encoder

    return local_encoder(str(ENCODER))


@lru_cache(maxsize=128)
def _record_vectors(heading, body):
    step = CHUNK_LENGTH - CHUNK_OVERLAP
    chunks = [f'{heading}\n{body[start:start + CHUNK_LENGTH]}'.strip()
              for start in range(0, max(1, len(body) - CHUNK_OVERLAP), step)]
    encoder, lock = _encoder()
    with lock:
        return encoder.encode(chunks, batch_size=8, normalize_embeddings=True, show_progress_bar=False)


def search_meaning(items, query):
    if not items:
        return []
    encoder, lock = _encoder()
    prompt = 'Instruct: Given a question about personal memories, retrieve relevant saved passages.\nQuery: '
    with lock:
        query_vector = encoder.encode(prompt + query, normalize_embeddings=True, show_progress_bar=False)

    ranked = []
    for item in items:
        heading = '\n'.join(part for part in (item['title'], item['note']) if part)
        body = item['text']
        if heading or body:
            similarity = float((_record_vectors(heading, body) @ query_vector).max())
            if similarity >= MIN_SIMILARITY:
                ranked.append((similarity, item))
    ranked.sort(key=lambda result: result[0], reverse=True)
    return [item for _, item in ranked[:RESULT_LIMIT]]
