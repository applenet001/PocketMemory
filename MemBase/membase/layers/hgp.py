"""MemBase adapter for the repository's HGP stores and Qwen classifier."""
import hashlib
import json
import os
from pathlib import Path
import sys

from .base import MemBaseLayer
from ..model_types.memory import MemoryEntry

ROOT = Path(__file__).resolve().parents[3]


def configure(config):
    from dotenv import load_dotenv
    load_dotenv(config.env_file, override=False)
    os.environ['NEO4J_USER'] = os.getenv('NEO4J_USER') or os.environ['NEO4J_USERNAME']
    os.environ['NEO4J_SEMANTIC_DATABASE'] = config.semantic_database
    os.environ['NEO4J_EPISODIC_DATABASE'] = config.episodic_database
    os.environ['DEEPSEEK_LLM_MODEL'] = os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')
    sys.path.insert(0, str(ROOT / 'HGP-')) if str(ROOT / 'HGP-') not in sys.path else None
    import torch
    torch.set_num_threads(4)
    from train.qwen_router import classifier
    model = classifier(str(Path(config.checkpoint).resolve()), 'cpu')

    class Embeddings:
        def encode(self, texts, **kwargs):
            vectors = model.encode(texts)
            return vectors[0] if isinstance(texts, str) else vectors

    from memory.core.episodic_graph_builder import EntityEmbedder
    from memory.episodic_memory.episode_store import EmbeddingHelper as EpisodeEmbeddings
    from memory.procedural_memory.store import EmbeddingHelper as RuleEmbeddings
    EntityEmbedder._model = EpisodeEmbeddings._model = RuleEmbeddings._model = Embeddings()
    from neo4j import GraphDatabase
    with GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])) as driver:
        for database in (config.semantic_database, config.episodic_database):
            with driver.session(database=database) as session:
                session.run('RETURN 1').consume()
                for row in session.run('SHOW VECTOR INDEXES YIELD options RETURN options'):
                    dimension = row['options'].get('indexConfig', {}).get('vector.dimensions')
                    if dimension not in (None, 1024):
                        raise ValueError(f'{database} has incompatible vector dimension {dimension}')
    return model


class HGPLayer(MemBaseLayer):
    layer_type = 'HGP'

    def __init__(self, config):
        self.config = config
        self.model = configure(config)
        from memory import Memory, MemoryConfig
        self.folder = Path(config.save_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        scope = 'membase_' + hashlib.sha256(f'{config.run_id}:{config.user_id}'.encode()).hexdigest()[:24]
        self.memory = Memory(user_id=config.user_id, namespace=scope, config=MemoryConfig(
            storage_dir=self.folder, classifier_backend='qwen', classifier_checkpoint=Path(config.checkpoint),
            classifier_threshold=config.classifier_threshold, auto_extract_enabled=False))
        self.sessions = set()
        self.seen = set()
        log = self.folder / 'construction.jsonl'
        if log.exists():
            for line in log.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if row['success']:
                    self.seen.add(row['message_id'])
                    self.sessions.add(row['session_id'])

    def add_message(self, message, **kwargs):
        if message.id in self.seen:
            return
        session = message.metadata.get('session_id', 'default')
        meta = dict(source_message_id=message.id, session_id=session, timestamp=message.timestamp,
                    time_mode='historical', name=message.name, role=message.role)
        # Deliberately exclude side_note, person profiles, QA and gold evidence content.
        if not message.content.strip():
            from memory import MemoryResult
            result = MemoryResult(success=True, details={'classification': {
                'labels': [], 'decision': 'none', 'source': 'empty_input', 'used_llm': False}})
        else:
            result = self.memory.add(message.content, metadata=meta)
        errors = {k: v for k, v in result.details.items() if isinstance(v, dict) and v.get('success') is False}
        success = result.success and not errors
        row = dict(message_id=message.id, session_id=session, timestamp=message.timestamp,
                   role=message.role, name=message.name, success=success,
                   memory_types=result.memory_types, details=result.details)
        with (self.folder / 'construction.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if not success:
            raise RuntimeError(f'HGP failed for {message.id}: {errors or result.details}')
        self.sessions.add(session)
        self.seen.add(message.id)

    def add_messages(self, messages, **kwargs):
        for message in messages:
            self.add_message(message, **kwargs)

    def update(self, *args, **kwargs):
        raise NotImplementedError('HGP benchmark adapter supports ingestion and retrieval')

    def delete(self, *args, **kwargs):
        raise NotImplementedError('Use a new run directory to isolate benchmark runs')

    def retrieve(self, query, k=10, **kwargs):
        import numpy as np
        result = self.memory.search(query, top_k=k, semantic_mode='profile',
                                    memory_types=['episodic', 'semantic', 'procedural'])
        items = []
        for kind in ('episodic', 'semantic', 'procedural'):
            for item in getattr(result, kind):
                if kind == 'procedural':
                    # Stored vectors are internal data, not text to re-encode or send to QA.
                    item = {**item, 'rule': {key: value for key, value in item['rule'].items()
                                            if key != 'embedding'}}
                content = json.dumps(item, ensure_ascii=False)
                items.append(MemoryEntry(content=content, formatted_content=f'[{kind}] {content}', metadata={'memory_type': kind}))
        for session in sorted(self.sessions):
            wm = self.memory._get_working_memory(session)
            for node in wm.trajectory.nodes.values():
                if node.is_active:
                    content = json.dumps(node.content, ensure_ascii=False)
                    items.append(MemoryEntry(content=content, formatted_content=f'[working] {content}',
                                             metadata={'memory_type': 'working', 'session_id': session}))
        unique = {item.formatted_content: item for item in items}
        items = list(unique.values())
        if not items:
            return []
        vectors = self.model.encode([item.content for item in items])
        query_vector = self.model.encode([query], query=True)[0]
        scores = vectors @ query_vector
        for item, score in zip(items, scores):
            item.metadata['score'] = float(score)
        return [items[i] for i in np.argsort(-scores)[:k]]

    def save_memory(self, **kwargs):
        (self.folder / 'hgp_manifest.json').write_text(json.dumps(dict(
            namespace=self.memory.namespace, sessions=sorted(self.sessions),
            message_ids=sorted(self.seen), checkpoint=self.config.checkpoint), indent=2), encoding='utf-8')

    def load_memory(self, user_id=None):
        path = self.folder / 'hgp_manifest.json'
        if not path.exists():
            return False
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['namespace'] != self.memory.namespace:
            raise ValueError('HGP namespace mismatch')
        self.sessions = set(state['sessions'])
        self.seen = set(state['message_ids'])
        return True

    def cleanup(self):
        store = self.memory._episodic_store
        if store is not None and store._graph:
            store._graph.close()
