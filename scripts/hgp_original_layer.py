"""MemBase bridge to the four original HGP storage implementations."""

import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock

from membase.layers.base import MemBaseLayer
from membase.model_types.memory import MemoryEntry
from rank_bm25 import BM25Okapi

from hgp3_router import AdaptiveRouter, deepseek_route, local_encoder
from hgp_neo4j_layer import HGPConfig, id_for, settings
from train_hgp3_2200 import ROOT


HGP_ROOT = ROOT / 'HGP-'
ENV_LOCK = Lock()
ENV_IDENTITY = None


def working_tokens(text):
    tokens = []
    for part in re.findall(r'[\u3400-\u9fff]+|[a-z0-9]+', text.lower()):
        if len(part) > 1 and '\u3400' <= part[0] <= '\u9fff':
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def build_hgp_qa_context(memories):
    if str(HGP_ROOT) not in sys.path:
        sys.path.insert(0, str(HGP_ROOT))
    from memory import SearchResult

    results = SearchResult()
    working_history = []
    for entry in memories:
        meta = entry.metadata
        kind = meta.get('kind')
        if kind == 'semantic':
            relation, separator, entity = entry.content.partition(': ')
            item = dict(relation=relation, entity=entity) if separator else dict(content=entry.content)
            results.semantic.append(dict(item, similarity=meta.get('score')))
        elif kind == 'episodic':
            timestamp, separator, text = entry.content.partition('] ')
            if not separator or not timestamp.startswith('['):
                timestamp, text = '', entry.content
            results.episodic.append(dict(timestamp=timestamp.lstrip('['), content=text, score=meta.get('score')))
        elif kind == 'procedural':
            rule = meta.get('rule')
            if rule is None:
                title, has_conditions, tail = entry.content.partition(': IF ')
                conditions, has_steps, steps = tail.partition(' THEN ')
                rule = dict(title=title, if_conditions=conditions.split('; ') if conditions else [],
                            then_steps=steps.split(' → ') if steps else []) if has_conditions and has_steps else entry.content
            results.procedural.append(dict(rule=rule))
        elif kind == 'working':
            if 'state_id' in meta:
                working_history.append(dict(timestamp=meta['timestamp'], text=entry.content))
            else:
                results.working = dict(state_name=entry.content[:50], content=dict(focus=entry.content))

    history = None if results.working and not working_history else working_history
    return results.to_context(max_items=len(memories), working_history=history)


class CachedStorageEmbeddings:
    def __init__(self, model_path):
        self.model, self.lock = local_encoder(model_path)
        if self.model.get_sentence_embedding_dimension() != 512:
            raise ValueError('Original HGP storage needs its 512-dimensional embedding model')

    @lru_cache(maxsize=8192)
    def _one(self, text):
        with self.lock:
            return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self._one(texts)
        if len(texts) == 1:
            return self._one(texts[0])[None, :]
        with self.lock:
            return self.model.encode(texts, **kwargs)


@lru_cache(maxsize=2)
def storage_embedder(model_path):
    return CachedStorageEmbeddings(model_path)


def restrict_relation_types(store_class):
    original = store_class.add_triplet
    if getattr(original, '_hgp_validates_relation', False):
        return

    def checked(store, user_id, relation, *args, **kwargs):
        relation = relation.strip().upper().replace(' ', '_') if isinstance(relation, str) else ''
        if not re.fullmatch(r'[A-Z_][A-Z0-9_]{0,95}', relation):
            return {'action': 'skip', 'reason': 'Invalid relationship type'}
        return original(store, user_id, relation, *args, **kwargs)

    checked._hgp_validates_relation = True
    store_class.add_triplet = checked


class OriginalHGPConfig(HGPConfig):
    neo4j_uri: str = 'neo4j://127.0.0.1:7687'
    semantic_database: str = 'semantic'
    episodic_database: str = 'episodic'
    allow_llm_extraction: bool = False
    storage_embedding: Path = ROOT / 'models/hgp-distiluse-base-multilingual-cased-v2'

    def get_llm_models(self):
        return ['deepseek-flash'] if self.allow_llm_fallback or self.allow_llm_extraction else []


def configure_original(config, *, search_only=False):
    global ENV_IDENTITY

    if not (HGP_ROOT / 'memory/__init__.py').is_file():
        raise FileNotFoundError('Restore the original HGP memory package in HGP-/memory')
    values = settings(config.env_file)
    user = os.environ.get('NEO4J_USERNAME') or values.get('NEO4J_USERNAME')
    password = os.environ.get('NEO4J_PASSWORD') or values.get('NEO4J_PASSWORD')
    if not user or not password:
        raise ValueError('Set NEO4J_USERNAME and NEO4J_PASSWORD')
    databases = (config.semantic_database, config.episodic_database)
    if databases[0] == databases[1] or any(not re.fullmatch(r'[a-z][a-z0-9.-]{2,62}', d) for d in databases):
        raise ValueError('Provide two distinct valid Neo4j database names')
    fixed = dict(NEO4J_URI=config.neo4j_uri, NEO4J_USER=user, NEO4J_PASSWORD=password,
                 NEO4J_SEMANTIC_DATABASE=databases[0], NEO4J_EPISODIC_DATABASE=databases[1])
    key = (os.environ.get('DEEPSEEK_API_KEY') or values.get('DEEPSEEK_API_KEY') or '').strip()
    if search_only:
        if not key:
            raise RuntimeError('DeepSeek API key unavailable')
        fixed['DEEPSEEK_API_KEY'] = key
    if not search_only:
        fixed.update(DEEPSEEK_MODEL='deepseek-flash', DEEPSEEK_LLM_MODEL='deepseek-flash',
                     DEEPSEEK_API_BASE='https://api.deepseek.com', DEEPSEEK_BASE_URL='https://api.deepseek.com',
                     OPENAI_BASE_URL='https://api.deepseek.com')
        if key:
            fixed.update(DEEPSEEK_API_KEY=key, OPENAI_API_KEY=key)
    with ENV_LOCK:
        if ENV_IDENTITY is not None and ENV_IDENTITY != id_for(fixed):
            raise ValueError('Original HGP stores cannot switch databases within the same process')
        os.environ.update(fixed)
        if str(HGP_ROOT) not in sys.path:
            sys.path.insert(0, str(HGP_ROOT))
        ENV_IDENTITY = id_for(fixed)
    return config.neo4j_uri, user, password


def initialize_original(config):
    from neo4j import GraphDatabase

    uri, user, password = configure_original(config)
    with GraphDatabase.driver(uri, auth=(user, password), connection_timeout=5) as driver:
        with driver.session(database='system') as session:
            for db in (config.semantic_database, config.episodic_database):
                session.run('CREATE DATABASE $name IF NOT EXISTS WAIT 20 SECONDS', name=db).consume()
        for db in (config.semantic_database, config.episodic_database):
            with driver.session(database=db) as session:
                session.run('RETURN 1').consume()
    from memory.semantic_memory.semantic_store import SemanticStore
    from memory.episodic_memory.episode_store import Neo4jEpisodeGraph

    semantic = SemanticStore(uri, user, password, database=config.semantic_database)
    episodic = Neo4jEpisodeGraph(namespace='hgp-schema')
    try:
        if not episodic.is_connected():
            raise RuntimeError('The original episodic graph is not connected')
        episodic._init_constraints()
    finally:
        semantic.close()
        episodic.close()


class OriginalHGPLayer(MemBaseLayer):
    def __init__(self, config):
        self.config = config
        self.credentials = configure_original(config)
        self.path = Path(config.save_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.scope = 'hgp_' + id_for(config.graph_namespace, config.user_id)
        self.router = AdaptiveRouter(
            config.classifier_dir, config.training_data, self.path / 'adaptive-router.pt',
            deepseek_route if config.allow_llm_fallback else None,
            tau0=config.tau0, tau_min=config.tau_min, beta=config.beta,
            batch_size=config.update_batch_size, adaptive=config.adaptive_updates,
        )
        self.identity = id_for(self.scope, self.router.identity, str(config.storage_embedding.resolve()),
                               config.semantic_database, config.episodic_database,
                               config.tau0, config.tau_min, config.beta, config.adaptive_updates)
        self.ledger = sqlite3.connect(self.path / 'ingestion.sqlite3')
        with self.ledger:
            self.ledger.execute('CREATE TABLE IF NOT EXISTS settings (id TEXT PRIMARY KEY, value TEXT)')
            self.ledger.execute('CREATE TABLE IF NOT EXISTS turns (id TEXT PRIMARY KEY, evidence TEXT, route TEXT)')
            self.ledger.execute('CREATE TABLE IF NOT EXISTS writes (id TEXT, kind TEXT, PRIMARY KEY(id, kind))')
            self.ledger.execute('INSERT OR IGNORE INTO settings VALUES (?,?)', ('identity', self.identity))
            recorded = self.ledger.execute("SELECT value FROM settings WHERE id='identity'").fetchone()
            if recorded[0] != self.identity:
                raise ValueError('HGP experiment identity changed. Select a new save_dir and namespace.')
        self._stores = {}
        self._search_memory = None
        self._working_index = None
        self._query_extractor = None
        self._read_only_search = False

    @classmethod
    def open_for_search(cls, config):
        """Open an existing run for retrieval without loading the adaptive classifier."""
        layer = cls.__new__(cls)
        layer.config = config
        layer.path = Path(config.save_dir).resolve()
        layer.scope = 'hgp_' + id_for(config.graph_namespace, config.user_id)
        layer.credentials = configure_original(config, search_only=True)
        layer._stores = {}
        layer._search_memory = None
        layer._working_index = None
        layer._query_extractor = None
        layer._read_only_search = True
        layer.query_extraction = 'fallback'
        layer.ledger = sqlite3.connect((layer.path / 'ingestion.sqlite3').as_uri() + '?mode=ro', uri=True)
        try:
            if not layer.load_memory(config.user_id):
                raise ValueError('Experiment memory is not ready')
            from memory.core.llm_extractor import SchemaBasedExtractor
            from openai import DefaultHttpxClient

            def record_response(response):
                layer.query_extraction = 'succeeded' if response.is_success else 'fallback'

            extractor = SchemaBasedExtractor(api_key=os.environ['DEEPSEEK_API_KEY'],
                                             base_url='https://api.deepseek.com', model='deepseek-flash')
            default_client = extractor.client
            extractor.client = default_client.with_options(
                http_client=DefaultHttpxClient(event_hooks={'response': [record_response]}),
                timeout=18, max_retries=0,
            )
            default_client.close()
            layer._query_extractor = extractor
        except Exception:
            layer.close()
            raise
        return layer

    def _store(self, kind):
        if kind in self._stores:
            return self._stores[kind]
        if kind == 'working':
            from memory.working_memory import WorkingMemory

            store = WorkingMemory(task_name=self.scope, storage_dir=self.path / 'working')
        else:
            from memory.core.episodic_graph_builder import EntityEmbedder
            from memory.episodic_memory.episode_store import EmbeddingHelper as EpisodeEmbedder, EpisodeStore
            from memory.procedural_memory.store import EmbeddingHelper as RuleEmbedder, ProceduralMemoryStore

            model = storage_embedder(str(self.config.storage_embedding.resolve()))
            EntityEmbedder._model = EpisodeEmbedder._model = RuleEmbedder._model = model
            if kind == 'episodic':
                store = EpisodeStore(storage_path=self.path / 'episodes.jsonl', namespace=self.scope,
                                     use_neo4j=not self._read_only_search)
                if self._read_only_search:
                    store._llm_extractor = self._query_extractor
                    store._load_cache()
                else:
                    if not store._graph.is_connected():
                        raise RuntimeError('Original episodic Neo4j store is unavailable')
                    if not self.config.allow_llm_extraction:
                        store._llm_extractor = None
            elif kind == 'procedural':
                store = ProceduralMemoryStore(storage_path=self.path / 'procedural_rules.jsonl', namespace=self.scope)
            else:
                from memory.semantic_memory.semantic_store import SemanticStore

                with ENV_LOCK:
                    restrict_relation_types(SemanticStore)
                store = SemanticStore(*self.credentials, database=self.config.semantic_database)
        self._stores[kind] = store
        return store

    def _write(self, kind, text, metadata):
        if kind in ('semantic', 'procedural') and not self.config.allow_llm_extraction:
            raise RuntimeError(f'Original HGP {kind} storage requires LLM extraction to be enabled')
        store = self._store(kind)
        source = metadata['source_message_id']
        if kind == 'episodic':
            from memory.episodic_memory.episode_store import Episode

            id = id_for(self.scope, source)
            previous = store.get_episode(id)
            if previous is None:
                store.add_episode(Episode(id=id, content=text, timestamp=metadata['timestamp'],
                                          actors=[metadata['name']], metadata=metadata))
            elif not self._has_episode(store, id):
                store._graph.add_episode(previous)
            if not self._has_episode(store, id):
                raise RuntimeError('Original HGP episode was not stored in Neo4j')
        elif kind == 'semantic':
            from memory.semantic_memory import store_semantic_profile

            store_semantic_profile(text, user_id=self.config.user_id, namespace=self.scope,
                                   evidence_episode_ids=[source])
        elif kind == 'procedural':
            from memory.procedural_memory import procedural_memory

            procedural_memory.llm_client.model = 'deepseek-flash'
            store.add_rules(procedural_memory.extract_procedure_rules(text, meta=metadata))
        else:
            content = dict(text=text, **metadata)
            timestamp = datetime.fromisoformat(metadata['timestamp']).isoformat()
            if store.get_current_state():
                store.transition_to(text[:50], content, episode_id=source, timestamp=timestamp)
            else:
                store.start_state(text[:50], content, episode_id=source, timestamp=timestamp)
            self._working_index = None

    def _has_episode(self, store, id):
        with store._graph._get_driver().session(database=self.config.episodic_database) as session:
            return session.run(
                'MATCH (e:Episode {episode_id:$id, namespace:$scope}) RETURN e.episode_id',
                id=id, scope=self.scope,
            ).single() is not None

    def add_message(self, message, **kwargs):
        text = message.content.strip()
        if not text:
            return
        evidence = id_for(text, message.role, message.name, message.timestamp)
        prior = self.ledger.execute('SELECT evidence, route FROM turns WHERE id=?', (message.id,)).fetchone()
        turn = dict(id=message.id, text=text, role=message.role)
        if prior:
            if prior[0] != evidence:
                raise ValueError('Source message ID reused with different content')
            result, vector = json.loads(prior[1]), None
        else:
            result, vector = self.router.route(turn)
            with self.ledger:
                self.ledger.execute('INSERT INTO turns VALUES (?,?,?)', (message.id, evidence, json.dumps(result)))
        kinds = [kind for kind in (result['label'], 'working' if result['working_state'] else None) if kind]
        metadata = dict(source_message_id=message.id, timestamp=message.timestamp,
                        role=message.role, name=message.name)
        for kind in kinds:
            if self.ledger.execute('SELECT 1 FROM writes WHERE id=? AND kind=?', (message.id, kind)).fetchone():
                continue
            self._write(kind, text, metadata)
            with self.ledger:
                self.ledger.execute('INSERT INTO writes VALUES (?,?)', (message.id, kind))
        if vector is not None:
            self.router.observe(turn, result, vector)

    def add_messages(self, messages, **kwargs):
        for message in messages:
            self.add_message(message, **kwargs)

    def _search_working(self, query, k):
        if self._working_index is None:
            states = self._store('working').get_execution_path()
            documents = [working_tokens(f"{state['timestamp']} {state['content'].get('text', '')}")
                         for state in states]
            vocabulary = [set(words) for words in documents]
            self._working_index = (list(zip(states, vocabulary, strict=True)),
                                   BM25Okapi(documents) if any(documents) else None)

        states, index = self._working_index
        terms = dict.fromkeys(working_tokens(query))
        if not terms or index is None:
            return []
        scores = index.get_scores(list(terms))
        matches = [(state, float(score)) for (state, words), score in zip(states, scores, strict=True)
                   if not words.isdisjoint(terms) and state['content'].get('text')]
        return sorted(matches, key=lambda row: (row[1], row[0]['timestamp']), reverse=True)[:k]

    def search_matches(self, query, k):
        if k < 1:
            raise ValueError('k must be positive')

        if self._search_memory is None:
            from memory import Memory, MemoryConfig
            from memory.semantic_memory.semantic_store import _get_store

            search_memory = Memory(user_id=self.config.user_id, config=MemoryConfig(enable_working=False))
            search_memory.namespace = self.scope
            search_memory._episodic_store = self._store('episodic')
            search_memory._procedural_store = self._store('procedural')
            with ENV_LOCK:
                _get_store()
            self._search_memory = search_memory

        return self._search_memory.search(query, top_k=k), self._search_working(query, k)

    def retrieve(self, query, k=10, **kwargs):
        result, working = self.search_matches(query, k)
        return self.entries_for_matches(result, working)

    def entries_for_matches(self, result, working):
        from memory import format_procedural_rule

        entries = []

        def append(kind, text, source, *, formatted_content=None, **metadata):
            entries.append(MemoryEntry(content=text, formatted_content=f'[{kind}] {formatted_content or text}',
                                       metadata=dict(kind=kind, kinds=[kind], source_id=source, **metadata)))

        for triple in result.semantic:
            evidence_ids = triple.get('evidence_ids') or []
            append('semantic', f'{triple["relation"]}: {triple["entity"]}',
                   evidence_ids[0] if evidence_ids else None,
                   evidence_ids=evidence_ids, score=triple.get('similarity'))
        for item in result.episodic:
            episode = self._store('episodic').get_episode(item['episode_id'])
            source = episode.metadata.get('source_message_id') if episode else None
            append('episodic', f'[{item["timestamp"]}] {item["content"]}', source,
                   score=float(item['score']), episode_id=item['episode_id'])
        for number, item in enumerate(result.procedural, 1):
            rule = item['rule']
            conditions = '; '.join(rule['if_conditions'])
            steps = ' → '.join(rule['then_steps'])
            heading = f'{rule["title"]}: IF {conditions} THEN'
            context = format_procedural_rule(rule, number)
            qa_rule = {field: rule[field] for field in ('title', 'if_conditions', 'then_steps')}
            append('procedural', f'{heading} {steps}', rule['meta'].get('source_message_id'),
                   formatted_content=context, score=float(item['score']), rule=qa_rule)

        for state, score in working:
            content = state['content']
            append('working', content['text'], content.get('source_message_id'),
                   state_id=state['state_id'], timestamp=state['timestamp'], priority=state['priority'], score=score)
        return entries

    def delete(self, memory_id):
        raise NotImplementedError('Original HGP stores have separate deletion APIs')

    def update(self, memory_id, **kwargs):
        raise NotImplementedError('Original HGP stores evolve through their per-type write APIs')

    def save_memory(self):
        with self.ledger:
            self.ledger.execute("INSERT OR REPLACE INTO settings VALUES ('ready','true')")

    def load_memory(self, user_id=None):
        return (user_id in (None, self.config.user_id)
                and self.ledger.execute("SELECT 1 FROM settings WHERE id='ready'").fetchone() is not None)

    def close(self):
        self.ledger.close()
        if 'semantic' in self._stores:
            self._stores['semantic'].close()
        if 'episodic' in self._stores and self._stores['episodic']._graph:
            self._stores['episodic']._graph.close()
        if self._query_extractor is not None:
            self._query_extractor.client.close()


def register_original():
    from membase.configs import CONFIG_MAPPING
    from membase.layers import MEMORY_LAYERS_MAPPING

    CONFIG_MAPPING.register('HGP', OriginalHGPConfig)
    MEMORY_LAYERS_MAPPING.register('HGP', OriginalHGPLayer)
