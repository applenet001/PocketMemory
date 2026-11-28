"""Replay one MobileMem trajectory through the original HGP memory pipeline."""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from deepseek_usage import summarize_construction_usage
from mobilemem_source import evidence_source_app, mobilemem_source_app
from run_hgp_membase import log_openai_usage


ROOT = Path(__file__).resolve().parents[1]
HGP = ROOT / 'HGP-'
V2 = ROOT / 'outputs/v2'
LABELS = ['episodic', 'semantic', 'procedural', 'none']
CHINA = timezone(timedelta(hours=8), 'Asia/Shanghai')


def event_time(value):
    time = datetime.fromisoformat(value)
    return time.replace(tzinfo=CHINA) if time.tzinfo is None else time.astimezone(CHINA)


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


class Classifier:
    """Adapt four-head-500 to Memory.add's original classifier result contract."""

    def __init__(self, directory, storage_encoder, target, fallback_client, *, encoder=None):
        import torch
        from sentence_transformers import SentenceTransformer
        from train.adaptive_threshold import AdaptiveThresholdManager

        self.config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        if self.config['labels'] != LABELS:
            raise ValueError('Unexpected four-head label order')
        checkpoint = torch.load(directory / 'head.pt', map_location='cpu', weights_only=True)
        self.head = torch.nn.Linear(self.config['embedding_dimension'], len(LABELS))
        self.head.load_state_dict(checkpoint)
        self.encoder = encoder if encoder is not None else SentenceTransformer(
            str(ROOT / 'models/qwen3-embedding-0.6b'), device='cpu', local_files_only=True)
        self.storage_encoder = storage_encoder
        self.threshold = AdaptiveThresholdManager(config_path=target / 'threshold_state.json')
        self.fallback_client = fallback_client
        self.target = target
        self.pending = []
        self.accepted = read_jsonl(target / 'adaptive_history.jsonl')
        iteration = self.threshold.state['iterations']
        if iteration:
            self.head.load_state_dict(torch.load(target / f'head-round-{iteration}.pt',
                                                 map_location='cpu', weights_only=True))
        self.inference = {}
        self.vectors = {}

    def prepare(self, messages):
        texts = [text for text in dict.fromkeys(messages) if text not in self.vectors]
        if not texts:
            return
        vectors = self.encoder.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False)
        self.vectors.update(zip(texts, vectors, strict=True))

    def infer(self, text):
        import torch
        from train.main import call_llm_for_classification

        vector = torch.as_tensor(self.vectors[text], dtype=torch.float32)
        with torch.no_grad():
            scores = dict(zip(LABELS, self.head(vector).sigmoid().tolist(), strict=True))
        hits = [label for label in LABELS if scores[label] >= self.config['thresholds'][label]]
        if 'none' in hits and len(hits) > 1:
            strongest = max(hits, key=scores.get)
            hits = ['none'] if strongest == 'none' else [label for label in hits if label != 'none']
        confidence = max(scores.values())
        threshold = self.threshold.get_current_threshold()
        used_llm = not hits or confidence < threshold
        labels = hits
        before = len(self.pending)
        response = None
        if used_llm:
            original = self.fallback_client.call_llm_with_json

            def capture(*args, **kwargs):
                nonlocal response
                response = original(*args, **kwargs)
                return response

            with patch.object(self.fallback_client, 'call_llm_with_json', capture):
                labels = call_llm_for_classification(text)
            if isinstance(response, dict):
                raw = response.get('labels')
                if (isinstance(raw, list) and raw and all(label in LABELS for label in raw)
                        and ('none' not in raw or raw == ['none'])):
                    self.pending.append((text, labels))

        self.inference = dict(labels=labels, local_labels=hits, scores=scores, source='llm' if used_llm else 'classifier',
                              max_confidence=confidence, threshold=threshold, queued=len(self.pending) > before)
        return {**self.inference, 'labels': [label for label in labels if label != 'none']}

    def update_if_ready(self):
        import torch
        from train.infer import semantic_dedup_new_samples
        from train.main import BUFFER_LIMIT

        if len(self.pending) < BUFFER_LIMIT:
            return
        history = self.target / 'adaptive_history.jsonl'
        candidates = [dict(text=text, labels=[int(label in labels) for label in LABELS]) for text, labels in self.pending]
        with patch.multiple('train.infer', ORIG_TRAIN_FILE=self.target / 'unused_original_training_data.jsonl', INCREMENTAL_TRAIN_FILE=history,
                            SentenceTransformer=lambda _: self.storage_encoder):
            accepted = semantic_dedup_new_samples(candidates)
        if not accepted:
            self.pending.clear()
            return
        batch = self.accepted + accepted
        self.prepare(row['text'] for row in batch)
        features = torch.as_tensor([self.vectors[row['text']].tolist() for row in batch], dtype=torch.float32)
        targets = torch.as_tensor([row['labels'] for row in batch], dtype=torch.float32)
        optimizer = torch.optim.SGD(self.head.parameters(), lr=0.1)
        for _ in range(20):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.head(features), targets)
            loss.backward()
            optimizer.step()
        iteration = self.threshold.state['iterations'] + 1
        torch.save(self.head.state_dict(), self.target / f'head-round-{iteration}.pt')
        with history.open('a', encoding='utf-8') as stream:
            for row in accepted:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.threshold.record_training_iteration(num_samples=len(accepted))
        self.accepted = batch
        self.pending.clear()


def configure_v2(graph_settings=None):
    from dotenv import dotenv_values

    config = dotenv_values(ROOT / 'config/local.env')
    uri, username, password = graph_settings or (config.get('NEO4J_URI') or 'neo4j://127.0.0.1:7687',
                                                 config['NEO4J_USERNAME'], config['NEO4J_PASSWORD'])
    os.environ.update(NEO4J_URI=uri, NEO4J_USER=username, NEO4J_PASSWORD=password,
                      NEO4J_EPISODIC_DATABASE='v2-episodic', NEO4J_SEMANTIC_DATABASE='v2-semantic',
                      DEEPSEEK_API_KEY=config['DEEPSEEK_API_KEY'], DEEPSEEK_BASE_URL='https://api.deepseek.com',
                      OPENAI_BASE_URL='https://api.deepseek.com')
    if str(HGP) not in sys.path:
        sys.path.insert(0, str(HGP))
    return uri, username, password


def load_storage_encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    from memory.core.episodic_graph_builder import EntityEmbedder
    from memory.episodic_memory.episode_store import EmbeddingHelper as EpisodeEmbedder
    from memory.procedural_memory.store import EmbeddingHelper as RuleEmbedder

    torch.set_num_threads(4)
    model_path = ROOT / 'models/hgp-distiluse-base-multilingual-cased-v2'
    encoder = SentenceTransformer(str(model_path), device='cpu', local_files_only=True)
    EntityEmbedder._model = EpisodeEmbedder._model = RuleEmbedder._model = encoder
    return encoder


def update_todo_working(row, working_update, tracks, sessions_by_todo, *, episode_id=None):
    """Retain Todo provenance and archive states for an explicitly completed task."""
    todo_id = row.get('todo_id')
    completed = row.get('is_completed')
    if row.get('source_app') != 'Todo' or not todo_id or type(completed) is not bool:
        return []
    sessions = sessions_by_todo.setdefault(todo_id, set())
    if working_update.get('success'):
        session = row['session_id']
        trajectory = tracks[session].trajectory
        node = trajectory.nodes[working_update['node_id']]
        node.content['todo_id'] = todo_id
        sessions.add(session)
        trajectory._save_node_to_file(node)
    if not completed:
        return []

    trigger = {'message_id': row['message_id'], 'event_timestamp': event_time(row['timestamp']).isoformat()}
    if episode_id:
        trigger['episode_id'] = episode_id
    archived = []
    for session in sorted(sessions):
        trajectory = tracks[session].trajectory
        for node_id, node in trajectory.nodes.items():
            if node.content.get('todo_id') == todo_id and node.is_active:
                node.content['archive_trigger'] = trigger
                node.is_active = False
                trajectory.archived_nodes[node_id] = node
                trajectory._save_node_to_file(node)
                archived.append(dict(session_id=session, state_id=node_id, todo_id=todo_id, archive_trigger=trigger))
    return archived


def initialize_v2_graph(graph_settings=None):
    from neo4j import GraphDatabase

    uri, username, password = configure_v2(graph_settings)
    graph = GraphDatabase.driver(uri, auth=(username, password))
    with graph:
        with graph.session(database='system') as connection:
            for database in ('v2-episodic', 'v2-semantic'):
                connection.run('CREATE DATABASE $name IF NOT EXISTS WAIT 20 SECONDS', name=database).consume()
        with graph.session(database='v2-episodic') as connection:
            connection.run('CREATE INDEX v2_episode_source_app IF NOT EXISTS '
                           'FOR (e:Episode) ON (e.namespace, e.source_app)').consume()
    return uri, username, password


@contextmanager
def bind_v2_stores(episodes, rules, track, *, replay=False):
    from memory import episodic_memory, procedural_memory, working_memory
    from memory.episodic_memory import episode_store, episodic_memory as episodic
    from memory.procedural_memory import procedural_memory as procedural

    with ExitStack() as stack:
        stack.enter_context(patch.object(episode_store, 'EpisodeStore', lambda **_: episodes))
        stack.enter_context(patch.object(episodic, '_get_store', lambda **_: episodes))
        stack.enter_context(patch.object(episodic_memory, '_get_store', lambda **_: episodes))
        stack.enter_context(patch.object(working_memory, 'WorkingMemory', track))
        stack.enter_context(patch.object(procedural, '_default_store', rules))
        stack.enter_context(patch.object(procedural_memory, '_default_store', rules))
        if replay:
            stack.enter_context(patch.dict(sys.modules, {'procedural_memory': procedural_memory}))
        yield


def run(rows, target):
    initialize_v2_graph()
    from memory import Memory, MemoryConfig, knowledge_evolution, semantic_memory
    from memory.episodic_memory import episode_store, episodic_memory as episodic
    from memory.procedural_memory.store import ProceduralMemoryStore
    from memory.working_memory import WorkingMemory
    from train.main import llm_client

    user = rows[0]['user_id']
    run_id = sha256(str(target.resolve()).encode()).hexdigest()[:12]
    namespace = f'v2_{run_id}_{user}'
    storage_encoder = load_storage_encoder()
    classifier = Classifier(ROOT / 'outputs/four-head-500', storage_encoder, target, llm_client)
    classifier.prepare(row['content'] for row in rows)
    print(f'Replaying {len(rows)} messages with {llm_client.model}', flush=True)

    episodes = episode_store.EpisodeStore(storage_path=target / 'episodes.jsonl', namespace=namespace)
    if not episodes._graph.is_connected():
        raise RuntimeError('The v2 episodic graph is unavailable')
    rules_path = target / 'v2-procedural/procedural_rules.jsonl'
    rules = ProceduralMemoryStore(storage_path=rules_path, namespace=namespace)
    memories = Memory(user_id=user, config=MemoryConfig(auto_extract_async=False, working_uses_event_time=True))
    memories.namespace = namespace
    memories._classifier = classifier
    tracks = {}
    sessions_by_todo = {}
    results = []

    def track(task_name):
        _, _, session_id = task_name.partition('session_')
        return WorkingMemory(task_name, target / 'v2-working' / session_id)

    def scoped_semantic(function, text=None, **kwargs):
        kwargs.update(user_id=user, namespace=namespace)
        return function(**kwargs) if text is None else function(text, **kwargs)

    class ReplayClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return event_time(rows[-1]['timestamp'])

    with bind_v2_stores(episodes, rules, track, replay=True), ExitStack() as stack:
        def bind(module, **values):
            stack.enter_context(patch.multiple(module, **values))

        query = partial(scoped_semantic, semantic_memory.query_semantic_profile)
        write = partial(scoped_semantic, semantic_memory.store_semantic_profile)
        bind(semantic_memory, query_semantic_profile=query, store_semantic_profile=write)

        with (target / 'turns.jsonl').open('w', encoding='utf-8') as log:
            for index, row in enumerate(rows, 1):
                session = row['session_id']
                memories.session_id = session
                row = dict(row)
                row.pop('source_app', None)
                source_app = mobilemem_source_app(row['role'], row['name'])
                if source_app:
                    row['source_app'] = source_app
                result = memories.add(row['content'], metadata={**row, 'session_date_time': event_time(row['timestamp']).isoformat()})
                record = {**row, **classifier.inference, 'memory_types': result.memory_types, 'details': result.details,
                          'success': result.success}
                if 'working' in result.memory_types:
                    tracks[session] = track(f'session_{session}')
                archived = update_todo_working(
                    row, result.details.get('working', {}), tracks, sessions_by_todo,
                    episode_id=result.details.get('episodic', {}).get('episode_id'),
                )
                if archived:
                    record['working_archived'] = archived
                log.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                log.flush()
                results.append(record)
                classifier.update_if_ready()
                print(f'[{index}/{len(rows)}] {classifier.inference["source"]}: {", ".join(result.memory_types) or "none"}', flush=True)

        procedure_extract = episodic.extract_to_procedural()
        bind(knowledge_evolution, datetime=ReplayClock)
        evolution = knowledge_evolution.run_knowledge_evolution(SimpleNamespace(episodes=episodes._episode_cache))
        semantic = query(limit=1000)
        source_by_id = {turn['message_id']: turn['source_app'] for turn in results if 'source_app' in turn}
        for triple in semantic:
            source_app = evidence_source_app(triple.get('evidence_ids') or [], source_by_id)
            if source_app:
                triple['source_app'] = source_app

    working = [dict(session_id=session, current=memory.get_current_state(),
                    states=[{**state, 'is_active': memory.trajectory.nodes[state['state_id']].is_active}
                            for state in memory.get_execution_path()], transitions=memory.get_state_transitions())
               for session, memory in tracks.items()]
    report = dict(user_id=user, namespace=namespace, input_count=len(rows), turns=results, semantic=semantic,
                  episodic=read_jsonl(episodes.storage_path), procedural=read_jsonl(rules_path), working=working,
                  procedural_extraction=procedure_extract, evolution=asdict(evolution),
                  adaptive=dict(iterations=classifier.threshold.state['iterations'], pending=len(classifier.pending),
                                threshold=classifier.threshold.get_current_threshold()))
    report['deepseek_construction'] = summarize_construction_usage(read_jsonl(target / 'deepseek-usage.jsonl'))
    write_json(target / 'report.json', report)
    episodes._graph.close()


def index_working_states(report):
    states = {(track['session_id'], state['state_id']): state
              for track in report['working'] for state in track['states']}
    working_by_episode = {}
    for turn in report['turns']:
        episode_id = turn['details'].get('episodic', {}).get('episode_id')
        update = turn['details'].get('working', {})
        if episode_id and update.get('success'):
            state = states.get((turn['session_id'], update['node_id']))
            if state:
                working_by_episode[episode_id] = dict(state_name=state['state_name'],
                                                      progress=state['content'].get('progress'))
    return working_by_episode


def attach_working_states(episodic, working_by_episode):
    return [{**episode, 'working_state': working_by_episode[episode['episode_id']]}
            if episode['episode_id'] in working_by_episode else episode
            for episode in episodic]


@contextmanager
def open_search(target):
    report = json.loads((target / 'report.json').read_text(encoding='utf-8'))
    configure_v2()
    from memory import Memory, MemoryConfig
    from memory.episodic_memory import EpisodeStore
    from memory.procedural_memory.store import ProceduralMemoryStore

    load_storage_encoder()
    memories = Memory(user_id=report['user_id'], config=MemoryConfig(enable_working=False))
    memories.namespace = report['namespace']
    episodes = EpisodeStore(storage_path=target / 'episodes.jsonl', namespace=memories.namespace)
    memories._episodic_store = episodes
    memories._procedural_store = ProceduralMemoryStore(
        storage_path=target / 'v2-procedural/procedural_rules.jsonl', namespace=memories.namespace)
    try:
        yield memories, report
    finally:
        episodes._graph.close()


def search(target, query, top_k):
    with open_search(target) as (memories, report):
        result = memories.search(query, top_k=top_k)
        result.episodic = attach_working_states(result.episodic, index_working_states(report))
        return result.to_context(query=query, max_items=top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=V2 / 'smoke20/messages.jsonl')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--search-run', type=Path, help='Read memories from a completed v2 run')
    parser.add_argument('--query', help='Search question for --search-run')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    if args.search_run is not None:
        if not args.query or not args.query.strip() or args.top_k < 1:
            parser.error('--search-run requires a nonempty --query and positive --top-k')
        target = args.search_run.resolve()
        with patch.dict(os.environ, {'HGP_DEEPSEEK_USAGE_LOG': str(target / 'deepseek-search-usage.jsonl')}):
            log_openai_usage('hgp-v2-search', record_missing=True, monitor_tokens=True)
            print(search(target, args.query, args.top_k))
        return
    if args.query is not None:
        parser.error('--query requires --search-run')
    rows = read_jsonl(args.source)[:args.limit]
    if args.limit < 1 or len(rows) != args.limit or len({row['user_id'] for row in rows}) != 1:
        parser.error('Provide enough messages from exactly one user')
    timestamps = [event_time(row['timestamp']) for row in rows]
    if timestamps != sorted(timestamps) or len({row['message_id'] for row in rows}) != len(rows):
        parser.error('Messages must be chronological and have unique IDs')
    run_root = V2 / f'smoke{args.limit}'
    target = run_root / uuid4().hex[:12]
    target.mkdir(parents=True)
    print(target, flush=True)
    with patch.dict(os.environ, {'HGP_DEEPSEEK_USAGE_LOG': str(target / 'deepseek-usage.jsonl')}):
        log_openai_usage('hgp-v2-smoke', record_missing=True, monitor_tokens=True)
        run(rows, target)


if __name__ == '__main__':
    main()
