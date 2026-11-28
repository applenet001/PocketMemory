"""Persist Companion shares through the same HGP pipeline as MobileMem v2."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from threading import Lock
from types import SimpleNamespace

from graph import LocalGraph
from memory_draft import memory_evidence, memory_labels


ROOT = Path(__file__).resolve().parents[1]
DATABASES = ('v2-episodic', 'v2-semantic')


def shortcut_fields(values):
    """Accept structured Todo evidence from JSON or a Shortcuts form."""
    todo_id = values.get('todo_id')
    completed = values.get('is_completed')
    if todo_id is None and completed is None:
        return {}
    source = values.get('source_app')
    if not isinstance(source, str) or source.strip().casefold() != 'todo':
        raise ValueError('Todo task fields require source_app=Todo')
    if not isinstance(todo_id, str) or not 0 < len(todo_id.strip()) <= 200:
        raise ValueError('todo_id must be a string of 1–200 characters')
    if isinstance(completed, str) and completed.lower() in ('true', 'false'):
        completed = completed.lower() == 'true'
    if type(completed) is not bool:
        raise ValueError('is_completed must be true or false')
    return {'todo_id': todo_id.strip(), 'is_completed': completed}


def replace_text(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


class CompanionV2:
    def __init__(self):
        self.lock = Lock()
        self.pipeline = None

    def initialize(self):
        if self.pipeline is not None:
            return
        sys.path.append(str(ROOT / 'scripts'))
        import run_hgp_v2_smoke as v2
        from neo4j import GraphDatabase
        from sentence_transformers import SentenceTransformer

        uri, user, password = v2.initialize_v2_graph(LocalGraph().connection_settings())
        storage_encoder = v2.load_storage_encoder()
        encoder = SentenceTransformer(str(ROOT / 'models/qwen3-embedding-0.6b'), device='cpu', local_files_only=True)
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.storage_encoder = storage_encoder
        self.encoder = encoder
        self.pipeline = v2

    @staticmethod
    def namespace(target, user_id):
        digest = sha256(str(target.resolve()).encode()).hexdigest()[:16]
        return f'v2_companion_{digest}_{user_id}'

    def ingest(self, target, user_id, item, checkpoint):
        state = item['v2_state']
        text = memory_evidence(item).strip()
        todo = state.get('todo', {})
        if not text and not todo:
            state['complete'] = True
            return {'status': 'skipped', 'model': 'hgp-v2/four-head-500'}
        with self.lock:
            self.initialize()
            from run_hgp_membase import scoped_openai_usage

            target.mkdir(parents=True, exist_ok=True)
            with scoped_openai_usage('hgp-v2-companion', target / 'deepseek-usage.jsonl', upload_id=item['id']):
                return self.write(target, self.namespace(target, user_id), user_id, item, text, checkpoint)

    def write(self, target, namespace, user_id, item, text, checkpoint):
        from memory import Memory, MemoryConfig
        from memory.episodic_memory.episode_store import EpisodeStore
        from memory.procedural_memory.store import ProceduralMemoryStore
        from memory.working_memory import WorkingMemory
        from train.main import llm_client

        v2 = self.pipeline
        state = item['v2_state']
        source_app = item.get('source_app') or ''
        todo = state.get('todo', {})
        if todo:
            source_app = 'Todo'
        session_id = sha256(source_app.casefold().encode()).hexdigest()[:16]
        metadata = dict(source_app=source_app, message_id=item['id'], session_id=session_id,
                        timestamp=item['created'], session_date_time=v2.event_time(item['created']).isoformat(),
                        companion_upload=True, **todo)
        classifier = v2.Classifier(ROOT / 'outputs/four-head-500', self.storage_encoder, target, llm_client,
                                   encoder=self.encoder)
        buffer = target / 'adaptive_pending.json'
        if buffer.exists():
            classifier.pending = json.loads(buffer.read_text(encoding='utf-8'))
        if 'prediction' not in state:
            if item['feedback']:
                state['prediction'] = {'labels': memory_labels(item['feedback']), 'source': 'user'}
            elif text:
                classifier.prepare([text])
                state['prediction'] = classifier.infer(text)
                if classifier.inference['queued']:
                    replace_text(buffer, json.dumps(classifier.pending, ensure_ascii=False))
                if classifier.inference['source'] == 'llm' and not classifier.inference['queued']:
                    del state['prediction']
                    raise RuntimeError('HGP fallback did not return valid labels')
            else:
                state['prediction'] = {'labels': [], 'source': 'classifier'}
            checkpoint(state)
        classifier.update_if_ready()
        if buffer.exists():
            replace_text(buffer, json.dumps(classifier.pending, ensure_ascii=False))

        prediction = state['prediction']
        labels = prediction['labels']
        types = state.setdefault('types', [])
        remaining = [label for label in labels if label not in types]
        tracks = {}

        class UploadEpisodeStore(EpisodeStore):
            def add_episode(store, episode, link_to_previous=True):
                episode.id = item['id']
                store._load_cache()
                existing = store.get_episode(episode.id)
                if existing is None:
                    return super().add_episode(episode, link_to_previous)
                # If a previous attempt wrote the local record first, finish its graph write.
                if not self.episode_exists(namespace, existing.id):
                    store._graph.add_episode(existing)
                    if existing.prev_episode_id:
                        store._graph.link_episodes(existing.prev_episode_id, existing.id)
                return existing.id

        class UploadWorkingMemory(WorkingMemory):
            def start_state(track, state_name, content, **kwargs):
                return super().start_state(state_name, {**content, **({'todo_id': todo['todo_id']} if todo else {})},
                                           **{**kwargs, 'episode_id': item['id']})

            def transition_to(track, next_state_name, content, **kwargs):
                return super().transition_to(next_state_name, {**content, **({'todo_id': todo['todo_id']} if todo else {})},
                                             **{**kwargs, 'episode_id': item['id']})

        def track(task_name):
            session = task_name.removeprefix('session_')
            if session not in tracks:
                tracks[session] = UploadWorkingMemory(task_name, target / 'v2-working' / session)
            return tracks[session]

        rules = ProceduralMemoryStore(storage_path=target / 'v2-procedural/procedural_rules.jsonl', namespace=namespace)
        if remaining:
            working = track(f'session_{session_id}')
            existing_state = next((node for node in working.trajectory.nodes.values()
                                   if node.source_episode_id == item['id']), None)
            episodes = UploadEpisodeStore(storage_path=target / 'episodes.jsonl', namespace=namespace)
            try:
                if not episodes._graph.is_connected():
                    raise RuntimeError('The v2 episodic graph is unavailable')
                memories = Memory(user_id=user_id, session_id=session_id,
                                  config=MemoryConfig(enable_working=existing_state is None, auto_extract_async=False,
                                                      working_uses_event_time=True))
                memories.namespace = namespace
                memories._classifier = SimpleNamespace(infer=lambda _: {**prediction, 'labels': remaining})
                if 'episodic' in remaining and not self.episode_exists(namespace, item['id']):
                    episodes._load_cache()
                    memories._episode_count_since_last_extract = len(episodes._episode_cache) % memories.config.auto_extract_threshold
                with v2.bind_v2_stores(episodes, rules, track):
                    result = memories.add(text, metadata=metadata)
                stored = [kind for kind in result.memory_types
                          if kind != 'episodic' or self.episode_exists(namespace, item['id'])]
                if existing_state and 'episodic' in stored:
                    stored.append('working')
                working_update = result.details.get('working', {})
                state['working_node'] = existing_state.node_id if existing_state else working_update.get('node_id')
                types.extend(kind for kind in stored if kind not in types)
                checkpoint(state)
                if not all(label in types for label in remaining):
                    raise RuntimeError('HGP did not finish writing every routed memory type')
            finally:
                episodes._graph.close()
        if todo:
            if todo['is_completed']:
                for path in (target / 'v2-working').glob('*/state_trajectory.json'):
                    track(f'session_{path.parent.name}')
            node_id = state.get('working_node')
            if node_id:
                track(f'session_{session_id}')
            sessions = {todo['todo_id']: {session for session, memory in tracks.items()
                        if any(node.content.get('todo_id') == todo['todo_id']
                               for node in memory.trajectory.nodes.values())}}
            archived = v2.update_todo_working(metadata, {'success': bool(node_id), 'node_id': node_id},
                                             tracks, sessions, episode_id=item['id'] if 'episodic' in types else None)
            state.setdefault('working_archived', []).extend(archived)
        state['complete'] = True
        return dict(status='classified' if text else 'skipped', model='hgp-v2/four-head-500',
                    label=labels[0] if labels else None, labels=labels, working_state='working' in types,
                    source=prediction['source'], confidence=prediction.get('max_confidence'),
                    threshold=prediction.get('threshold'), uncertain=False)

    def episode_exists(self, namespace, identity):
        with self.driver.session(database=DATABASES[0]) as session:
            return session.run('MATCH (e:Episode {namespace:$namespace, episode_id:$id}) RETURN count(e) AS count',
                               namespace=namespace, id=identity).single()['count'] > 0

    def delete(self, target, user_id, item):
        state = item['v2_state']
        if 'prediction' not in state or not state['prediction']['labels']:
            return
        with self.lock:
            self.initialize()
            namespace = self.namespace(target, user_id)
            identity = item['id']
            with self.driver.session(database=DATABASES[0]) as session:
                session.run('MATCH (e:Episode {namespace:$namespace, episode_id:$id}) '
                            'OPTIONAL MATCH (e)-[]-(entity) WHERE entity.namespace=$namespace '
                            'WITH e, collect(DISTINCT entity) AS entities DETACH DELETE e '
                            'WITH entities UNWIND entities AS node '
                            'WITH node WHERE NOT node:Episode AND NOT (node)--() DELETE node',
                            namespace=namespace, id=identity).consume()
            with self.driver.session(database=DATABASES[1]) as session:
                session.run('MATCH (:User {namespace:$namespace})-[r]->(e:Entity {namespace:$namespace}) '
                            'WHERE $id IN r.evidence_episode_ids '
                            'WITH r, e, [x IN r.evidence_episode_ids WHERE x <> $id] AS other '
                            'FOREACH (_ IN CASE WHEN other = [] THEN [1] ELSE [] END | DELETE r) '
                            'FOREACH (_ IN CASE WHEN other <> [] THEN [1] ELSE [] END | SET r.evidence_episode_ids=other) '
                            'WITH DISTINCT e WHERE NOT (e)--() DELETE e',
                            namespace=namespace, id=identity).consume()
            for path, keep in (
                (target / 'episodes.jsonl', lambda row: row['id'] != identity),
                (target / 'v2-procedural/procedural_rules.jsonl',
                 lambda row: (row.get('meta') or {}).get('message_id') != identity),
            ):
                if path.exists():
                    rows = self.pipeline.read_jsonl(path)
                    replace_text(path, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows if keep(row)))
            for path in (target / 'v2-working').glob('*/state_trajectory.json'):
                data = json.loads(path.read_text(encoding='utf-8'))
                for group in data.values():
                    group['nodes'] = [node for node in group['nodes'] if node['source_episode_id'] != identity]
                    ids = {node['node_id'] for node in group['nodes']}
                    group['transitions'] = [edge for edge in group['transitions']
                                            if edge['from_node_id'] in ids and edge['to_node_id'] in ids]
                replace_text(path, json.dumps(data, ensure_ascii=False))

    def close(self):
        with self.lock:
            if self.pipeline is not None:
                self.driver.close()
