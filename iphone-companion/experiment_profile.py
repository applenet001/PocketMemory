"""Build a read-only profile from one evaluation trajectory."""
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Lock

from graph import LocalGraph
from local_memory import LocalMemories
from experiment_search import search_run

RUN = Path(__file__).resolve().parents[1] / 'outputs/mobilemem-qwen/hgp-history-time-qa'
DATASET = RUN.parents[2] / 'data/text/mobilemem_data.json'
RECENT_K = 5
GRAPH_SCOPE_PATTERN = re.compile(r'hgp_[0-9a-f]{64}')
_config_lock = Lock()


class ExperimentConfigUnavailable(RuntimeError):
    pass


def evaluated_users():
    if not RUN.is_dir():
        return []
    return sorted(path.name for path in RUN.iterdir()
                  if re.fullmatch(r'mobilemem-person_[A-Za-z0-9]+', path.name)
                  and path.is_dir() and not path.is_symlink()
                  and (path / 'episodes.jsonl').is_file())


@lru_cache(maxsize=1)
def _read_dataset(path, modified):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dataset_samples():
    try:
        return _read_dataset(DATASET, DATASET.stat().st_mtime_ns)
    except (OSError, ValueError):
        return []


@lru_cache(maxsize=16)
def _archived_display_name(path, modified):
    display_name = None
    with path.open(encoding='utf-8-sig') as archive:
        for line in archive:
            if not line.strip():
                continue
            metadata = json.loads(line).get('metadata') or {}
            name = metadata.get('name')
            if metadata.get('role') != 'user' or not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            if display_name is not None and name != display_name:
                return None
            display_name = name
    return display_name


def evaluated_user_names(users):
    allowed = set(users)
    names = {}
    for sample in dataset_samples():
        person = sample.get('person', {})
        identity = 'mobilemem-' + str(person.get('id', ''))
        name = person.get('personname') or person.get('name')
        if identity in allowed and isinstance(name, str) and name.strip():
            names[identity] = name.strip()
    for identity in allowed - names.keys():
        path = RUN / identity / 'episodes.jsonl'
        try:
            name = _archived_display_name(path, path.stat().st_mtime_ns)
        except (OSError, ValueError):
            continue
        if name:
            names[identity] = name
    return names


def source_messages(user):
    for sample in dataset_samples():
        if 'mobilemem-' + str(sample.get('person', {}).get('id', '')) == user:
            return {message['id']: message for session in sample.get('sessions', [])
                    for message in session.get('messages', []) if message.get('id')}
    return {}


def time_key(value):
    return (value or '').replace(' ', 'T', 1)


def _archive_episode_sources(user):
    sources = {}
    with (RUN / user / 'episodes.jsonl').open(encoding='utf-8-sig') as archive:
        for line in archive:
            if not line.strip():
                continue
            episode = json.loads(line)
            episode_id = episode.get('id')
            source_id = (episode.get('metadata') or {}).get('source_message_id')
            if isinstance(episode_id, str) and isinstance(source_id, str) and source_id:
                sources[episode_id] = source_id
    return sources


def _generate_run_config(users):
    sources_by_user = {user: _archive_episode_sources(user) for user in users}
    episode_ids = sorted({identity for sources in sources_by_user.values() for identity in sources})
    graph = LocalGraph()
    rows = graph.experiment_episode_scopes(episode_ids) if episode_ids else []
    scopes_by_episode = {}
    for row in rows:
        scope = row['scope']
        if isinstance(scope, str) and GRAPH_SCOPE_PATTERN.fullmatch(scope):
            scopes_by_episode.setdefault(row['episode_id'], set()).add(scope)

    scopes = {}
    for user, sources in sources_by_user.items():
        matches = set()
        for identity, source in sources.items():
            for scope in scopes_by_episode.get(identity, ()):
                # The HGP writer derives each episode ID from its graph scope and source message ID.
                payload = json.dumps((scope, source), ensure_ascii=False).encode()
                if hashlib.sha256(payload).hexdigest() == identity:
                    matches.add(scope)
        if len(matches) != 1:
            raise ExperimentConfigUnavailable('无法为所有评测用户匹配图数据。请恢复同一次实验的事件归档和 Neo4j 数据库。')
        scopes[user] = matches.pop()
    if len(set(scopes.values())) != len(scopes):
        raise ExperimentConfigUnavailable('评测用户的图数据空间重叠，请检查实验归档。')

    uri, _, _ = graph.connection_settings()
    return {
        'save_dir': str(RUN),
        'graph_scopes': scopes,
        'neo4j_uri': uri,
        'update_batch_size': 100,
        'adaptive_updates': True,
        'allow_llm_fallback': True,
        'allow_llm_extraction': True,
    }


def _read_run_config(user):
    users = evaluated_users()
    if user not in users:
        raise ValueError('未知的实验用户')
    path = RUN / 'live-config.json'
    with _config_lock:
        try:
            return json.loads(path.read_text(encoding='utf-8-sig'))
        except FileNotFoundError:
            pass

        config = _generate_run_config(users)
        with TemporaryDirectory(dir=RUN) as directory:
            staged = Path(directory) / path.name
            staged.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            try:
                path.hardlink_to(staged)
            except FileExistsError:
                return json.loads(path.read_text(encoding='utf-8-sig'))
        return config


def experiment_namespace(user, run_config=None):
    if run_config is None:
        run_config = _read_run_config(user)
    elif user not in evaluated_users():
        raise ValueError('未知的实验用户')
    scope = run_config.get('graph_scopes', {}).get(user)
    if scope is not None:
        if not isinstance(scope, str) or not GRAPH_SCOPE_PATTERN.fullmatch(scope):
            raise ValueError('评测用户的图数据空间配置无效')
        return scope
    if 'graph_namespace' not in run_config:
        raise ExperimentConfigUnavailable('实验配置缺少该评测用户的图数据空间。请恢复对应实验配置。')
    identity = json.dumps((run_config['graph_namespace'], user), ensure_ascii=False).encode()
    return 'hgp_' + hashlib.sha256(identity).hexdigest()


def read_profile(user, query=''):
    run_config = _read_run_config(user)
    namespace = experiment_namespace(user, run_config)
    folder = RUN / user
    memories = LocalMemories(folder)
    episodes = []
    episodes_by_id = {}
    with (folder / 'episodes.jsonl').open(encoding='utf-8-sig') as file:
        for line in file:
            if line.strip():
                episode = json.loads(line)
                record = {
                    'text': episode.get('content') or '',
                    'date': episode.get('timestamp') or '',
                    'source': episode.get('metadata', {}).get('source_message_id') or episode.get('id'),
                    'execution_status': episode.get('execution_status'),
                }
                episodes.append(record)
                episodes_by_id[episode['id']] = record
    episodes.sort(key=lambda item: time_key(item['date']), reverse=True)

    rules_source = next(iter(memories.sources('procedural')), None)
    rules = memories.read('procedural', rules_source['id'], limit=None)['rules'] if rules_source else []
    rules = [rule for rule in rules if rule['status'] in (None, 'active')]
    for rule in rules:
        rule['source_timestamp'] = (rule['meta'] or {}).get('timestamp') or ''
    rules.sort(key=lambda rule: time_key(rule['source_timestamp']), reverse=True)

    working_source = next(iter(memories.sources('working')), None)
    trajectories = memories.read('working', working_source['id'], limit=None)['groups'] if working_source else []
    trajectories = [trajectory for trajectory in trajectories if trajectory['event'] == namespace]
    states = [node for trajectory in trajectories for node in trajectory['nodes']]
    states.sort(key=lambda node: time_key(node['timestamp']), reverse=True)

    semantic_error = False
    query_extraction = None
    qa_context = None
    if query:
        result, working_matches, query_extraction, qa_context = search_run(run_config, folder, user, query, scope=namespace)
        facts = [{'relation': fact['relation'], 'entity': fact['entity'],
                  'date': fact.get('last_seen_at') or fact.get('created_at'),
                  'evidence': fact.get('evidence_ids'), 'score': fact.get('similarity')}
                 for fact in result.semantic]
        visible_episodes = [episodes_by_id[item['episode_id']] for item in result.episodic
                            if item['episode_id'] in episodes_by_id]
        visible_rules = []
        for item in result.procedural:
            rule = {key: value for key, value in item['rule'].items() if key != 'embedding'}
            rule['source_timestamp'] = (rule.get('meta') or {}).get('timestamp') or ''
            visible_rules.append(rule)
        state_by_id = {state['node_id']: state for state in states}
        visible_states = [state_by_id[state['state_id']] for state, _ in working_matches
                          if state['state_id'] in state_by_id]
    else:
        visible_episodes = episodes[:RECENT_K]
        visible_rules = rules[:RECENT_K]
        visible_states = states[:RECENT_K]
        try:
            facts = LocalGraph().experiment_facts(user, namespace)
        except Exception:
            facts = []
            semantic_error = True
    episodes_by_source = {episode['source']: episode for episode in episodes if episode['source']}
    messages = source_messages(user)
    for fact in facts:
        evidence = []
        for source in fact['evidence'] or []:
            if source in messages:
                message = messages[source]
                evidence.append((message.get('timestamp'), message.get('content')))
            elif source in episodes_by_source:
                episode = episodes_by_source[source]
                evidence.append((episode['date'], episode['text']))
        source_time, source_text = max(evidence, key=lambda item: time_key(item[0]), default=(None, None))
        fact['source_timestamp'] = source_time
        fact['source_text'] = source_text
    if not query:
        facts.sort(key=lambda fact: time_key(fact['source_timestamp']), reverse=True)
        facts = facts[:RECENT_K]

    def with_episode(node):
        return {**node, 'related_episode': episodes_by_source.get(node['source_episode_id'])}

    linked_states = [with_episode(node) for node in states
                     if node['source_episode_id'] in episodes_by_source][:RECENT_K] if not query else []
    visible_ids = {node['node_id'] for node in visible_states + linked_states}
    transitions = [edge for trajectory in trajectories for edge in trajectory['transitions']
                   if edge['from_node_id'] in visible_ids and edge['to_node_id'] in visible_ids]
    return {
        'semantic': facts,
        'episodic': visible_episodes,
        'procedural': visible_rules,
        'working': [with_episode(node) for node in visible_states],
        'working_linked': linked_states,
        'working_transitions': transitions,
        'semantic_error': semantic_error,
        'query_extraction': query_extraction,
        'qa_context': qa_context,
    }
