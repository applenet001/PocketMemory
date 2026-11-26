"""Build a read-only profile from one evaluation trajectory."""
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import re

from graph import LocalGraph
from local_memory import LocalMemories
from experiment_search import search_run

RUN = Path(__file__).resolve().parents[1] / 'outputs/mobilemem-qwen/hgp-history-time-qa'
DATASET = RUN.parents[2] / 'data/text/mobilemem_data.json'
RECENT_K = 5


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


def evaluated_user_names(users):
    allowed = set(users)
    names = {}
    for sample in dataset_samples():
        person = sample.get('person', {})
        identity = 'mobilemem-' + str(person.get('id', ''))
        if identity in allowed and person.get('name'):
            names[identity] = person['name']
    return names


def source_messages(user):
    for sample in dataset_samples():
        if 'mobilemem-' + str(sample.get('person', {}).get('id', '')) == user:
            return {message['id']: message for session in sample.get('sessions', [])
                    for message in session.get('messages', []) if message.get('id')}
    return {}


def time_key(value):
    return (value or '').replace(' ', 'T', 1)


def experiment_namespace(user, run_config=None):
    if user not in evaluated_users():
        raise ValueError('未知的实验用户')
    if run_config is None:
        run_config = json.loads((RUN / 'live-config.json').read_text(encoding='utf-8-sig'))
    identity = json.dumps((run_config['graph_namespace'], user), ensure_ascii=False).encode()
    return 'hgp_' + hashlib.sha256(identity).hexdigest()


def read_profile(user, query=''):
    run_config = json.loads((RUN / 'live-config.json').read_text(encoding='utf-8-sig'))
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
        result, working_matches, query_extraction, qa_context = search_run(run_config, folder, user, query)
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
