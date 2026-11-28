"""Adapt the HGP v2 replay and retrieval to MemBase's memory layer contract."""

import json
from pathlib import Path

from membase.configs import CONFIG_MAPPING
from membase.configs.base import MemBaseConfig
from membase.layers import MEMORY_LAYERS_MAPPING
from membase.layers.base import MemBaseLayer
from membase.model_types.memory import MemoryEntry

from mobilemem_source import evidence_source_app, todo_completion_fields
from run_hgp_v2_smoke import attach_working_states, event_time, index_working_states, open_search, run


MEMORY_TYPE = 'HGP-v2'


class HGPV2Config(MemBaseConfig):
    construction_dir: Path
    search_only: bool = False

    def get_llm_models(self):
        return [MEMORY_TYPE]


class HGPV2Layer(MemBaseLayer):
    def __init__(self, config):
        self.config = config
        self.target = config.construction_dir
        self.rows = []
        self.reader = None

    def add_message(self, message, **kwargs):
        if not message.content.strip():
            return
        metadata = message.metadata
        self.rows.append(dict(user_id=self.config.user_id.removeprefix('mobilemem-'), message_id=message.id,
                              session_id=metadata['session_id'], role=message.role, name=message.name,
                              timestamp=message.timestamp, content=message.content,
                              **todo_completion_fields({'role': message.role, 'name': message.name, 'metadata': metadata})))

    def add_messages(self, messages, **kwargs):
        for message in messages:
            self.add_message(message, **kwargs)

    def flush(self):
        if not self.rows:
            raise ValueError('HGP v2 expects at least one nonempty message')
        if len({row['message_id'] for row in self.rows}) != len(self.rows):
            raise ValueError('HGP v2 requires unique message IDs')
        self.rows.sort(key=lambda row: (event_time(row['timestamp']), row['session_id'], row['message_id']))
        run(self.rows, self.target)

    def save_memory(self):
        if not (self.target / 'report.json').exists():
            raise RuntimeError('HGP v2 construction did not finish')

    def load_memory(self, user_id=None):
        if user_id not in (None, self.config.user_id) or not (self.target / 'report.json').is_file():
            return False
        if self.config.search_only:
            self.reader = open_search(self.target)
            self.memory, report = self.reader.__enter__()
            self.states = index_working_states(report)
            self.sources = {turn['message_id']: turn['source_app'] for turn in report['turns'] if turn.get('source_app')}
        return True

    def retrieve(self, query, k=5, **kwargs):
        result = self.memory.search(query, top_k=k)
        result.episodic = attach_working_states(result.episodic, self.states)
        for triple in result.semantic:
            app = evidence_source_app(triple.get('evidence_ids') or [], self.sources)
            if app:
                triple['source_app'] = app
        entries = []
        for kind in ('semantic', 'episodic', 'procedural'):
            for item in getattr(result, kind)[:k]:
                content = item.get('content') or item.get('text') or item.get('entity') or item.get('title') or ''
                entries.append(MemoryEntry(content=str(content),
                                           metadata={'kind': kind, 'memory': json.loads(json.dumps(item, default=str))}))
        return entries

    def cleanup(self):
        if self.reader is not None:
            self.reader.__exit__(None, None, None)

    def delete(self, memory_id):
        raise NotImplementedError('Use the HGP per-store deletion APIs')

    def update(self, memory_id, **kwargs):
        raise NotImplementedError('Use the HGP per-store write APIs')


def build_hgp_v2_context(entries):
    from memory import SearchResult

    search = SearchResult()
    for entry in entries:
        getattr(search, entry.metadata['kind']).append(entry.metadata['memory'])
    return search.to_context(max_items=max((len(search.semantic), len(search.episodic), len(search.procedural)), default=0))


def register_v2():
    CONFIG_MAPPING.register(MEMORY_TYPE, HGPV2Config)
    MEMORY_LAYERS_MAPPING.register(MEMORY_TYPE, HGPV2Layer)
