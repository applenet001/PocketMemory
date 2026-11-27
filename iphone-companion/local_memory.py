"""Read HGP files and persist per-user, source-only upload projections."""
import hashlib
import json
from pathlib import Path
from memory_draft import memory_evidence, memory_labels

ROOT = Path(__file__).resolve().parents[1] / 'outputs'
FILENAMES = {'working': 'state_trajectory.json', 'procedural': 'procedural_rules.jsonl'}


class LocalMemories:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()

    def write_uploads(self, items):
        self.root.mkdir(parents=True, exist_ok=True)
        working, rules = {}, []
        for item in reversed(items):
            result = item.get('classification') or {}
            if result.get('status') != 'classified' or result.get('uncertain'):
                continue
            identity = item['id']
            content = memory_evidence(item) or item['title']
            if result.get('working_state'):
                working[identity] = {'nodes': [{
                    'node_id': identity, 'state_name': item['title'], 'content': content,
                    'timestamp': item['created'], 'source_episode_id': identity,
                    'is_active': True}], 'transitions': []}
            if 'procedural' in memory_labels(result):
                rules.append({'id': identity, 'title': item['title'], 'source_text': content,
                              'if_conditions': [], 'then_steps': [], 'version': 1,
                              'status': 'source_only', 'valid_from': item['created'],
                              'meta': {'source_id': identity, 'extraction': 'not_extracted'}})
        for kind, content in [('working', json.dumps(working, ensure_ascii=False)),
                              ('procedural', '\n'.join(json.dumps(rule, ensure_ascii=False) for rule in rules))]:
            path = self.root / FILENAMES[kind]
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(content, encoding='utf-8')
            temporary.replace(path)

    def sources(self, kind):
        if kind not in FILENAMES:
            raise ValueError('未知记忆类型')
        result = []
        for candidate in self.root.rglob(FILENAMES[kind]):
            path = candidate.resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                continue
            relative = path.relative_to(self.root).as_posix()
            result.append({'id': hashlib.sha256(relative.encode()).hexdigest(),
                           'label': relative, 'kind': kind})
        return sorted(result, key=lambda item: item['label'])

    def read(self, kind, identity, *, limit=200):
        source = next((item for item in self.sources(kind) if item['id'] == identity), None)
        if source is None:
            raise ValueError('未找到该数据来源，请刷新列表')
        path = (self.root / source['label']).resolve()
        if not path.is_relative_to(self.root) or path.stat().st_size > 30 * 1024 * 1024:
            raise ValueError('文件超出本地查看范围或超过 30 MB')
        if kind == 'working':
            payload = json.loads(path.read_text(encoding='utf-8-sig'))
            groups = []
            for event, data in payload.items():
                nodes = data if isinstance(data, list) else data.get('nodes', [])
                nodes = sorted(nodes, key=lambda node: node.get('timestamp', ''))
                # Same current-state convention as StateTrajectoryGraph: latest active node.
                latest = nodes[-1] if nodes and nodes[-1].get('is_active', True) else None
                fields = ('node_id', 'state_name', 'timestamp', 'duration_seconds',
                          'source_episode_id', 'content', 'is_active', 'priority')
                selected = nodes[-limit:] if limit is not None else nodes
                visible = [{key: node.get(key) for key in fields} for node in selected]
                ids = {node['node_id'] for node in visible}
                transitions = [] if isinstance(data, list) else data.get('transitions', [])
                transitions = [{key: edge.get(key) for key in ('from_node_id', 'to_node_id', 'timestamp', 'reason')}
                               for edge in transitions if edge.get('from_node_id') in ids and edge.get('to_node_id') in ids]
                groups.append({'event': event, 'nodes': visible, 'transitions': transitions,
                               'total': len(nodes), 'current_id': latest.get('node_id') if latest else None})
            return {'kind': kind, 'source': source, 'groups': groups}
        records = {}
        for index, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines()):
            if not line.strip():
                continue
            rule = json.loads(line)
            # Append-only files can contain multiple versions of the same ID.
            records[rule.get('id') or str(index)] = rule
        fields = ('id', 'title', 'if_conditions', 'if_condition', 'then_steps', 'tree',
                  'source_text', 'meta', 'priority', 'status', 'version', 'valid_from', 'valid_to')
        rules = list(records.values())
        if limit is not None:
            rules = rules[-limit:]
        return {'kind': kind, 'source': source, 'total': len(records),
                'rules': [{key: rule.get(key) for key in fields} for rule in rules]}
