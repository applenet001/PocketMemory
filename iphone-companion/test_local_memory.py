import json
from pathlib import Path
import tempfile
import unittest

from local_memory import LocalMemories


class LocalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reader = LocalMemories(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_working_time_order_current_and_transition(self):
        data = {'task': {'nodes': [
            {'node_id': 'b', 'timestamp': '2026-01-02', 'state_name': 'confirm', 'is_active': True},
            {'node_id': 'a', 'timestamp': '2026-01-01', 'state_name': 'draft', 'is_active': False}],
            'transitions': [{'from_node_id': 'a', 'to_node_id': 'b', 'reason': 'ready'}]}}
        (self.root / 'state_trajectory.json').write_text(json.dumps(data), encoding='utf-8')
        source = self.reader.sources('working')[0]
        group = self.reader.read('working', source['id'])['groups'][0]
        self.assertEqual([node['node_id'] for node in group['nodes']], ['a', 'b'])
        self.assertEqual(group['current_id'], 'b')
        self.assertEqual(group['transitions'][0]['reason'], 'ready')

    def test_procedural_last_version_and_embedding_exclusion(self):
        rules = [{'id': 'r', 'version': 1, 'embedding': [1, 2]},
                 {'id': 'r', 'version': 2, 'then_steps': ['confirm'], 'embedding': [3, 4]}]
        (self.root / 'procedural_rules.jsonl').write_text('\n'.join(json.dumps(rule) for rule in rules), encoding='utf-8')
        source = self.reader.sources('procedural')[0]
        result = self.reader.read('procedural', source['id'])
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['rules'][0]['version'], 2)
        self.assertNotIn('embedding', result['rules'][0])

    def test_no_client_supplied_filesystem_paths(self):
        with self.assertRaises(ValueError):
            self.reader.read('working', '../../config/local.env')
        with self.assertRaises(ValueError):
            self.reader.sources('../../config')
        self.assertEqual(self.reader.sources('working'), [])


if __name__ == '__main__':
    unittest.main()
