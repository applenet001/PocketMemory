import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from accounts import Accounts, graph_namespace
from server import Store, make_server


class MultiUserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.owner_key = 'owner-private-test-key-' * 3
        self.store = Store(self.root)
        self.legacy, _ = self.store.add(text='Owner legacy private memory')
        self.upload_graph = Mock()
        self.server = make_server(self.store, self.owner_key, port=0, classifier=self.classify,
                                  upload_graph=self.upload_graph, background=False)
        self.accounts = Accounts(self.root, self.owner_key)
        self.a, self.key_a = self.accounts.create('Alice')
        self.b, self.key_b = self.accounts.create('Bob')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    @staticmethod
    def classify(item):
        if item['text'] in ('Event', 'Uncertain event'):
            return {'status': 'classified', 'label': 'episodic', 'working_state': False,
                    'uncertain': item['text'] == 'Uncertain event', 'confidence': 0.9}
        if item['text'] == 'Working only':
            return {'status': 'classified', 'label': None, 'working_state': True}
        return {'status': 'classified', 'label': 'procedural', 'working_state': False}

    def test_personal_graph_sync_retries_and_respects_ownership(self):
        self.upload_graph.write_upload.side_effect = RuntimeError('offline')
        status, upload, _ = self.request('/api/memories', self.key_a, {'text': 'Event'})
        self.assertEqual(status, 201)
        self.assertEqual(upload['item']['graph_state'], 'pending')
        memory = '/api/memories/' + upload['item']['id']
        self.assertEqual(self.request(memory, self.key_b)[0], 404)
        self.assertEqual(self.request(memory + '/graph-sync', self.key_b, method='POST')[0], 404)
        self.upload_graph.write_upload.side_effect = None
        self.assertEqual(self.request(memory + '/graph-sync', self.key_a, method='POST')[1]['item']['graph_state'], 'synced')
        self.assertEqual(self.upload_graph.write_upload.call_args.args[0], graph_namespace(self.a['id']))
        self.request(memory + '/graph-sync', self.key_a, method='POST')
        self.request('/api/memories', self.key_a, {'text': 'Uncertain event'})
        self.assertEqual(self.upload_graph.write_upload.call_count, 2)
        self.assertEqual(self.request('/api/graph?database=personal-episodic&namespace=' +
                                      graph_namespace(self.a['id']), self.owner_key)[0], 403)
        self.assertTrue(self.request(memory, self.key_a, method='DELETE')[1]['deleted'])
        self.upload_graph.delete_upload.assert_called_once_with(graph_namespace(self.a['id']), upload['item']['id'])

    def test_profile_uses_only_personal_records_and_survives_search(self):
        for text in ('Old rule', 'Next rule', 'Recent rule', 'Latest rule', 'Working only'):
            self.assertEqual(self.request('/api/memories', self.key_a, {'text': text})[0], 201)
        self.request('/api/memories', self.key_b, {'text': 'Other user rule'})

        alice = self.request('/api/memories?q=no-match', self.key_a)[1]
        self.assertEqual(alice['items'], [])
        self.assertEqual([item['text'] for item in alice['profile']['procedural']],
                         ['Latest rule', 'Recent rule', 'Next rule'])
        self.assertEqual([item['text'] for item in alice['profile']['working']], ['Working only'])
        self.assertEqual(alice['profile']['semantic'], [])
        self.assertEqual(alice['profile']['episodic'], [])
        bob = self.request('/api/memories', self.key_b)[1]['profile']
        self.assertEqual([item['text'] for item in bob['procedural']], ['Other user rule'])
        self.assertEqual(bob['working'], [])

    def test_personal_workflows_have_sources_and_are_isolated(self):
        _, rule, _ = self.request('/api/memories', self.key_a, {'text': 'My reusable method'})
        _, state, _ = self.request('/api/memories', self.key_a, {'text': 'Working only'})
        self.request('/api/memories', self.key_b, {'text': 'Other user method'})
        status, data, _ = self.request('/api/workflows', self.key_a)
        self.assertEqual(status, 200)
        self.assertEqual(len(data['procedural']), 1)
        self.assertEqual(data['procedural'][0]['source_text'], 'My reusable method')
        self.assertEqual(data['procedural'][0]['upload_id'], rule['item']['id'])
        self.assertEqual(data['working'][0]['nodes'][0]['upload_id'], state['item']['id'])
        bob = self.request('/api/workflows?user=' + self.a['id'], self.key_b)[1]
        self.assertEqual(bob['working'], [])
        self.assertEqual(bob['procedural'][0]['source_text'], 'Other user method')
        self.assertEqual(self.request('/api/workflows')[0], 401)

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.temp.cleanup()

    def request(self, path, key=None, value=None, method=None, cookie=None):
        headers = {}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        if cookie:
            headers['Cookie'] = cookie
        if value is not None:
            headers['Content-Type'] = 'application/json'
        request = Request(self.url + path, data=json.dumps(value).encode() if value is not None else None, headers=headers, method=method)
        try:
            response = urlopen(request, timeout=20)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read()
            body = json.loads(raw) if response.headers['Content-Type'].startswith('application/json') else raw
            return response.status, body, response.headers

    def test_lists_exports_files_deletion_and_dedup_are_isolated(self):
        payload = {'name': 'note.txt', 'base64': base64.b64encode(b'private shared text').decode(), 'user_id': 'owner'}
        _, first, _ = self.request('/api/memories', self.key_a, payload)
        _, second, _ = self.request('/api/memories', self.key_b, payload)
        id_a, id_b = first['item']['id'], second['item']['id']
        self.assertNotEqual(id_a, id_b)
        self.assertTrue(first['created']); self.assertTrue(second['created'])
        for key, identity in [(self.key_a, id_a), (self.key_b, id_b), (self.owner_key, self.legacy['id'])]:
            items = self.request('/api/memories', key)[1]['items']
            self.assertEqual([item['id'] for item in items], [identity])
            self.assertEqual([item['id'] for item in self.request('/api/export', key)[1]['items']], [identity])
        self.assertEqual(self.request('/api/files/' + id_a, self.key_b)[0], 404)
        self.assertFalse(self.request('/api/memories/' + id_a, self.key_b, method='DELETE')[1]['deleted'])
        self.assertEqual(self.request('/api/files/' + id_a, self.key_a)[1], b'private shared text')
        self.assertEqual(self.request('/api/memories', self.key_a, payload)[0], 200)
        self.assertIsNotNone(Store(self.root / 'users' / self.a['id']).get(id_a))
        self.assertIsNone(Store(self.root / 'users' / self.b['id']).get(id_a))
        self.assertIsNotNone(self.store.get(self.legacy['id']))
        self.assertEqual(Store(self.root / 'users' / self.a['id']).get(id_a)['classification']['label'], 'procedural')
        self.assertEqual(self.request('/api/memories/' + id_a + '/classify', self.key_b, method='POST')[0], 404)
        self.assertIsNone(self.store.get(self.legacy['id'])['classification'])
        status, _, _ = self.request('/api/memories/' + self.legacy['id'] + '/classify', self.owner_key, method='POST')
        self.assertEqual(status, 200)
        self.assertEqual(self.store.get(self.legacy['id'])['classification']['label'], 'procedural')

    def test_admin_creation_has_no_client_role_or_id_override(self):
        self.assertEqual(self.request('/api/admin/users', self.key_a)[0], 403)
        self.assertEqual(self.request('/api/admin/users', self.key_a, {'name': 'Mallory'})[0], 403)
        status, result, _ = self.request('/api/admin/users', self.owner_key, {'name': 'New person', 'id': 'owner', 'role': 'admin'})
        self.assertEqual(status, 201)
        self.assertEqual(result['user']['role'], 'user')
        self.assertNotEqual(result['user']['id'], 'owner')
        self.assertEqual(self.request('/api/me', result['key'])[1]['user']['id'], result['user']['id'])

    def test_experiment_profile_is_admin_only_and_requires_one_user(self):
        path = '/api/admin/experiment-profile'
        with patch('server.evaluated_users', return_value=['experiment-a', 'experiment-b']), \
                patch('server.read_profile', return_value={'semantic': []}) as read:
            self.assertEqual(self.request(path, self.key_a)[0], 403)
            self.assertEqual(self.request(path + '?user=experiment-a', self.key_a)[0], 403)
            self.assertEqual(self.request(path, self.owner_key)[1]['users'], ['experiment-a', 'experiment-b'])
            read.assert_not_called()
            self.assertEqual(self.request(path + '?user=experiment-b', self.owner_key)[0], 200)
            read.assert_called_once_with('experiment-b')

    def test_experiment_search_is_admin_only_and_requires_user_and_query(self):
        path = '/api/admin/experiment-search'
        query = '?user=experiment-a&q=library'
        with patch('server.read_profile', return_value={'episodic': []}) as read:
            self.assertEqual(self.request(path + query, self.key_a)[0], 403)
            self.assertEqual(self.request(path + '?q=library', self.owner_key)[0], 400)
            self.assertEqual(self.request(path + '?user=experiment-a', self.owner_key)[0], 400)
            read.assert_not_called()
            self.assertEqual(self.request(path + query, self.owner_key)[0], 200)
            read.assert_called_once_with('experiment-a', 'library')

    def test_named_experiment_graph_is_scoped_to_selected_user(self):
        path = '/api/graph?database=episodic&user=experiment-a'
        with patch('server.experiment_namespace', return_value='scope-a'), \
                patch('server.LocalGraph.view', return_value={'nodes': [], 'edges': []}) as view:
            self.assertEqual(self.request(path, self.key_a)[0], 403)
            self.assertEqual(self.request(path + '&namespace=scope-b', self.owner_key)[0], 403)
            view.assert_not_called()
            self.assertEqual(self.request(path, self.owner_key)[0], 200)
            view.assert_called_once_with('episodic', 'scope-a', allowed_namespace='scope-a')

    def test_portrait_only_uses_selected_account_memories(self):
        self.request('/api/memories', self.key_a, {'text': 'Alice is trying oolong tea'})
        self.request('/api/memories', self.key_b, {'text': 'Bob is learning guitar'})
        with patch('server.generate_portrait', return_value={'status': 'ready', 'text': 'Portrait'}) as generate:
            self.assertEqual(self.request('/api/portrait', self.owner_key, {'consent': True})[0], 403)
            self.assertEqual(self.request('/api/admin/experiment-portrait?user=experiment-a', self.key_a)[0], 403)
            self.assertEqual(self.request('/api/portrait', self.key_a, {})[0], 400)
            generate.assert_not_called()
            self.assertEqual(self.request('/api/portrait', self.key_a, {'consent': True})[0], 200)
            scope, evidence = generate.call_args.args
            self.assertEqual(scope, 'user:' + self.a['id'])
            self.assertEqual(len(evidence['procedural_saved_records']), 1)
            self.assertTrue(evidence['procedural_saved_records'][0].endswith('Alice is trying oolong tea'))

    def test_eval_answer_uses_server_side_context_from_the_same_search(self):
        path = '/api/admin/experiment-answer'
        with patch('server.read_profile', return_value={'episodic': [], 'qa_context': 'Selected HGP memories'}), \
                patch('server.answer_question', return_value={'status': 'ready', 'text': 'Answer'}) as answer:
            _, result, _ = self.request('/api/admin/experiment-search?user=experiment-a&q=question', self.owner_key)
            self.assertNotIn('qa_context', result['profile'])
            ticket = result['answer_ticket']
            self.assertTrue(ticket)
            self.assertEqual(self.request(path, self.key_a, {'ticket': ticket})[0], 403)
            self.assertEqual(self.request(path, self.owner_key, {'ticket': 'z' * 32})[0], 404)
            answer.assert_not_called()
            self.assertEqual(self.request(path, self.owner_key, {'ticket': ticket, 'context': 'client text'})[0], 200)
            answer.assert_called_once_with('question', 'Selected HGP memories')

    def test_sessions_logout_rotation_and_disable(self):
        status, _, headers = self.request('/api/login', value={'token': self.key_a})
        self.assertEqual(status, 200)
        cookie = headers['Set-Cookie'].split(';')[0]
        self.assertNotIn(self.key_a, cookie)
        self.assertEqual(self.request('/api/me', cookie=cookie)[1]['user']['id'], self.a['id'])
        self.request('/api/logout', method='POST', cookie=cookie)
        self.assertEqual(self.request('/api/me', cookie=cookie)[0], 401)
        _, _, headers = self.request('/api/login', value={'token': self.key_a})
        cookie = headers['Set-Cookie'].split(';')[0]
        path = '/api/admin/users/' + self.a['id']
        self.assertEqual(self.request(path + '/rotate', self.key_b, method='POST')[0], 403)
        _, result, _ = self.request(path + '/rotate', self.owner_key, method='POST')
        new_key = result['key']
        self.assertEqual(self.request('/api/me', self.key_a)[0], 401)
        self.assertEqual(self.request('/api/me', cookie=cookie)[0], 401)
        self.assertEqual(self.request('/api/me', new_key)[0], 200)
        self.request(path + '/disable', self.owner_key, method='POST')
        self.assertEqual(self.request('/api/me', new_key)[0], 401)
        self.request(path + '/enable', self.owner_key, method='POST')
        self.assertEqual(self.request('/api/me', new_key)[0], 200)

    def test_graph_and_local_history_cannot_cross_users(self):
        with patch('server.LocalGraph.view', return_value={'scopes': [], 'nodes': [], 'edges': []}) as view:
            self.assertEqual(self.request('/api/graph', self.key_a)[0], 403)
            self.assertEqual(self.request('/api/graph?database=personal-episodic', self.key_a)[0], 200)
            self.assertEqual(view.call_args.kwargs['allowed_namespace'], graph_namespace(self.a['id']))
            view.reset_mock()
            self.assertEqual(self.request('/api/graph?database=personal-episodic&namespace=' + graph_namespace(self.b['id']), self.key_a)[0], 403)
            view.assert_not_called()
        for user in (self.a, self.b):
            folder = self.root / 'users' / user['id'] / 'hgp' / user['id']
            folder.mkdir(parents=True)
            (folder / 'state_trajectory.json').write_text(json.dumps({user['id']: []}), encoding='utf-8')
        a_sources = self.request('/api/local-memory?kind=working', self.key_a)[1]['sources']
        b_sources = self.request('/api/local-memory?kind=working', self.key_b)[1]['sources']
        self.assertEqual(len(a_sources), 1)
        self.assertNotEqual(a_sources[0]['id'], b_sources[0]['id'])
        self.assertEqual(self.request('/api/local-memory?kind=working&source=' + b_sources[0]['id'], self.key_a)[0], 400)

    def test_hashed_keys_expired_sessions_and_restart(self):
        raw = (self.root / 'accounts.sqlite3').read_bytes()
        self.assertNotIn(self.key_a.encode(), raw)
        self.assertNotIn(self.owner_key.encode(), raw)
        restored = Accounts(self.root, self.owner_key)
        self.assertEqual(restored.by_key(self.key_a)['id'], self.a['id'])
        _, session = restored.login(self.key_a)
        with restored.connect() as db:
            db.execute('UPDATE sessions SET expires=0')
        self.assertIsNone(restored.by_session(session))
        self.assertEqual(self.request('/api/me', cookie='pm_session=' + self.owner_key)[0], 401)


if __name__ == '__main__':
    unittest.main()
