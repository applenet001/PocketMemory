import base64
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from unittest.mock import Mock

from local_memory import LocalMemories
from memory_draft import ocr_lines
from server import Store, make_server
from v2_memory import CompanionV2


class ProcessingTests(unittest.TestCase):
    def test_v2_shortcut_uses_one_account_scoped_entry_after_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            legacy, _ = store.add(text='Legacy inbox item')
            legacy_classifier = Mock()
            legacy_graph = Mock()
            backend = Mock(spec=CompanionV2)
            token = 'x' * 40

            def ingest(target, user_id, item, checkpoint):
                self.assertEqual(target, Path(folder) / 'hgp' / 'v2')
                self.assertEqual(user_id, 'owner')
                self.assertEqual(item['text'], 'Scanned Todo')
                self.assertEqual(item['v2_state']['todo'], {'todo_id': 'task-1', 'is_completed': True})
                item['v2_state'].update(complete=True, types=['episodic', 'working'])
                return {'status': 'classified', 'label': 'episodic', 'working_state': True}

            backend.ingest.side_effect = ingest
            server = make_server(store, token, port=0, classifier=legacy_classifier,
                                 upload_graph=legacy_graph, v2_backend=backend)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            payload = {'name': 'task.png', 'base64': base64.b64encode(b'image').decode(),
                       'source_app': 'Todo', 'todo_id': 'task-1', 'is_completed': True}
            request = Request(f'http://127.0.0.1:{server.server_port}/api/memories',
                              data=json.dumps(payload).encode(),
                              headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
            try:
                with unittest.mock.patch.object(store, 'extract', return_value=('Scanned Todo', 'OCR', [])):
                    with urlopen(request, timeout=5) as response:
                        identity = json.load(response)['item']['id']
                    deadline = time.monotonic() + 5
                    while store.get(identity)['processing'] not in ('complete', 'failed') and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertEqual(store.get(identity)['processing'], 'complete')
                    with urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 200)
                        self.assertEqual(json.load(response)['item']['id'], identity)
                backend.ingest.assert_called_once()
                self.assertIsNone(store.get(legacy['id'])['v2_state'])
                legacy_classifier.assert_not_called()
                legacy_graph.write_upload.assert_not_called()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_ocr_draft_excludes_garble_but_retains_raw_lines(self):
        blocks = ocr_lines(SimpleNamespace(txts=['Boarding at 7:15', '\ufffd\ufffd?', 'Garbled'],
                                           boxes=[[[1, 2], [3, 4]]] * 3, scores=[.98, .99, .55]))
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            with unittest.mock.patch.object(store, 'extract', return_value=('Boarding at 7:15', 'OCR', blocks)):
                item, _ = store.add(name='flight.png', content=b'image', source_app='Safari')
            item = Store(folder).get(item['id'], detail=True)
            self.assertEqual(item['draft']['summary'], 'Boarding at 7:15')
            self.assertEqual(item['source_app'], 'Safari')
            self.assertEqual([line['accepted'] for line in item['ocr']], [True, False, False])
            self.assertEqual(item['ocr'][2]['text'], 'Garbled')
            self.assertEqual(item['ocr'][0]['box'], [[1, 2], [3, 4]])

    def test_file_without_text_can_become_a_training_example(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            item, _ = store.add(name='clip.bin', content=b'\x00\x01', source_app='Files')
            self.assertEqual(item['draft']['status'], 'source_only')
            self.assertEqual(store.get(item['id'], file=True)['content'], b'\x00\x01')
            feedback = {'label': 'episodic', 'working_state': False, 'text': ''}
            store.record_feedback(item['id'], feedback)
            self.assertEqual(store.feedback_examples(), ())
            feedback['text'] = 'Signed the lease on Friday.'
            store.record_feedback(item['id'], feedback)
            self.assertEqual(Store(folder).feedback_examples(), ((feedback['text'], 'episodic', False),))
            store.delete(item['id'])
            self.assertEqual(store.feedback_examples(), ())

    def test_upload_returns_before_extraction_and_classification_then_persists(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            started, release = threading.Event(), threading.Event()

            def extract(content, name):
                started.set()
                release.wait(5)
                return content.decode(), 'extracted', []

            store.extract = extract
            classifier = Mock(return_value={'status': 'classified', 'label': 'procedural',
                                            'working_state': True, 'uncertain': False})
            server = make_server(store, 'x' * 40, port=0, classifier=classifier)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{server.server_port}/api/memories'
            try:
                request = Request(url, data=json.dumps({'name': 'task.txt',
                    'base64': base64.b64encode(b'Follow these instructions').decode()}).encode(),
                    headers={'Authorization': 'Bearer ' + 'x' * 40, 'Content-Type': 'application/json'})
                with urlopen(request, timeout=2) as response:
                    item = json.load(response)['item']
                self.assertTrue(started.wait(1))
                self.assertFalse(release.is_set())
                self.assertEqual(store.get(item['id'], file=True)['content'], b'Follow these instructions')
                classifier.assert_not_called()
                release.set()
                deadline = time.monotonic() + 5
                while store.get(item['id'])['processing'] != 'complete' and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(store.get(item['id'])['processing'], 'complete')
                reader = LocalMemories(Path(folder) / 'hgp')
                for kind in ('working', 'procedural'):
                    self.assertEqual(len(reader.sources(kind)), 1)
                rules = reader.read('procedural', reader.sources('procedural')[0]['id'])['rules']
                self.assertEqual(rules[0]['source_text'], 'Follow these instructions')
                self.assertEqual(rules[0]['status'], 'source_only')
                store.sync_local()
                self.assertEqual(reader.read('procedural', reader.sources('procedural')[0]['id'])['total'], 1)
                with store.graph_lock:
                    store.delete(item['id'])
                self.assertEqual(reader.read('working', reader.sources('working')[0]['id'])['groups'], [])
                self.assertEqual(reader.read('procedural', reader.sources('procedural')[0]['id'])['total'], 0)
            finally:
                release.set()
                server.shutdown()
                server.server_close()
                thread.join()

    def test_uncertain_and_other_users_are_excluded_from_files(self):
        with tempfile.TemporaryDirectory() as folder:
            alice, bob = Store(Path(folder) / 'a'), Store(Path(folder) / 'b')
            item, _ = alice.add(text='Uncertain instructions')
            alice.classify(item, lambda _: {'status': 'classified', 'label': 'procedural',
                                            'working_state': True, 'uncertain': True})
            alice.sync_local()
            bob.sync_local()
            for store in (alice, bob):
                reader = LocalMemories(store.root / 'hgp')
                self.assertEqual(reader.read('working', reader.sources('working')[0]['id'])['groups'], [])
                self.assertEqual(reader.read('procedural', reader.sources('procedural')[0]['id'])['total'], 0)
