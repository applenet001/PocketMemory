import base64
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from unittest.mock import Mock

from local_memory import LocalMemories
from server import Store, make_server


class ProcessingTests(unittest.TestCase):
    def test_upload_returns_before_extraction_and_classification_then_persists(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            started, release = threading.Event(), threading.Event()

            def extract(content, name):
                started.set()
                release.wait(5)
                return content.decode(), 'extracted'

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
