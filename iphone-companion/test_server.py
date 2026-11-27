import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import Store, make_server


class CompanionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.token = 'test-token-' * 4
        self.server = make_server(self.store, self.token, background=False)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, value=None, method=None, auth=True, headers=None):
        headers = headers or {}
        if auth:
            headers['Authorization'] = 'Bearer ' + self.token
        data = value
        if isinstance(value, dict):
            data = json.dumps(value).encode()
            headers['Content-Type'] = 'application/json'
        request = Request(self.url + path, data=data, method=method, headers=headers)
        try:
            response = urlopen(request, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            body = response.read()
            return response.status, json.loads(body) if response.headers['Content-Type'].startswith('application/json') else body

    def test_authorization_and_cross_origin(self):
        self.assertEqual(self.request('/api/memories', auth=False)[0], 401)
        self.assertEqual(self.request('/api/login', {'token': 'wrong'}, auth=False)[0], 401)
        self.assertEqual(self.request('/api/memories', {'text': 'secret'}, headers={'Origin': 'https://evil.example'})[0], 403)

    def test_save_search_deduplicate_restart_and_delete(self):
        payload = {'text': 'https://example.com/sushi', 'note': '周六和朋友去日料餐厅'}
        status, first = self.request('/api/memories', payload)
        self.assertEqual(status, 201)
        self.assertEqual(self.request('/api/memories', payload)[0], 200)
        self.assertEqual(Store(self.temp.name).get(first['item']['id'])['note'], payload['note'])
        self.assertEqual(len(self.store.list('日料')), 1)
        self.assertEqual(len(self.store.list('不存在的关键词xyz')), 0)
        self.assertEqual(self.request('/api/export')[1]['schema'], 'pocketmemory.share.v1')
        self.assertTrue(self.request('/api/memories/' + first['item']['id'], method='DELETE')[1]['deleted'])
        self.assertEqual(self.store.list(), [])

    def test_shortcut_multipart_and_file_auth(self):
        body = (b'--boundary\r\nContent-Disposition: form-data; name="source_app"\r\n\r\nNotes\r\n'
                b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="../../notes.txt"\r\n'
                b'Content-Type: text/plain\r\n\r\nremember the library\r\n--boundary--\r\n')
        status, result = self.request('/api/memories', body, headers={'Content-Type': 'multipart/form-data; boundary=boundary'})
        self.assertEqual(status, 201)
        item = result['item']
        self.assertEqual(item['name'], 'notes.txt')
        self.assertEqual(item['source_app'], 'Notes')
        self.assertIn('library', item['text'])
        self.assertEqual(self.request('/api/files/' + item['id'], auth=False)[0], 401)
        self.assertEqual(self.request('/api/files/' + item['id'])[1], b'remember the library')
        self.assertFalse((Path(self.temp.name).parent / 'notes.txt').exists())

    def test_shortcut_multipart_link_without_base64(self):
        body = (b'--boundary\r\nContent-Disposition: form-data; name="text"\r\n\r\nhttps://example.com\r\n'
                b'--boundary\r\nContent-Disposition: form-data; name="source_app"\r\n\r\nSafari\r\n--boundary--\r\n')
        status, result = self.request('/api/memories', body, headers={'Content-Type': 'multipart/form-data; boundary=boundary'})
        self.assertEqual(status, 201)
        self.assertEqual(result['item']['url'], 'https://example.com')
        self.assertEqual(result['item']['source_app'], 'Safari')

    def test_reject_empty_and_invalid_base64(self):
        self.assertEqual(self.request('/api/memories', {})[0], 400)
        self.assertEqual(self.request('/api/memories', {'base64': '%%%'} )[0], 400)
        self.assertEqual(self.request('/api/memories', {'text': 'a' * 200001})[0], 400)

    def test_shortcut_upload_error_explains_request_body(self):
        status, result = self.request('/api/memories', b'', method='POST')
        self.assertEqual(status, 400)
        self.assertIn('请求没有带上传内容', result['error'])

    def test_shortcut_chunked_json_upload(self):
        text = '携程行程'
        encoded = base64.b64encode(text.encode()).decode()
        chunks = iter((b'{"shared_base64":"', encoded.encode(), b'"}'))
        status, result = self.request('/api/memories', chunks, headers={'Content-Type': 'application/json'})
        self.assertEqual(status, 201, result)
        self.assertEqual(self.store.list()[0]['text'], text)

    def test_shortcut_rejects_conflicting_framing(self):
        status, result = self.request('/api/memories', b'{}', headers={
            'Content-Type': 'application/json', 'Content-Length': '2', 'Transfer-Encoding': 'chunked',
        })
        self.assertEqual(status, 400)
        self.assertIn('请求长度标头冲突', result['error'])

    def test_shortcut_image_file_with_plain_text_response(self):
        path = '/api/memories/image'
        screenshot = b'\x89PNG\r\n\x1a\nimage bytes'
        self.assertEqual(self.request(path, screenshot, auth=False)[0], 401)
        status, result = self.request(path, b'', method='POST')
        self.assertEqual(status, 400)
        self.assertIn('快捷指令输入', result.decode('utf-8'))
        with patch.object(Store, 'extract', return_value=('行程', '图片已保存', [])):
            status, result = self.request(path, screenshot, headers={'Content-Type': 'application/octet-stream'})
            self.assertEqual(status, 201)
            self.assertEqual(result.decode('utf-8'), '已保存到 PocketMemory')
            self.assertEqual(self.request(path, screenshot)[0], 200)
            status, result = self.request('/api/memories', screenshot, headers={'Content-Type': 'application/octet-stream'})
            self.assertEqual(status, 200, result)
        item, = self.store.list()
        self.assertEqual(item['name'], 'iPhone.png')
        self.assertEqual(self.request('/api/files/' + item['id'])[1], screenshot)

    def test_one_shortcut_saves_link_text_and_image(self):
        path = '/api/memories'
        link = 'https://example.com/行程'
        text = '携程行程\n周六全家出发 👨‍👩‍👧'
        screenshot = b'\x89PNG\r\n\x1a\nimage bytes'
        for content in (link.encode(), text.encode(), screenshot):
            with self.subTest(content=content), patch.object(Store, 'extract', return_value=('截图正文', '图片已保存', [])):
                status, result = self.request(path, {'shared_base64': base64.encodebytes(content).decode()})
                self.assertEqual(status, 201)
                self.assertEqual(result.decode(), '已保存到 PocketMemory')
        items = self.store.list()
        self.assertIn(link, {item['url'] for item in items})
        self.assertIn(text, {item['text'] for item in items})
        image = next(item for item in items if item['name'] == 'iPhone.png')
        self.assertEqual(self.request('/api/files/' + image['id'])[1], screenshot)

    def test_one_shortcut_reports_missing_or_invalid_share(self):
        for invalid in ('', 'bad%data', ['not a single value']):
            with self.subTest(invalid=invalid):
                status, result = self.request('/api/memories', {'shared_base64': invalid})
                self.assertEqual(status, 400)
                self.assertIn('分享内容', result.decode())
        self.assertEqual(self.store.list(), [])

    def test_graph_auth_and_unavailable_is_explicit(self):
        self.assertEqual(self.request('/api/graph', auth=False)[0], 401)
        with patch('server.LocalGraph.view', side_effect=RuntimeError('private connection detail')):
            status, result = self.request('/api/graph')
            self.assertEqual(status, 503)
            self.assertNotIn('private connection detail', result['error'])
        with patch('server.LocalGraph.view', return_value={'nodes': [], 'edges': []}):
            self.assertEqual(self.request('/api/graph')[0], 200)

    def test_local_memory_requires_auth_and_rejects_path(self):
        self.assertEqual(self.request('/api/local-memory', auth=False)[0], 401)
        self.assertEqual(self.request('/api/local-memory?kind=unknown')[0], 400)
        self.assertEqual(self.request('/api/local-memory?kind=working&source=../../config/local.env')[0], 400)

    def test_hgp_export_preserves_source_and_rejects_unknown_schema(self):
        from hgp_bridge import records
        item, _ = self.store.add(text='https://example.com', note='source note')
        result = records({'schema': 'pocketmemory.share.v1', 'items': [item]})
        self.assertEqual(result[0]['id'], 'iphone-' + item['id'])
        self.assertIn('source note', result[0]['content'])
        self.assertEqual(result[0]['timestamp'], item['created'])
        with self.assertRaises(ValueError):
            records({'schema': 'unknown'})

    def test_pdf_extraction(self):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=100)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 14 Tf 20 50 Td (PocketMemory PDF evidence) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        output = io.BytesIO(); writer.write(output)
        item, _ = self.store.add(content=output.getvalue(), name='test.pdf')
        self.assertIn('PocketMemory PDF evidence', item['text'])

    def test_image_and_scanned_pdf_ocr(self):
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new('RGB', (1000, 160), 'white')
        font = ImageFont.load_default(size=48)
        ImageDraw.Draw(image).text((30, 45), 'PocketMemory library 2026', fill='black', font=font)
        for format, page in [('PNG', None), ('PDF', 1)]:
            with self.subTest(format=format):
                output = io.BytesIO()
                image.save(output, format=format)
                item, _ = self.store.add(content=output.getvalue(), name='test.' + format.lower())
                self.assertIn('library', item['draft']['summary'].lower(), item['status'])
                self.assertEqual(self.store.get(item['id'], detail=True)['ocr'][0]['page'], page)


if __name__ == '__main__':
    unittest.main()
