"""Multi-user LAN companion with per-user stores and server-enforced isolation."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import mimetypes
from pathlib import Path
import re
import secrets
import socket
import ssl
import sqlite3
import sys
import threading
import queue
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
import uuid
from contextlib import contextmanager
from graph import LocalGraph, PERSONAL_VIEWS
from local_memory import LocalMemories
from accounts import Accounts, graph_namespace
from experiment_profile import (ExperimentConfigUnavailable, evaluated_user_names, evaluated_users,
                                experiment_namespace, read_profile)
from portrait import experiment_evidence, generate_portrait, personal_evidence
from experiment_answer import SearchAnswers, answer_question
from semantic_search import MAX_QUERY_LENGTH, search_meaning
from memory_draft import (MEMORY_LABELS, make_draft, memory_body, memory_evidence, memory_labels,
                          ocr_lines, readable_lines)
from classifier import InboxClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
MAX_BODY = 30 * 1024 * 1024
MAX_FILE = 20 * 1024 * 1024
MEMORY_COLUMNS = 'id,created,title,text,note,url,name,mime,status,classification,graph_state,processing,source_app,draft,feedback'


def shared_image_name(content):
    if content.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'iPhone.png'
    if content.startswith(b'\xff\xd8\xff'):
        return 'iPhone.jpg'
    if content[4:8] == b'ftyp' and content[8:12] in {b'heic', b'heix', b'mif1'}:
        return 'iPhone.heic'
    return ''


def decode_shortcut_share(encoded, name=''):
    if not isinstance(encoded, str) or not encoded.strip():
        raise ValueError('没有收到分享内容。请选择「Base64 编码」的输出作为 shared_base64 的值。')
    if not isinstance(name, str):
        raise ValueError('文件名必须是文本。')
    try:
        content = base64.b64decode(''.join(encoded.split()), validate=True)
    except ValueError as exc:
        raise ValueError('无法读取分享内容。请检查 Base64 动作是否设为「编码」，然后一次分享一项。') from exc
    if len(content) > MAX_FILE:
        raise ValueError('单个文件不能超过 20 MB')
    image_name = shared_image_name(content)
    if image_name or content.startswith(b'%PDF-'):
        return {'name': name or image_name or 'iPhone.pdf', 'content': content}
    if name:
        return {'name': name, 'content': content}
    try:
        plain_text = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        plain_text = ''
    if plain_text and all(char >= ' ' or char in '\t\r\n' for char in plain_text):
        if len(plain_text) <= 200000:
            return {'text': plain_text}
        return {'name': 'iPhone.txt', 'content': content}
    return {'name': 'iPhone.bin', 'content': content}


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'memories.sqlite3'
        self.ocr_lock = threading.Lock()
        self.graph_lock = threading.Lock()
        self.ocr = None
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, created TEXT, title TEXT, text TEXT, note TEXT, url TEXT, name TEXT, mime TEXT, status TEXT, digest TEXT UNIQUE, content BLOB)')
            db.execute('BEGIN IMMEDIATE')
            columns = {column['name'] for column in db.execute('PRAGMA table_info(memories)')}
            for column in ('classification', 'graph_state', 'processing', 'source_app', 'draft', 'ocr', 'feedback'):
                if column not in columns:
                    db.execute(f'ALTER TABLE memories ADD COLUMN {column} TEXT')
            db.execute('CREATE TABLE IF NOT EXISTS memory_feedback (memory_id TEXT, revision INTEGER, event TEXT NOT NULL, PRIMARY KEY (memory_id, revision))')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def recognize(self, image, page=None):
        import numpy as np
        from rapidocr import RapidOCR

        image.thumbnail((2400, 2400))
        pixels = np.array(image.convert('RGB'))[:, :, ::-1].copy()
        with self.ocr_lock:
            if self.ocr is None:
                self.ocr = RapidOCR()
            return ocr_lines(self.ocr(pixels, text_score=0.0), page)

    def extract_pdf(self, content):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages, blocks, scanned = [], [], 0
        for number, page in enumerate(reader.pages[:100], start=1):
            text = '\n'.join(readable_lines((page.extract_text() or '')[:20000]))
            if not text and scanned < 8:
                scanned += 1
                try:
                    page_blocks = []
                    for image in page.images[:4]:
                        if image.image is not None:
                            page_blocks.extend(self.recognize(image.image, page=number))
                    blocks.extend(page_blocks)
                    text = '\n'.join(block['text'] for block in page_blocks if block['accepted'])
                except Exception as error:
                    logging.warning('Scanned PDF image OCR unavailable (%s)', type(error).__name__)
            pages.append(text)
        text = '\n'.join(filter(None, pages))[:200000]
        status = '已提取 PDF 文本（最多前 100 页，其中最多 8 页运行 OCR）' if text else 'PDF 暂无可用正文，可补充说明'
        return text, status, blocks

    def extract(self, content, name):
        suffix = Path(name).suffix.lower()
        if suffix in {'.txt', '.md', '.csv', '.json', '.log'}:
            encoding = 'utf-16' if content.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
            return '\n'.join(readable_lines(content.decode(encoding, errors='replace')[:200000])), '已读取文本文件', []
        if suffix == '.pdf':
            try:
                return self.extract_pdf(content)
            except ImportError:
                return '', 'PDF 提取依赖暂不可用，原件已保存', []
            except Exception:
                return '', 'PDF 提取失败或文件已加密，原件已保存', []
        if suffix in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
            try:
                from PIL import Image, ImageOps
                with Image.open(io.BytesIO(content)) as image:
                    blocks = self.recognize(ImageOps.exif_transpose(image))
                text = '\n'.join(block['text'] for block in blocks if block['accepted'])[:200000]
                return text, '已完成本地 OCR，可核对识别结果' if text else '图片暂无可靠文字，可补充说明', blocks
            except ImportError:
                return '', '图片已保存，OCR 依赖暂不可用', []
            except Exception:
                return '', '图片识别失败，原件已保存，可补充说明', []
        return '', '原件已保存，此格式暂无正文提取', []

    def add(self, text='', note='', content=None, name='', title='', source_app='', defer=False):
        text, note, title = str(text).strip(), str(note).strip(), str(title).strip()
        if not isinstance(source_app, str) or len(source_app.strip()) > 120:
            raise ValueError('来源 App 必须是 120 字以内的文本')
        source_app = source_app.strip()
        if len(text) > 200000 or len(note) > 10000 or len(title) > 300:
            raise ValueError('文字、备注或标题过长')
        if not text and not content:
            raise ValueError('请分享文字、链接或文件')
        if content and len(content) > MAX_FILE:
            raise ValueError('单个文件不能超过 20 MB')
        name = name.replace('\\', '/').split('/')[-1][:180]
        if content and not name:
            name = 'attachment.bin'
        parts = [text, note, name, source_app] if source_app else [text, note, name]
        digest = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode() + (content or b'')).hexdigest()
        with self.connect() as db:
            row = db.execute('SELECT id FROM memories WHERE digest=?', (digest,)).fetchone()
        if row:
            return self.get(row['id']), False
        extracted, status, blocks = (self.extract(content, name) if content and not defer else
                                     ('', '原件已保存，等待后台提取' if content else '已保存分享内容', []))
        url = text if re.fullmatch(r'https?://\S+', text) else ''
        if url:
            status = '链接已保存；未自动抓取网页正文'
        item = dict(id=uuid.uuid4().hex, created=datetime.now(timezone.utc).isoformat(),
                    title=title or name or text[:80], text=text, note=note, url=url, source_app=source_app,
                    name=name, mime=mimetypes.guess_type(name)[0] or 'application/octet-stream',
                    status=status, digest=digest)
        draft = None if defer and content else make_draft(item, extracted)
        item['text'] = '\n'.join(filter(None, (text, extracted)))[:200000]
        with self.connect() as db:
            cursor = db.execute('''INSERT OR IGNORE INTO memories
                (id,created,title,text,note,url,name,mime,status,digest,content,processing,source_app,draft,ocr)
                VALUES (:id,:created,:title,:text,:note,:url,:name,:mime,:status,:digest,:content,:processing,:source_app,:draft,:ocr)''',
                {**item, 'content': content, 'draft': json.dumps(draft) if draft else None, 'ocr': json.dumps(blocks) if blocks else None,
                 'processing': 'extract' if defer and content else 'queued' if defer else None})
            inserted = cursor.rowcount == 1
            row = db.execute('SELECT id FROM memories WHERE digest=?', (digest,)).fetchone()
        return self.get(row['id']), inserted

    def get(self, identity, file=False, detail=False):
        columns = '*' if file else MEMORY_COLUMNS + (',ocr' if detail else '')
        with self.connect() as db:
            row = db.execute(f'SELECT {columns} FROM memories WHERE id=?', (identity,)).fetchone()
            return self.decode(row) if row else None

    @staticmethod
    def decode(row):
        record = dict(row)
        for field in ('classification', 'draft', 'feedback', 'ocr'):
            if record.get(field):
                record[field] = json.loads(record[field])
        return record

    def classify(self, item, classifier):
        previous = item['classification']
        retry_review = previous and previous.get('uncertain') and previous.get('fallback_status') != 'reviewed'
        if classifier is None or (previous and previous['status'] != 'failed' and not retry_review):
            return item
        result = (classifier(item, self.feedback_examples()) if isinstance(classifier, InboxClassifier)
                  else classifier(item))
        with self.connect() as db:
            db.execute('UPDATE memories SET classification=? WHERE id=? AND classification IS ?',
                       (json.dumps(result), item['id'], json.dumps(previous) if previous else None))
        return self.get(item['id'])

    def record_feedback(self, identity, payload):
        if not isinstance(payload, dict) or 'label' not in payload or payload['label'] not in (*MEMORY_LABELS, None):
            raise ValueError('请选择有效的记忆标签')
        label = payload['label']
        additional = payload.get('additional_labels', [])
        if (not isinstance(additional, list) or len(additional) > len(MEMORY_LABELS)
                or any(value not in MEMORY_LABELS for value in additional) or (additional and label is None)):
            raise ValueError('请选择主标签和有效的附加标签')
        labels = [label, *(value for value in MEMORY_LABELS if value != label and value in additional)] if label else []
        if type(payload.get('working_state')) is not bool:
            raise ValueError('请选择是否需要跟进')
        text = payload.get('text', '')
        if not isinstance(text, str) or len(text) > 10000:
            raise ValueError('更正正文最多 10000 字')
        correction = {'label': label, 'labels': labels, 'working_state': payload['working_state'], 'text': text.strip()}
        with self.graph_lock, self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute(f'SELECT {MEMORY_COLUMNS} FROM memories WHERE id=?', (identity,)).fetchone()
            if row is None:
                return None
            item = self.decode(row)
            previous = item['feedback'] or {}
            previous_values = {**previous, 'labels': memory_labels(previous)} if previous else {}
            if all(previous_values.get(key) == value for key, value in correction.items()):
                return item
            feedback = {**correction, 'revision': previous.get('revision', 0) + 1,
                        'created': datetime.now(timezone.utc).isoformat()}
            result = {'status': 'classified', 'source': 'user', 'uncertain': False,
                      'label': label, 'labels': labels, 'working_state': feedback['working_state']}
            graph_state = item['graph_state']
            if self.graph_eligible({'classification': result}):
                graph_state = 'pending'
            elif graph_state:
                graph_state = 'remove_pending'
            db.execute('INSERT INTO memory_feedback VALUES (?, ?, ?)', (identity, feedback['revision'],
                       json.dumps({**feedback, 'previous_classification': item['classification']})))
            db.execute('UPDATE memories SET feedback=?,classification=?,graph_state=? WHERE id=?',
                       (json.dumps(feedback), json.dumps(result), graph_state, identity))
        return self.get(identity)

    def feedback_examples(self):
        with self.connect() as db:
            items = [self.decode(row) for row in db.execute(
                f'SELECT {MEMORY_COLUMNS} FROM memories WHERE feedback IS NOT NULL ORDER BY id')]
        examples = []
        for item in items:
            evidence = memory_evidence(item)
            if evidence:
                examples.append((evidence, item['feedback']['label'], item['feedback']['working_state']))
        return tuple(examples)

    @staticmethod
    def graph_eligible(item):
        result = item['classification'] or {}
        return (result.get('status') == 'classified' and not result.get('uncertain')
                and bool({'semantic', 'episodic'}.intersection(memory_labels(result))))

    def sync_graph(self, identity, graph, namespace):
        with self.graph_lock:
            item = self.get(identity)
            if not item or not graph:
                return item
            removing = item['graph_state'] == 'remove_pending'
            if not removing and (not self.graph_eligible(item) or item['graph_state'] == 'synced'):
                return item
            if not removing:
                with self.connect() as db:
                    db.execute("UPDATE memories SET graph_state='pending' WHERE id=?", (identity,))
                item['graph_state'] = 'pending'
            try:
                if removing:
                    graph.delete_upload(namespace, identity)
                else:
                    graph.write_upload(namespace, item)
            except Exception as error:
                logging.warning('Personal graph sync failed (%s)', type(error).__name__)
                return item
            item['graph_state'] = None if removing else 'synced'
            with self.connect() as db:
                db.execute('UPDATE memories SET graph_state=? WHERE id=?', (item['graph_state'], identity))
            return item

    def list(self, query=''):
        with self.connect() as db:
            items = [self.decode(row) for row in db.execute(f'SELECT {MEMORY_COLUMNS} FROM memories ORDER BY created DESC')]
        return self.search(items, query)

    @staticmethod
    def profile(items):
        groups = {kind: [] for kind in ('semantic', 'episodic', 'procedural', 'working')}
        for item in items:
            result = item['classification']
            if not result or result['status'] != 'classified':
                continue
            kinds = memory_labels(result).copy()
            if result['working_state']:
                kinds.append('working')
            for kind in kinds:
                if kind in groups and len(groups[kind]) < 3:
                    groups[kind].append(item)
        return groups

    @staticmethod
    def search(items, query):
        if not query.strip():
            return items
        terms = re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]{2}|[\u4e00-\u9fff]', query.lower())
        ranked = []
        for item in items:
            hay = ' '.join(filter(None, (item['title'], item['url'], item['source_app'], item['note'], memory_body(item)))).lower()
            score = sum(1 for term in terms if term in hay)
            if score:
                ranked.append((score, item))
        return [item for _, item in sorted(ranked, key=lambda pair: pair[0], reverse=True)]

    def delete(self, identity):
        with self.connect() as db:
            deleted = db.execute('DELETE FROM memories WHERE id=?', (identity,)).rowcount > 0
            db.execute('DELETE FROM memory_feedback WHERE memory_id=?', (identity,))
        self.sync_local()
        return deleted

    def sync_local(self):
        LocalMemories(self.root / 'hgp' / 'uploads').write_uploads(self.list())

    def workflows(self):
        reader = LocalMemories(self.root / 'hgp')
        records = {item['id'] for item in self.list()}
        groups, rules = [], []
        for source in reader.sources('working'):
            for group in reader.read('working', source['id'])['groups']:
                for node in group['nodes']:
                    identity = node.get('source_episode_id')
                    node['upload_id'] = identity if identity in records else None
                groups.append(group)
        for source in reader.sources('procedural'):
            for rule in reader.read('procedural', source['id'])['rules']:
                identity = (rule.get('meta') or {}).get('source_id')
                rule['upload_id'] = identity if identity in records else None
                rules.append(rule)
        groups.sort(key=lambda group: max((node.get('timestamp') or '' for node in group['nodes']), default=''), reverse=True)
        rules.sort(key=lambda rule: rule.get('valid_from') or '', reverse=True)
        return {'working': groups, 'procedural': rules}


def make_server(store, token, host='127.0.0.1', port=8766, secure=False, classifier=None, upload_graph=None, background=True):
    answers = SearchAnswers()
    accounts = Accounts(store.root, token)
    stores = {'owner': store}
    stores_lock = threading.Lock()
    jobs = queue.Queue()
    scheduled = set()
    jobs_lock = threading.Lock()

    def process(personal, identity, user_id):
        item = personal.get(identity, file=True)
        if not item:
            return
        if item['processing'] == 'extract':
            text, status, blocks = personal.extract(item['content'], item['name'])
            draft = make_draft(item, text)
            with personal.connect() as db:
                db.execute("""UPDATE memories SET text=?,status=?,draft=?,ocr=?,processing='queued',
                           graph_state=CASE WHEN feedback IS NOT NULL AND graph_state='synced' THEN 'pending' ELSE graph_state END
                           WHERE id=? AND processing='extract'""",
                           ('\n'.join(filter(None, (item['text'], text)))[:200000], status, json.dumps(draft),
                            json.dumps(blocks) if blocks else None, identity))
        with personal.connect() as db:
            db.execute("UPDATE memories SET processing='classifying' WHERE id=?", (identity,))
        item = personal.get(identity)
        if item:
            personal.classify(item, classifier)
        with personal.graph_lock:
            personal.sync_local()
        with personal.connect() as db:
            db.execute("UPDATE memories SET processing='syncing' WHERE id=?", (identity,))
        personal.sync_graph(identity, upload_graph, graph_namespace(user_id))
        item = personal.get(identity)
        with personal.connect() as db:
            db.execute('UPDATE memories SET processing=? WHERE id=?',
                       ('failed' if item and (item['classification'] or {}).get('status') == 'failed' else 'complete', identity))

    def worker():
        while True:
            job = jobs.get()
            if job is None:
                jobs.task_done()
                return
            personal, identity, user_id = job
            try:
                process(personal, identity, user_id)
            except Exception as error:
                logging.warning('Memory processing failed (%s)', type(error).__name__)
                with personal.connect() as db:
                    # Preserve extraction stage so a restart/retry still extracts the original.
                    db.execute("UPDATE memories SET processing=CASE WHEN processing='extract' THEN 'extract' ELSE 'failed' END WHERE id=?", (identity,))
            finally:
                with jobs_lock:
                    scheduled.discard((str(personal.root), identity))
                jobs.task_done()

    def schedule(personal, identity, user_id):
        if not background:
            process(personal, identity, user_id)
        else:
            key = (str(personal.root), identity)
            with jobs_lock:
                if key not in scheduled:
                    scheduled.add(key)
                    jobs.put((personal, identity, user_id))
        return personal.get(identity)

    if background:
        processing_thread = threading.Thread(target=worker, daemon=True, name='memory-processing')
        processing_thread.start()

    def user_store(user):
        with stores_lock:
            if user['id'] not in stores:
                stores[user['id']] = Store(store.root / 'users' / user['id'])
                for item in stores[user['id']].list():
                    schedule(stores[user['id']], item['id'], user['id'])
            return stores[user['id']]

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(90)

        def log_message(self, fmt, *args):
            # Do not log shared content, credentials, or query strings.
            pass

        def send(self, status, value, mime='application/json; charset=utf-8', extra=None):
            data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            for key, val in {'Content-Type': mime, 'Content-Length': str(len(data)), 'Cache-Control': 'no-store',
                             'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer',
                             'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'", **(extra or {})}.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(data)

        def send_share_result(self, status, value):
            message = value['message'] if 'message' in value else value['error']
            self.send(status, message.encode('utf-8'), 'text/plain; charset=utf-8')

        def session_cookie(self):
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get('Cookie', ''))
                return cookie['pm_session'].value if 'pm_session' in cookie else ''
            except Exception:
                return ''

        def authorized(self):
            header = self.headers.get('Authorization')
            self.user = accounts.by_key(header[7:]) if header and header.startswith('Bearer ') else accounts.by_session(self.session_cookie()) if not header else None
            return self.user is not None

        def cookie_header(self, session, clear=False):
            return f'pm_session={session}; HttpOnly; SameSite=Strict; Path=/' + ('; Secure' if secure else '') + ('; Max-Age=0' if clear else '')

        def allowed_origin(self):
            origin = self.headers.get('Origin')
            return not origin or origin == ('https://' if secure else 'http://') + self.headers.get('Host', '')

        def read_body(self):
            lengths = self.headers.get_all('Content-Length', [])
            encodings = self.headers.get_all('Transfer-Encoding', [])
            if encodings:
                self.close_connection = True
                if lengths:
                    raise ValueError('请求长度标头冲突，请移除手动添加的 Content-Length 或 Transfer-Encoding。')
                if [encoding.strip().lower() for encoding in encodings] != ['chunked']:
                    raise ValueError('不支持此上传传输格式。')
                return self.read_chunked_body()
            if not lengths:
                return b''
            if len(lengths) != 1 or not re.fullmatch(r'[0-9]{1,18}', lengths[0].strip()):
                self.close_connection = True
                raise ValueError('请求的 Content-Length 无效。')
            length = int(lengths[0])
            if length > MAX_BODY:
                raise ValueError('请求过大；单文件上限 20 MB')
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('上传未完成')
            return body

        def read_chunked_body(self):
            body = bytearray()
            while True:
                line = self.rfile.readline(8193)
                chunk_size = line[:-2].partition(b';')[0].rstrip(b' \t')
                if len(line) > 8192 or not line.endswith(b'\r\n') or not re.fullmatch(rb'[0-9A-Fa-f]{1,16}', chunk_size):
                    raise ValueError('上传分段格式无效。')
                size = int(chunk_size, 16)
                if size == 0:
                    break
                if size > MAX_BODY - len(body):
                    raise ValueError('请求过大；单文件上限 20 MB')
                data = self.rfile.read(size + 2)
                if len(data) != size + 2 or not data.endswith(b'\r\n'):
                    raise ValueError('上传未完成，请重新分享。')
                body.extend(memoryview(data)[:-2])
            remaining = 8192
            while remaining > 0:
                trailer = self.rfile.readline(remaining)
                if trailer == b'\r\n':
                    return bytes(body)
                if not trailer.endswith(b'\r\n'):
                    break
                remaining -= len(trailer)
            raise ValueError('上传分段缺少有效的结束标记。')

        def do_GET(self):
            parsed = urlsplit(self.path)
            assets = {'/': ('index.html','text/html; charset=utf-8'), '/app.js': ('app.js','text/javascript; charset=utf-8'), '/style.css': ('style.css','text/css; charset=utf-8')}
            if parsed.path in assets:
                filename, mime = assets[parsed.path]
                return self.send(200, (ROOT / 'web' / filename).read_bytes(), mime)
            if not self.authorized():
                return self.send(401, {'error': '请使用自己的访问密钥登录'})
            personal = user_store(self.user)
            if parsed.path == '/api/me':
                return self.send(200, {'user': self.user})
            if parsed.path == '/api/admin/users':
                if self.user['role'] != 'admin':
                    return self.send(403, {'error': '仅管理员可以管理用户'})
                return self.send(200, {'users': accounts.list()})
            if parsed.path in {'/api/admin/experiment-profile', '/api/admin/experiment-search',
                               '/api/admin/experiment-portrait'}:
                if self.user['role'] != 'admin':
                    return self.send(403, {'error': '仅管理员可以查看实验记忆'})
                params = parse_qs(parsed.query)
                selected = params.get('user', [''])[0]
                search = parsed.path.endswith('-search')
                portrait = parsed.path.endswith('-portrait')
                query = params.get('q', [''])[0].strip()
                if search and (not selected or not query or len(query) > 160):
                    return self.send(400, {'error': '请选择评测用户，并输入 1 到 160 个字符的问题'})
                if portrait and not selected:
                    return self.send(400, {'error': '请选择评测用户'})
                if not selected:
                    users = evaluated_users()
                    return self.send(200, {'users': users, 'names': evaluated_user_names(users)})
                try:
                    if search:
                        profile = read_profile(selected, query)
                        context = profile.pop('qa_context', None)
                        ticket = answers.remember(self.user['id'], query, context)
                        return self.send(200, {'profile': profile, 'answer_ticket': ticket})
                    if portrait:
                        evidence = experiment_evidence(read_profile(selected))
                        return self.send(200, {'portrait': generate_portrait('experiment:' + selected, evidence)})
                    return self.send(200, {'profile': read_profile(selected)})
                except ValueError:
                    return self.send(400, {'error': '实验用户无效或记录格式异常'})
                except ExperimentConfigUnavailable as exc:
                    return self.send(503, {'error': str(exc)})
                except (OSError, RuntimeError):
                    return self.send(503, {'error': '实验记忆暂不可用，请检查电脑端的实验库和检索配置。'})
                except Exception as error:
                    logging.warning('Experiment retrieval unavailable (%s)', type(error).__name__)
                    return self.send(503, {'error': '实验记忆暂不可用，请稍后重试。'})
            if parsed.path == '/api/memories':
                params = parse_qs(parsed.query)
                query = params.get('q', [''])[0]
                semantic = params.get('mode', [''])[0] == 'semantic' and bool(query.strip())
                if semantic and len(query.strip()) > MAX_QUERY_LENGTH:
                    return self.send(400, {'error': f'按意思搜索最多支持 {MAX_QUERY_LENGTH} 个字符'})
                items = personal.list()
                fallback = False
                if semantic:
                    try:
                        matches = search_meaning(items, query.strip())
                    except Exception as error:
                        logging.warning('Personal semantic search unavailable (%s)', type(error).__name__)
                        fallback = True
                if not semantic or fallback:
                    matches = personal.search(items, query)
                return self.send(200, {'items': matches, 'profile': personal.profile(items),
                                       'user': self.user, 'search_fallback': fallback})
            memory_detail = re.fullmatch(r'/api/memories/([a-f0-9]{32})', parsed.path)
            if memory_detail:
                item = personal.get(memory_detail[1], detail=True)
                return self.send(200, {'item': item}) if item else self.send(404, {'error': '记忆不存在'})
            if parsed.path == '/api/export':
                return self.send(200, {'schema': 'pocketmemory.share.v1', 'user_id': self.user['id'], 'items': personal.list()}, extra={'Content-Disposition': 'attachment; filename="pocketmemory.json"'})
            if parsed.path == '/api/workflows':
                try:
                    return self.send(200, personal.workflows())
                except (ValueError, TypeError, KeyError, OSError):
                    return self.send(503, {'error': '任务与流程暂时无法读取，请稍后刷新。'})
            if parsed.path == '/api/local-memory':
                params = parse_qs(parsed.query)
                kind = params.get('kind', ['working'])[0]
                identity = params.get('source', [''])[0]
                try:
                    reader = LocalMemories() if self.user['role'] == 'admin' else LocalMemories(personal.root / 'hgp')
                    return self.send(200, reader.read(kind, identity) if identity else {'sources': reader.sources(kind)})
                except (ValueError, TypeError, KeyError):
                    return self.send(400, {'error': '来源无效或记忆文件格式不正确，请检查原始数据。'})
                except OSError:
                    return self.send(503, {'error': '记忆文件暂时无法读取，请稍后重试。'})
            if parsed.path == '/api/graph':
                params = parse_qs(parsed.query)
                try:
                    namespace = params.get('namespace', [''])[0]
                    database = params.get('database', ['episodic'])[0]
                    experiment_user = params.get('user', [''])[0]
                    if database not in PERSONAL_VIEWS and self.user['role'] != 'admin':
                        return self.send(403, {'error': '仅管理员可以查看实验图'})
                    allowed = graph_namespace(self.user['id']) if database in PERSONAL_VIEWS else None
                    if experiment_user:
                        if self.user['role'] != 'admin' or database in PERSONAL_VIEWS:
                            return self.send(403, {'error': '这个视图不支持指定实验用户'})
                        allowed = experiment_namespace(experiment_user)
                        namespace = namespace or allowed
                    if allowed and namespace and namespace != allowed:
                        return self.send(403, {'error': '不能访问其他用户的数据空间'})
                    graph = LocalGraph().view(database, namespace, allowed_namespace=allowed)
                    return self.send(200, graph)
                except ExperimentConfigUnavailable as exc:
                    return self.send(503, {'error': str(exc)})
                except ValueError as exc:
                    return self.send(400, {'error': str(exc)})
                except Exception:
                    return self.send(503, {'error': 'Neo4j 暂不可用，请检查本机实例、所选数据库和凭据。分享保存仍可使用。'})
            match = re.fullmatch(r'/api/files/([a-f0-9]{32})', parsed.path)
            if match:
                item = personal.get(match[1], file=True)
                if not item or item['content'] is None:
                    return self.send(404, {'error': '原件不存在'})
                # Never execute HTML/SVG shared by another app on this origin.
                suffix = Path(item['name']).suffix
                suffix = suffix if re.fullmatch(r'\.[a-zA-Z0-9]{1,12}', suffix) else '.bin'
                return self.send(200, item['content'], 'application/octet-stream', {'Content-Disposition': 'attachment; filename="' + item['id'] + suffix + '"'})
            return self.send(404, {'error': '不存在'})

        def do_POST(self):
            path = urlsplit(self.path).path
            image_upload = path == '/api/memories/image'
            respond = self.send_share_result if image_upload else self.send
            if not self.allowed_origin():
                return respond(403, {'error': '不允许跨站请求'})
            try:
                if path == '/api/login':
                    value = json.loads(self.read_body())
                    login = accounts.login(value.get('token', ''))
                    if not login:
                        return self.send(401, {'error': '访问密钥不正确'})
                    user, session = login
                    return self.send(200, {'ok': True, 'user': user}, extra={'Set-Cookie': self.cookie_header(session)})
                if path == '/api/logout':
                    accounts.logout(self.session_cookie())
                    return self.send(200, {'ok': True}, extra={'Set-Cookie': self.cookie_header('', clear=True)})
                if not self.authorized():
                    return respond(401, {'error': '未授权'})
                if path == '/api/portrait':
                    if self.user['role'] != 'user':
                        return self.send(403, {'error': '请使用个人账号生成自己的画像'})
                    payload = json.loads(self.read_body())
                    if not isinstance(payload, dict) or payload.get('consent') is not True:
                        return self.send(400, {'error': '请确认使用记忆片段生成画像'})
                    personal = user_store(self.user)
                    items = personal.list()
                    evidence = personal_evidence(personal.profile(items), items)
                    return self.send(200, {'portrait': generate_portrait('user:' + self.user['id'], evidence)})
                if path == '/api/admin/experiment-answer':
                    if self.user['role'] != 'admin':
                        return self.send(403, {'error': '仅管理员可以使用实验问答'})
                    payload = json.loads(self.read_body())
                    ticket = payload.get('ticket') if isinstance(payload, dict) else None
                    if not isinstance(ticket, str) or not re.fullmatch(r'[A-Za-z0-9_-]{32}', ticket):
                        return self.send(400, {'error': '缺少有效的检索凭证'})
                    saved = answers.get(self.user['id'], ticket)
                    if saved is None:
                        return self.send(404, {'error': '这次搜索已过期，请重新搜索。'})
                    question, context = saved
                    return self.send(200, {'answer': answer_question(question, context)})
                match = re.fullmatch(r'/api/memories/([a-f0-9]{32})/(classify|graph-sync)', path)
                if match:
                    personal = user_store(self.user)
                    item = personal.get(match[1])
                    if item:
                        item = schedule(personal, item['id'], self.user['id'])
                    if not item:
                        return self.send(404, {'error': '记忆不存在'})
                    return self.send(200, {'item': item})
                match = re.fullmatch(r'/api/memories/([a-f0-9]{32})/feedback', path)
                if match:
                    personal = user_store(self.user)
                    if not personal.record_feedback(match[1], json.loads(self.read_body())):
                        return self.send(404, {'error': '记忆不存在'})
                    with personal.graph_lock:
                        personal.sync_local()
                    personal.sync_graph(match[1], upload_graph, graph_namespace(self.user['id']))
                    item = personal.get(match[1], detail=True)
                    return self.send(200, {'item': item}) if item else self.send(404, {'error': '记忆不存在'})
                if path == '/api/admin/users':
                    if self.user['role'] != 'admin':
                        return self.send(403, {'error': '仅管理员可以创建用户'})
                    value = json.loads(self.read_body())
                    user, key = accounts.create(value.get('name'))
                    return self.send(201, {'user': user, 'key': key})
                match = re.fullmatch(r'/api/admin/users/([a-f0-9]{32})/(rotate|disable|enable)', path)
                if match:
                    if self.user['role'] != 'admin':
                        return self.send(403, {'error': '仅管理员可以管理用户'})
                    key = accounts.manage(match[1], match[2])
                    return self.send(200, {'ok': True, 'key': key})
                if path != '/api/memories' and not image_upload:
                    return self.send(404, {'error': '不存在'})
                body = self.read_body()
                if not body:
                    if image_upload:
                        raise ValueError('请将请求体设为「文件」，值选「快捷指令输入」，再分享一张截图或照片。')
                    raise ValueError('请求没有带上传内容。请检查快捷指令的「表单」，并确认 file 或 text 字段已选择要分享的内容。')
                name = shared_image_name(body)
                if image_upload or name:
                    if not name:
                        raise ValueError('没有收到可识别的图片。请将请求体设为「文件」，值选「快捷指令输入」，再分享一张截图或照片。')
                    respond = self.send_share_result
                    values = {'name': name, 'content': body, 'source_app': parse_qs(urlsplit(self.path).query).get('source_app', [''])[0]}
                elif self.headers.get('Content-Type', '').lower().startswith('multipart/form-data'):
                    message = BytesParser(policy=policy.default).parsebytes(('Content-Type: ' + self.headers['Content-Type'] + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + body)
                    if not message.is_multipart():
                        raise ValueError('上传表单格式无效。请检查「请求体 → 表单」里的 file 字段，并让快捷指令自动生成 Content-Type。')
                    values, files = {}, []
                    for part in message.iter_parts():
                        field = part.get_param('name', header='content-disposition')
                        filename = part.get_filename()
                        data = part.get_payload(decode=True) or b''
                        if filename is not None:
                            files.append((filename, data))
                        elif field in {'text', 'note', 'title', 'file', 'source_app'}:
                            values['text' if field == 'file' else field] = data.decode('utf-8', errors='replace')
                    if len(files) > 1:
                        raise ValueError('每次分享一个文件；多个文件请在快捷指令中逐项重复')
                    if files:
                        values.update(name=files[0][0], content=files[0][1])
                else:
                    try:
                        payload = json.loads(body)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise ValueError('无法读取请求体。请在快捷指令中选择「表单」，用 file 上传原件，或用 text 分享文字或链接。') from exc
                    if not isinstance(payload, dict):
                        raise ValueError('链接或文字请使用带有 text 字段的 JSON 对象。')
                    if 'shared_base64' in payload:
                        respond = self.send_share_result
                        values = decode_shortcut_share(payload['shared_base64'], payload.get('name', ''))
                    else:
                        values = {key: payload[key] for key in ['text', 'note', 'title', 'name'] if key in payload}
                        if payload.get('base64'):
                            values['content'] = base64.b64decode(payload['base64'], validate=True)
                    values['source_app'] = payload.get('source_app', '')
                personal = user_store(self.user)
                item, created = personal.add(**values, defer=background)
                if item:
                    item = schedule(personal, item['id'], self.user['id'])
                if not item:
                    return respond(404, {'error': '这条记忆已被删除'})
                return respond(201 if created else 200, {'item': item, 'created': created, 'message': '已保存到 PocketMemory' if created else '这条记忆已存在'})
            except (ValueError, TypeError, KeyError) as exc:
                return respond(400, {'error': str(exc)})
            except Exception:
                return respond(500, {'error': '保存失败，请重试；未返回成功前不要关闭分享'})

        def do_DELETE(self):
            if not self.allowed_origin() or not self.authorized():
                return self.send(403, {'error': '未授权'})
            match = re.fullmatch(r'/api/memories/([a-f0-9]{32})', urlsplit(self.path).path)
            if not match:
                return self.send(404, {'error': '不存在'})
            personal = user_store(self.user)
            with personal.graph_lock:
                item = personal.get(match[1])
                if upload_graph and item and item['graph_state'] in {'synced', 'pending', 'remove_pending'}:
                    try:
                        upload_graph.delete_upload(graph_namespace(self.user['id']), item['id'])
                    except Exception:
                        return self.send(503, {'error': 'Neo4j 暂不可用，保留源记录以便稍后重试删除'})
                return self.send(200, {'deleted': personal.delete(match[1])})

    class MemoryServer(ThreadingHTTPServer):
        def server_close(self):
            super().server_close()
            if background:
                jobs.put(None)
                processing_thread.join(timeout=2)

    server = MemoryServer((host, port), Handler)
    if background:
        for user in accounts.list():
            personal = user_store(user)
            if user['id'] == 'owner':
                for item in personal.list():
                    schedule(personal, item['id'], user['id'])
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'runtime')
    parser.add_argument('--tls-cert', type=Path)
    parser.add_argument('--tls-key', type=Path)
    args = parser.parse_args()
    store = Store(args.data_dir)
    key_file = args.data_dir / 'access-key.txt'
    if not key_file.exists():
        key_file.write_text(secrets.token_urlsafe(32), encoding='utf-8')
    token = key_file.read_text(encoding='utf-8').strip()
    if len(token) < 32:
        raise ValueError('Access key must contain at least 32 characters')
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error('--tls-cert and --tls-key must be provided together')
    server = make_server(store, token, args.host, args.port, secure=bool(args.tls_cert),
                         classifier=InboxClassifier(), upload_graph=LocalGraph())
    scheme = 'http'
    if args.tls_cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = 'https'
    addresses = sorted({address[4][0] for address in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
    print(f'PocketMemory: {scheme}://127.0.0.1:{args.port}', flush=True)
    if args.host == '0.0.0.0':
        for address in addresses:
            print(f'iPhone LAN: {scheme}://{address}:{args.port}', flush=True)
    print(f'Administrator key: {token}\nCreate personal keys in User Management. Never share this administrator key.\nPrivate LAN only. Do not forward this port to the internet. Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
