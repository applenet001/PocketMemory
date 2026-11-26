"""Server-owned identities, hashed API keys, and revocable browser sessions."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

SESSION_SECONDS = 7 * 24 * 60 * 60


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


class Accounts:
    def __init__(self, root, owner_key):
        self.root = Path(root)
        self.db = self.root / 'accounts.sqlite3'
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL, key_hash TEXT UNIQUE NOT NULL, active INTEGER NOT NULL, created REAL NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS sessions (hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires REAL NOT NULL)')
            # The pre-existing shared inbox belongs to its owner, never a newly created user.
            db.execute('INSERT OR IGNORE INTO users VALUES (?,?,?,?,?,?)',
                       ('owner', '管理员', 'admin', digest(owner_key), 1, time.time()))
            db.execute('DELETE FROM sessions WHERE expires <= ?', (time.time(),))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def public(row):
        return {key: row[key] for key in ('id', 'name', 'role', 'active', 'created')} if row else None

    def by_key(self, key):
        if not isinstance(key, str) or not 32 <= len(key) <= 256:
            return None
        with self.connect() as db:
            return self.public(db.execute('SELECT * FROM users WHERE key_hash=? AND active=1', (digest(key),)).fetchone())

    def by_session(self, session):
        if not isinstance(session, str) or not 32 <= len(session) <= 256:
            return None
        with self.connect() as db:
            return self.public(db.execute('SELECT u.* FROM users u JOIN sessions s ON s.user_id=u.id WHERE s.hash=? AND s.expires>? AND u.active=1', (digest(session), time.time())).fetchone())

    def login(self, key):
        user = self.by_key(key)
        if not user:
            return None
        session = secrets.token_urlsafe(32)
        with self.connect() as db:
            # Recheck under the write transaction so a concurrent revocation cannot restore access.
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM users WHERE id=? AND active=1 AND key_hash=?', (user['id'], digest(key))).fetchone()
            if not row:
                return None
            db.execute('DELETE FROM sessions WHERE expires<=?', (time.time(),))
            db.execute('INSERT INTO sessions VALUES (?,?,?)', (digest(session), user['id'], time.time() + SESSION_SECONDS))
        return user, session

    def logout(self, session):
        with self.connect() as db:
            db.execute('DELETE FROM sessions WHERE hash=?', (digest(session),))

    def list(self):
        with self.connect() as db:
            return [self.public(row) for row in db.execute('SELECT * FROM users ORDER BY created')]

    def create(self, name):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
            raise ValueError('用户名称需为 1–80 个字符')
        key, identity = secrets.token_urlsafe(32), uuid.uuid4().hex
        with self.connect() as db:
            db.execute('INSERT INTO users VALUES (?,?,?,?,?,?)', (identity, name.strip(), 'user', digest(key), 1, time.time()))
        return self.by_key(key), key

    def manage(self, identity, action):
        if identity == 'owner' or action not in {'rotate', 'disable', 'enable'}:
            raise ValueError('不支持此账户操作')
        key = secrets.token_urlsafe(32) if action == 'rotate' else None
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT id FROM users WHERE id=?', (identity,)).fetchone():
                raise ValueError('用户不存在')
            if key:
                db.execute('UPDATE users SET key_hash=? WHERE id=?', (digest(key), identity))
            else:
                db.execute('UPDATE users SET active=? WHERE id=?', (int(action == 'enable'), identity))
            db.execute('DELETE FROM sessions WHERE user_id=?', (identity,))
        return key


def graph_namespace(identity):
    # Matches OriginalHGPLayer's id_for(graph_namespace, user_id).
    import json
    return 'hgp_' + hashlib.sha256(json.dumps(('iphone-shares-v1', identity), ensure_ascii=False).encode()).hexdigest()
