"""Shared label and data helpers for HGP3-2200."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['semantic', 'episodic', 'procedural']


def read(path):
    p = Path(path)
    return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()] if p.exists() else []


def write(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalized(text):
    return re.sub(r'\s+', '', text).casefold()
