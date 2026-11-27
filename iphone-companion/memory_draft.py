"""Turn extracted evidence into a conservative, locally generated memory draft."""
import math
import re
import unicodedata

MEMORY_LABELS = ('semantic', 'episodic', 'procedural')


def memory_labels(result):
    labels = result.get('labels')
    if labels is not None:
        return labels
    label = result.get('label')
    return [label] if label else []


def readable_lines(text):
    for raw in text.splitlines():
        line = raw.strip()
        if ('\ufffd' not in line and not re.search(r'\(cid:\d+\)', line)
                and not any(unicodedata.category(char) in {'Cc', 'Co'} for char in line if char != '\t')
                and any(char.isalnum() for char in line)):
            yield line


def ocr_lines(result, page=None):
    if result.boxes is None or result.scores is None or result.txts is None:
        return []
    lines = []
    for text, box, score in zip(result.txts, result.boxes, result.scores):
        confidence = float(score)
        readable = ' '.join(readable_lines(text)) if confidence >= .75 else ''
        lines.append({'text': text, 'box': [[round(float(x), 2), round(float(y), 2)] for x, y in box],
                      'confidence': round(confidence, 4) if math.isfinite(confidence) else None,
                      'accepted': bool(readable), 'page': page})
    return lines


def make_draft(item, extracted):
    shared = item['text'] if not item['url'] else ''
    body = '\n'.join(filter(None, (shared, extracted)))[:200000]
    excerpt = '\n'.join(body.strip().splitlines()[:8])
    if len(excerpt) > 800:
        excerpt = excerpt[:799].rstrip() + '…'
    summary = excerpt or ('已保存链接。' if item['url'] else f"已保存附件 {item['name']}。暂无可用正文。")
    return {'summary': summary, 'body': body, 'status': 'ready' if body else 'source_only'}


def memory_body(item):
    feedback = item.get('feedback') or {}
    if feedback.get('text'):
        return feedback['text']
    draft = item.get('draft')
    if draft is not None:
        return draft['body']
    return item['text'].removeprefix(item.get('url') or '').strip() if item.get('url') else item['text']


def memory_evidence(item):
    return '\n'.join(filter(None, (item['note'], memory_body(item))))
