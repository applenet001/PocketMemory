"""Read MobileMem app provenance from the recorded message envelope."""

APP_NAMES = frozenset({'Bill', 'Calendar', 'Document', 'Note', 'Screen Memo', 'Todo', 'Voice Memo'})


def mobilemem_source_app(role, name):
    if role == 'system' and name in APP_NAMES:
        return name
    return None


def todo_completion_fields(message):
    if mobilemem_source_app(message['role'], message['name']) != 'Todo':
        return {}
    metadata = message.get('metadata') or {}
    todo_id = metadata.get('id')
    is_completed = metadata.get('is_completed')
    if isinstance(todo_id, str) and todo_id and type(is_completed) is bool:
        return dict(todo_id=todo_id, is_completed=is_completed)
    return {}


def evidence_source_app(evidence_ids, source_by_id):
    sources = {source_by_id.get(evidence_id) for evidence_id in evidence_ids}
    if len(sources) == 1:
        return sources.pop()
    return None
