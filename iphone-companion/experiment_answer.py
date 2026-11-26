"""Answer with the evaluation prompt and the exact context from one search."""
from collections import OrderedDict
import logging
import secrets
from threading import Lock
from time import monotonic

from deepseek_client import complete


def answer_question(question, context):
    try:
        from membase.inference_utils.prompts import get_prompt

        prompt = get_prompt('default-question-answering').substitute(question=question, context=context)
        answer = complete([{'role': 'system', 'content': 'You are a helpful assistant.'},
                           {'role': 'user', 'content': prompt}])
        if answer:
            return {'status': 'ready', 'text': answer}
    except Exception as error:
        logging.warning('DeepSeek answer generation unavailable (%s)', type(error).__name__)
    return {'status': 'unavailable'}


class SearchAnswers:
    def __init__(self):
        self._pending = OrderedDict()
        self._lock = Lock()

    def remember(self, owner, question, context):
        if not context:
            return None
        ticket = secrets.token_urlsafe(24)
        now = monotonic()
        with self._lock:
            while self._pending:
                oldest = next(iter(self._pending))
                if len(self._pending) < 32 and self._pending[oldest][0] > now:
                    break
                self._pending.popitem(last=False)
            self._pending[ticket] = (now + 300, owner, question, context)
        return ticket

    def get(self, owner, ticket):
        with self._lock:
            entry = self._pending.get(ticket)
            if entry is None or entry[0] <= monotonic() or entry[1] != owner:
                return None
            return entry[2:]
