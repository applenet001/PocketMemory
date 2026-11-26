"""Run the trained HGP3 heads locally for the iPhone inbox."""
import logging
import json
from pathlib import Path
import sys
from threading import Lock
from deepseek_client import complete, MODEL

ROOT = Path(__file__).resolve().parents[1]
HEAD = ROOT / 'outputs/hgp3-train-2200/classifier'
ENCODER = ROOT / 'models/qwen3-embedding-0.6b'


def review_uncertain(text, local):
    prompt = (ROOT / 'scripts/hgp-message-label-prompt.txt').read_text(encoding='utf-8')
    prompt += '\n这是个人记忆库：只将用户本人表达的信息作为用户事实。引用、教程或助手建议不代表用户已采纳。Working 只在仍有待办、等待、阻碍或后续跟进时为 true；已彻底完成的历史事件和单纯时间信息为 false。无法确定归属或是否仍需跟进时，标 uncertain=true。'
    try:
        answer = complete([{'role': 'system', 'content': prompt},
                           {'role': 'user', 'content': json.dumps([{'id': '0', 'role': 'user', 'target': text}], ensure_ascii=False)}])
        if answer.startswith('```'):
            answer = '\n'.join(answer.splitlines()[1:-1])
        entries = json.loads(answer)['items']
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError('Expected one annotation')
        reviewed = entries[0]
        if (str(reviewed.get('id')) != '0' or 'label' not in reviewed
                or reviewed['label'] not in ('semantic', 'episodic', 'procedural', None)
                or type(reviewed.get('working_state')) is not bool
                or type(reviewed.get('uncertain')) is not bool
                or not isinstance(reviewed.get('reason'), str) or not reviewed['reason'].strip()):
            raise ValueError('Invalid annotation')
        return {**local, **{key: reviewed[key] for key in ('label', 'working_state', 'uncertain', 'reason')},
                'source': 'deepseek', 'review_model': MODEL, 'fallback_status': 'reviewed',
                'local_prediction': {key: local.get(key) for key in ('label', 'working_state', 'confidence', 'threshold')}}
    except Exception as error:
        logging.warning('DeepSeek classification unavailable (%s)', type(error).__name__)
        return {**local, 'uncertain': True, 'fallback_status': 'failed'}


class InboxClassifier:
    def __init__(self):
        self.router = None
        self.lock = Lock()

    def __call__(self, item):
        body = item['text']
        if item['url']:
            body = body.removeprefix(item['url']).strip()
        text = '\n'.join(part for part in (item['note'], body) if part).strip()
        if not text:
            return {'status': 'skipped'}

        previous = item.get('classification') or {}
        if previous.get('status') == 'classified' and previous.get('uncertain') and previous.get('source') != 'deepseek':
            return review_uncertain(text, previous)

        try:
            with self.lock:
                if self.router is None:
                    scripts = str(ROOT / 'scripts')
                    if scripts not in sys.path:
                        sys.path.append(scripts)
                    from hgp3_router import AdaptiveRouter
                    from train_hgp3_2200 import DATA

                    router = AdaptiveRouter(HEAD, DATA, None, fallback=None, adaptive=False)
                    if (ENCODER / 'modules.json').is_file():
                        router.config['encoder'] = str(ENCODER)
                    router.tau0 = router.config['confidence_threshold']
                    self.router = router
                result, _ = self.router.route({'text': text}, allow_uncertain=True)
                result = {'status': 'classified', 'model': self.router.identity, **result}
            return review_uncertain(text, result) if result.get('uncertain') else result
        except Exception as error:
            logging.warning('Local HGP3 classification unavailable (%s)', type(error).__name__)
            return {'status': 'failed'}
