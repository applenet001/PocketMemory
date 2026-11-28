"""Run MemBase stages with PocketMemory's HGP layer registered."""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
from functools import lru_cache
from threading import Lock, get_ident
from time import perf_counter
from urllib.parse import urlsplit
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MEMBASE = ROOT / 'MemBase'
STAGES = {'construction': 'memory_construction.py', 'search': 'memory_search.py',
          'evaluation': 'memory_evaluation.py'}


@lru_cache(maxsize=1)
def membase_token_monitor():
    tokenizer = str(ROOT / 'models/qwen3-embedding-0.6b/tokenizer.json')
    if not Path(tokenizer).is_file():
        raise FileNotFoundError(tokenizer)
    with patch.dict(os.environ, {'LITELLM_LOCAL_MODEL_COST_MAP': 'True'}):
        module = runpy.run_path(str(MEMBASE / 'membase/utils/token_monitor.py'))
    manager = module['CostStateManager']
    manager.register(tokenizer, state=module['CostState'](window_size=0))
    return manager, tokenizer


def count_membase_tokens(messages, response, elapsed, *, state_manager=None, model_key=None):
    manager, tokenizer = (state_manager, model_key) if state_manager is not None else membase_token_monitor()
    result = {'input': {'messages': messages},
              'output': {'messages': response.choices[0].message.content or ''},
              'elapsed': elapsed}
    manager.update(tokenizer, result)
    return {key + '_tokens': result[key][key + '_tokens'] for key in ('input', 'output')}


def log_openai_usage(stage, *, record_missing=False, monitor_tokens=False):
    destination = os.getenv('HGP_DEEPSEEK_USAGE_LOG')
    if not destination:
        return
    from openai.resources.chat.completions import Completions

    if monitor_tokens:
        membase_token_monitor()
    Completions.create = usage_logger(Completions.create, stage, destination,
                                      record_missing=record_missing, monitor_tokens=monitor_tokens)


def usage_logger(original, stage, destination, *, record_missing=False, thread_id=None, fields=None, monitor_tokens=False,
                 token_counter=count_membase_tokens):
    lock = Lock()

    def record(entry):
        with lock, Path(destination).open('a', encoding='utf-8') as file:
            file.write(json.dumps(entry) + '\n')

    def logged(resource, *args, **kwargs):
        if thread_id is not None and get_ident() != thread_id:
            return original(resource, *args, **kwargs)
        if stage in ('construction', 'search') and str(kwargs.get('model', '')).startswith('deepseek'):
            kwargs['extra_body'] = {**(kwargs.get('extra_body') or {}), 'thinking': {'type': 'disabled'}}
        requested_model = kwargs.get('model')
        entry = dict(timestamp=datetime.now(timezone.utc).isoformat(), stage=stage, **(fields or {}))
        started = perf_counter()
        try:
            response = original(resource, *args, **kwargs)
        except Exception as error:
            if record_missing:
                record(dict(**entry, requested_model=requested_model, model=requested_model, usage=None,
                            response_id=getattr(error, 'request_id', None), error_type=type(error).__name__))
            raise
        usage = getattr(response, 'usage', None)
        if usage is not None or record_missing:
            entry.update(model=getattr(response, 'model', None) or requested_model,
                         usage=usage.model_dump() if usage is not None else None)
            if record_missing:
                entry.update(requested_model=requested_model, response_id=getattr(response, 'id', None))
            if monitor_tokens:
                try:
                    entry['membase_tokens'] = token_counter(kwargs['messages'], response, perf_counter() - started)
                except Exception as error:
                    entry['membase_monitor_error'] = type(error).__name__
            record(entry)
        return response

    return logged


@contextmanager
def scoped_openai_usage(stage, destination, *, monitor_tokens=False, **fields):
    from openai.resources.chat.completions import Completions

    if monitor_tokens:
        membase_token_monitor()
    logged = usage_logger(Completions.create, stage, destination, record_missing=True,
                          thread_id=get_ident(), fields=fields, monitor_tokens=monitor_tokens)
    with patch.object(Completions, 'create', logged):
        yield


def bootstrap_password(env_file):
    from neo4j import GraphDatabase
    from hgp_neo4j_layer import settings

    values = settings(env_file)
    uri, username, password = (values[key] for key in ('NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD'))
    if urlsplit(uri).hostname not in ('localhost', '127.0.0.1', '::1') or username != 'neo4j':
        raise ValueError('Password bootstrap is only supported on a fresh local default Neo4j account')
    with GraphDatabase.driver(uri, auth=('neo4j', 'neo4j')) as driver, driver.session(database='system') as session:
        session.run('ALTER CURRENT USER SET PASSWORD FROM $old TO $new',
                    old='neo4j', new=password).consume()


def smoke(env_file):
    from membase.model_types.dataset import Message
    from hgp3_router import AdaptiveRouter
    from hgp_neo4j_layer import HGPConfig, HGPLayer

    labels = {
        'smoke-profile': ('semantic', False),
        'smoke-event-a': ('episodic', False),
        'smoke-event-b': ('episodic', False),
        'smoke-procedure': ('procedural', False),
        'smoke-working': (None, True),
    }
    turns = [
        ('smoke-profile', '我长期喜欢无糖茶。'),
        ('smoke-event-a', '昨天我去了青石图书馆。'),
        ('smoke-event-b', '今天我把书还给了青石图书馆。'),
        ('smoke-procedure', '归还图书的办法：\n1. 打开还书登记页。\n2. 扫描图书条码。'),
        ('smoke-working', '还剩一本借来的书，明天要还。'),
    ]
    common = dict(env_file=env_file, graph_namespace='hgp-local-synthetic-smoke',
                  tau0=1.0, adaptive_updates=False)
    config = HGPConfig(user_id='synthetic-graph-smoke', **common)

    def simulated_fallback(message):
        label, working = labels[message['id']]
        return dict(label=label, working_state=working, uncertain=False)

    router = AdaptiveRouter(config.classifier_dir, config.training_data,
                            ROOT / 'outputs/hgp-neo4j-smoke/unused.pt',
                            fallback=simulated_fallback, tau0=1.0, adaptive=False)
    layer = HGPLayer(config, router=router)
    other = HGPLayer(HGPConfig(user_id='synthetic-other-user', **common), router=router)
    try:
        messages = [
            Message(id=id, name='Synthetic fixture', role='user',
                    content=text, timestamp=f'2025-01-{i:02d}T09:00:00')
            for i, (id, text) in enumerate(turns, 1)
        ]
        layer.add_messages(messages)
        layer.add_message(messages[0])
        layer.save_memory()
        retrieved = layer.retrieve('青石图书馆怎么还书？', k=10)
        if not layer.load_memory() or other.retrieve('青石图书馆', k=10):
            raise RuntimeError('Graph save/reload or user isolation check failed')
        kinds = {item.metadata['kind'] for item in retrieved}
        if kinds != {'semantic', 'episodic', 'procedural', 'working'} or len(retrieved) != 5:
            raise RuntimeError('Unexpected number or type of retrieved memories')
        if not any('Adjacent context:' in entry.formatted_content for entry in retrieved):
            raise RuntimeError('Episodic adjacency did not appear in retrieved context')
        print(json.dumps({'synthetic_memories': len(retrieved), 'types': sorted(kinds),
                          'user_isolation': True, 'llm': 'simulated'}, ensure_ascii=False), flush=True)
    finally:
        layer.close()
        other.close()


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.path.insert(0, str(MEMBASE))
    if len(sys.argv) > 1 and sys.argv[1] in STAGES:
        from hgp_original_layer import register_original

        stage = sys.argv.pop(1)
        register_original()
        log_openai_usage(stage)
        if stage != 'evaluation':
            if '--memory-type' in sys.argv:
                raise ValueError('This launcher sets --memory-type HGP')
            sys.argv.extend(['--memory-type', 'HGP'])
        elif not any(arg == '--context-builder' or arg.startswith('--context-builder=') for arg in sys.argv[1:]):
            sys.argv.extend(['--context-builder', 'hgp_original_layer.build_hgp_qa_context'])
        path = MEMBASE / STAGES[stage]
        sys.argv[0] = str(path)
        os.chdir(MEMBASE)
        runpy.run_path(str(path), run_name='__main__')
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init-graph', 'legacy-init-graph', 'legacy-smoke', *STAGES])
    parser.add_argument('--env-file', type=Path, default=ROOT / 'config/local.env')
    parser.add_argument('--config-path', type=Path, default=ROOT / 'config/templates/hgp_neo4j.json')
    parser.add_argument('--bootstrap-default-password', action='store_true')
    args = parser.parse_args()
    if args.command == 'init-graph':
        from hgp_original_layer import OriginalHGPConfig, initialize_original

        if args.bootstrap_default_password:
            parser.error('Original HGP uses the existing Neo4j Desktop credentials')
        values = json.loads(args.config_path.read_text(encoding='utf-8'))
        values['env_file'] = args.env_file
        initialize_original(OriginalHGPConfig(**values))
        print('The original HGP semantic and episodic stores are initialized.', flush=True)
        return

    from hgp_neo4j_layer import initialize_graph
    if args.bootstrap_default_password:
        if args.command != 'legacy-init-graph':
            parser.error('--bootstrap-default-password applies to legacy-init-graph')
        bootstrap_password(args.env_file)
    initialize_graph(args.env_file)
    if args.command == 'legacy-smoke':
        smoke(args.env_file)
    else:
        print('Neo4j HGP graph constraints are ready.', flush=True)


if __name__ == '__main__':
    main()
