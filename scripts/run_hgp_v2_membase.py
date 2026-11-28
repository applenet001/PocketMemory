"""Run HGP v2 for one MobileMem trajectory using the native MemBase runners."""

import argparse
import json
import os
from functools import partial
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

from deepseek_usage import summarize_construction_usage
from run_hgp_membase import ROOT, MEMBASE, membase_token_monitor, usage_logger, count_membase_tokens
from run_hgp_v2_mobilemem import write_construct_token_comparison
from run_hgp_v2_smoke import configure_v2, read_jsonl, write_json


def seed_second_user_adaptive_state(source, destination):
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    adaptive = report['adaptive']
    iteration = adaptive['iterations']
    threshold = adaptive['threshold']
    checkpoint = None
    if iteration:
        state = json.loads((source / 'threshold_state.json').read_text(encoding='utf-8'))
        if state['iterations'] != iteration or state['history'][-1]['threshold'] != threshold:
            raise RuntimeError('User 1 adaptive checkpoint does not match its final Construct report')
        checkpoint = source / f'head-round-{iteration}.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'User 1 adaptive weights are missing: {checkpoint}')

    destination.mkdir(parents=True)
    if checkpoint is not None:
        shutil.copyfile(checkpoint, destination / checkpoint.name)
        write_json(destination / 'threshold_state.json', {**state, 'history': []})
    write_json(destination / 'adaptive_start.json',
               dict(source_user='user-01', source_namespace=report['namespace'],
                    initial_iteration=iteration, initial_threshold=threshold,
                    checkpoint=checkpoint.name if checkpoint is not None else 'four-head-500/head.pt'))


def construct(dataset, index, target, config):
    from membase import ConstructionRunner, ConstructionRunnerConfig
    from membase.utils.token_monitor import CostState, CostStateManager
    from hgp_v2_membase_layer import MEMORY_TYPE
    from openai.resources.chat.completions import Completions

    directory = target / 'construct'
    if directory.exists():
        if (directory / 'report.json').is_file() and (directory / 'token_cost.json').is_file():
            write_construct_token_comparison(target.parent)
            return
        raise RuntimeError(f'Construct is incomplete in {directory}. Use a new run root.')
    if index == 2:
        seed_second_user_adaptive_state(target.parent / 'user-01/construct', directory)
    else:
        directory.mkdir(parents=True)
    token_cost = directory / 'token_cost'
    tokenizer = ROOT / 'models/qwen3-embedding-0.6b/tokenizer.json'
    write_json(token_cost.with_suffix('.json'), {MEMORY_TYPE: CostState(window_size=0).to_dict()})
    counter = partial(count_membase_tokens, state_manager=CostStateManager, model_key=MEMORY_TYPE)
    logged = usage_logger(Completions.create, 'hgp-v2-construct', directory / 'deepseek-usage.jsonl',
                          record_missing=True, monitor_tokens=True, token_counter=counter)
    runner = ConstructionRunner(ConstructionRunnerConfig(
        memory_type=MEMORY_TYPE, dataset_type='MobileMem', dataset_path=str(dataset),
        memory_config={**config, 'save_dir': str(directory)}, start_idx=index-1, end_idx=index,
        num_workers=1, tokenizer_path=str(tokenizer), token_cost_save_filename=str(token_cost)))
    with patch.object(Completions, 'create', logged):
        runner.run()
    if not (directory / 'report.json').is_file():
        raise RuntimeError('MemBase did not finish constructing the user trajectory. See the runner log.')
    write_construct_token_comparison(target.parent)


def search(dataset, index, target, config, top_k):
    from membase import SearchRunner, SearchRunnerConfig
    from hgp_v2_membase_layer import MEMORY_TYPE
    from openai.resources.chat.completions import Completions

    membase_token_monitor()
    directory = target / 'search'
    directory.mkdir(parents=True, exist_ok=True)
    usage_path = directory / 'deepseek-usage.jsonl'
    logged = usage_logger(Completions.create, 'hgp-v2-search', usage_path, record_missing=True, monitor_tokens=True)
    runner = SearchRunner(SearchRunnerConfig(
        memory_type=MEMORY_TYPE, dataset_type='MobileMem', dataset_path=str(dataset),
        memory_config={**config, 'save_dir': str(directory), 'search_only': True},
        start_idx=index-1, end_idx=index, top_k=top_k, num_workers=1, strict=True))
    with patch.object(Completions, 'create', logged):
        runner.run()
    write_json(directory / 'usage.json', summarize_construction_usage(read_jsonl(usage_path), scope='search'))


def evaluate(index, target, top_k):
    from dotenv import dotenv_values
    from membase import EvaluationRunner, EvaluationRunnerConfig
    from membase.inference_utils.backends import OpenAIClient
    from hgp_v2_membase_layer import build_hgp_v2_context
    from openai.resources.chat.completions import Completions

    membase_token_monitor()
    filename = f'{top_k}_{index-1}_{index}.json'
    source = target / 'search' / filename
    if not source.is_file():
        raise FileNotFoundError(f'Search must finish first ({source})')
    directory = target / 'eval'
    directory.mkdir(parents=True, exist_ok=True)
    retrievals = directory / filename
    if retrievals.with_name(retrievals.stem + '_evaluation.json').is_file():
        return
    shutil.copyfile(source, retrievals)
    keys = dotenv_values(ROOT / 'config/local.env')
    base_url = 'https://api.deepseek.com/v1'
    answer_model = 'deepseek-flash'
    judge_model = keys.get('JUDGE_MODEL') or 'deepseek-v4-pro'
    judge_url = keys.get('JUDGE_BASE_URL') or base_url
    if judge_model != 'deepseek-v4-pro' or judge_url.rstrip('/') not in (base_url, 'https://api.deepseek.com'):
        raise ValueError('Expected the configured DeepSeek V4 Pro judge on the DeepSeek endpoint')

    paths = {role: directory / f'deepseek-{role}-usage.jsonl' for role in ('answer', 'judge')}
    original = Completions.create
    clients = {model: usage_logger(original, f'hgp-v2-{role}', paths[role], record_missing=True, monitor_tokens=True)
               for model, role in ((answer_model, 'answer'), (judge_model, 'judge'))}
    generate = OpenAIClient.get_text_generation_output

    def non_streaming(client, messages, model, **kwargs):
        kwargs.pop('reasoning_effort', None)
        kwargs.update(stream=False, max_tokens=64 if model == judge_model else 512,
                      extra_body={'thinking': {'type': 'disabled'}})
        return generate(client, messages, model=model, **kwargs)

    def logged(resource, *args, **kwargs):
        return clients.get(kwargs.get('model'), original)(resource, *args, **kwargs)

    runner = EvaluationRunner(EvaluationRunnerConfig(
        search_results_path=str(retrievals), dataset_type='MobileMem', qa_model=answer_model, judge_model=judge_model,
        api_keys=[keys['DEEPSEEK_API_KEY']], base_urls=[base_url], qa_batch_size=1, judge_batch_size=1,
        metric_configs={'llm_judge': {'api_keys': [keys['JUDGE_API_KEY']], 'base_urls': [judge_url]}},
        context_builder=build_hgp_v2_context))
    with patch.dict(os.environ, {'DEEPSEEK_USAGE_LOG': str(directory / 'deepseek-patch-usage.jsonl'),
                                 'MEMORY_METHOD': 'HGP-v2'}), patch.object(
        OpenAIClient, 'get_text_generation_output', non_streaming
    ), patch.object(Completions, 'create', logged):
        runner.run()
    write_json(directory / 'usage.json', {role: summarize_construction_usage(read_jsonl(path), scope=role)
                                         for role, path in paths.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('construct', 'search', 'eval'))
    parser.add_argument('--user', type=int, choices=(1, 2), required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'outputs/v2/mobilemem-text-989ba009.json')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    run_root = args.run_root.resolve()
    selection = run_root / 'selected-users.json'
    if selection.is_file():
        users = json.loads(selection.read_text(encoding='utf-8-sig'))
        if not isinstance(users, list) or not users or any(type(user) is not int or user not in (1, 2) for user in users):
            parser.error('selected-users.json must contain a nonempty list of MobileMem user indices')
        if args.user not in users:
            print(f'Skipping user {args.user} {args.stage}: excluded by selected-users.json', flush=True)
            return
    sys.path.insert(0, str(MEMBASE))
    os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    from hgp_v2_membase_layer import register_v2

    register_v2()
    target = run_root / f'user-{args.user:02d}'
    config = dict(user_id='guest', construction_dir=str(target / 'construct'))
    if args.stage == 'construct':
        construct(args.dataset.resolve(), args.user, target, config)
    elif args.stage == 'search':
        search(args.dataset.resolve(), args.user, target, config, args.top_k)
    else:
        configure_v2()
        evaluate(args.user, target, args.top_k)


if __name__ == '__main__':
    main()
