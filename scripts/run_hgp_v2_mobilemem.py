"""Construct, search, or evaluate one user from the pinned MobileMem Text dataset."""

import argparse
import json
import os
from pathlib import Path
from runpy import run_path
from string import Template
from unittest.mock import patch

from deepseek_usage import summarize_construction_usage
from mobilemem_source import evidence_source_app, todo_completion_fields
from run_hgp_membase import log_openai_usage, scoped_openai_usage
from run_hgp_v2_smoke import (
    ROOT, attach_working_states, event_time, index_working_states, open_search, read_jsonl, run, write_json,
)


def append_row(path, item):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')


def read_trajectory(dataset, index):
    trajectories = json.loads(dataset.read_text(encoding='utf-8'))
    user = trajectories[index - 1]
    user_id = user['person']['id']
    rows = []
    for session in user['sessions']:
        for message in session['messages']:
            if message['content'].strip():
                rows.append(dict(user_id=user_id, session_id=session['id'], message_id=message['id'],
                                 timestamp=message['timestamp'], role=message['role'], name=message['name'],
                                 content=message['content'], **todo_completion_fields(message)))
    rows.sort(key=lambda row: (event_time(row['timestamp']), row['session_id'], row['message_id']))
    if not rows or len({row['message_id'] for row in rows}) != len(rows):
        raise ValueError('Expected nonempty messages with unique IDs')

    questions = []
    gold = []
    for category in user['question_type_toolbook']['question_types']:
        for qa in category['qa_pairs']:
            questions.append(dict(question_id=qa['id'], question=qa['question']))
            gold.append(dict(question_id=qa['id'], question_type=qa['question_type'], golden_answers=qa['golden_answers']))
    if not questions or len({row['question_id'] for row in questions}) != len(questions):
        raise ValueError('Expected nonempty questions with unique IDs')
    return rows, questions, gold


def construct(dataset, index, target):
    if (target / 'construct/report.json').exists():
        write_construct_token_comparison(target.parent)
        print('Construct already completed.', flush=True)
        return
    directory = target / 'construct'
    if directory.exists():
        raise RuntimeError('Construct is incomplete. Use a new run directory to avoid duplicate memory writes.')
    rows, questions, gold = read_trajectory(dataset, index)
    with patch.dict(os.environ, {'HGP_DEEPSEEK_USAGE_LOG': str(directory / 'deepseek-usage.jsonl')}):
        log_openai_usage('hgp-v2-construct', record_missing=True, monitor_tokens=True)
        directory.mkdir(parents=True)
        write_json(target / 'questions.json', questions)
        write_json(target / 'gold.json', gold)
        run(rows, directory)
    write_construct_token_comparison(target.parent)


def write_construct_token_comparison(run_root):
    """MobileMem Text Table 1 uses one complete user trajectory as the unit."""
    users = []
    for index in (1, 2):
        target = run_root / f'user-{index:02d}'
        report_path = target / 'construct/report.json'
        if not report_path.is_file():
            return
        report = json.loads(report_path.read_text(encoding='utf-8'))
        usage = report['deepseek_construction']
        monitor = usage.get('membase_monitor') or summarize_construction_usage(
            read_jsonl(target / 'construct/deepseek-usage.jsonl'))['membase_monitor']
        users.append(dict(user=index, messages=report['input_count'], calls=usage['requests'],
                          input_tokens=monitor['input_tokens'], output_tokens=monitor['output_tokens'],
                          missing_usage_requests=usage['requests'] - monitor['counted_requests']))
    average = {label: sum(user[key] for user in users) / len(users)
               for label, key in (('In. Tok.', 'input_tokens'), ('Out. Tok.', 'output_tokens'))}
    complete_usage = all(user['missing_usage_requests'] == 0 for user in users)
    write_json(run_root / 'construct-token-comparison.json',
               dict(unit='tokens per complete user trajectory', measurement='MemBase token monitor',
                    tokenizer='Qwen3-Embedding-0.6B', users=users, complete_usage=complete_usage,
                    average_tokens=average if complete_usage else None,
                    average_thousand_tokens={key: value / 1000 for key, value in average.items()} if complete_usage else None,
                    observed_average_tokens=average))


def search(target, top_k):
    directory = target / 'search'
    directory.mkdir(exist_ok=True)
    questions = json.loads((target / 'questions.json').read_text(encoding='utf-8'))
    retrievals = directory / 'retrievals.jsonl'
    completed = {row['question_id']: row for row in read_jsonl(retrievals)}
    if any(row['top_k'] != top_k for row in completed.values()):
        raise ValueError('Search already has results for a different top_k')
    usage_path = directory / 'deepseek-usage.jsonl'
    pending = [row for row in questions if row['question_id'] not in completed]
    if pending:
        with patch.dict(os.environ, {'HGP_DEEPSEEK_USAGE_LOG': str(usage_path)}):
            log_openai_usage('hgp-v2-search', record_missing=True, monitor_tokens=True)
            with open_search(target / 'construct') as (memory, report):
                working_by_episode = index_working_states(report)
                sources = {turn['message_id']: turn['source_app'] for turn in report['turns'] if turn.get('source_app')}
                for position, question in enumerate(pending, len(completed) + 1):
                    result = memory.search(question['question'], top_k=top_k)
                    result.episodic = attach_working_states(result.episodic, working_by_episode)
                    for triple in result.semantic:
                        app = evidence_source_app(triple.get('evidence_ids') or [], sources)
                        if app:
                            triple['source_app'] = app
                    append_row(retrievals, dict(**question, top_k=top_k, context=result.to_context(max_items=top_k),
                                                episodic=result.episodic, semantic=result.semantic, procedural=result.procedural))
                    if position % 25 == 0 or position == len(questions):
                        print(f'Search {position}/{len(questions)}', flush=True)
    write_json(directory / 'usage.json', summarize_construction_usage(read_jsonl(usage_path), scope='search'))
    write_json(directory / 'summary.json', dict(questions=len(questions), top_k=top_k))


def complete(client, model, prompt, question_id, usage_path, max_tokens, *, stage):
    with scoped_openai_usage(stage, usage_path, question_id=question_id, monitor_tokens=True):
        response = client.chat.completions.create(
            model=model, messages=[{'role': 'user', 'content': prompt}], temperature=0,
            max_tokens=max_tokens, extra_body={'thinking': {'type': 'disabled'}},
        )
    return (response.choices[0].message.content or '').strip()


def evaluate(target):
    from dotenv import dotenv_values
    from openai import OpenAI

    search_dir = target / 'search'
    if not (search_dir / 'summary.json').is_file():
        raise FileNotFoundError('Search must complete before evaluation')
    retrievals = read_jsonl(search_dir / 'retrievals.jsonl')
    gold = {row['question_id']: row for row in json.loads((target / 'gold.json').read_text(encoding='utf-8'))}
    if len(retrievals) != len(gold) or {row['question_id'] for row in retrievals} != gold.keys():
        raise ValueError('Retrievals and gold answers must cover the same unique questions')

    config = dotenv_values(ROOT / 'config/local.env')
    judge_model = config.get('JUDGE_MODEL') or 'deepseek-v4-pro'
    judge_base = config.get('JUDGE_BASE_URL') or 'https://api.deepseek.com/v1'
    if judge_model != 'deepseek-v4-pro' or judge_base.rstrip('/') not in ('https://api.deepseek.com', 'https://api.deepseek.com/v1'):
        raise ValueError('This v2 eval expects DeepSeek V4 Pro on the official DeepSeek endpoint')
    prompts = run_path(str(ROOT / 'MemBase/membase/inference_utils/prompts.py'))['PROMPT_COLLECTIONS']
    answer_prompt = Template(prompts['default-question-answering'])
    judge_prompt = Template(prompts['default-exact-match'])
    directory = target / 'eval'
    directory.mkdir(exist_ok=True)
    answers_path = directory / 'answers.jsonl'
    judged_path = directory / 'judgments.jsonl'
    answer_usage = directory / 'deepseek-answer-usage.jsonl'
    judge_usage = directory / 'deepseek-judge-usage.jsonl'
    answers = {row['question_id']: row['prediction'] for row in read_jsonl(answers_path)}
    judgments = {row['question_id']: row for row in read_jsonl(judged_path)}
    with OpenAI(api_key=config['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com/v1', timeout=90) as qa_client, \
         OpenAI(api_key=config['JUDGE_API_KEY'], base_url=judge_base, timeout=90) as judge_client:
        for position, row in enumerate(retrievals, 1):
            question_id = row['question_id']
            if question_id not in answers:
                prompt = answer_prompt.substitute(question=row['question'], context=row['context'])
                answers[question_id] = complete(qa_client, 'deepseek-flash', prompt, question_id, answer_usage, 512,
                                                stage='hgp-v2-answer')
                append_row(answers_path, dict(question_id=question_id, prediction=answers[question_id]))
            if question_id not in judgments:
                refs = gold[question_id]
                prompt = judge_prompt.substitute(question=row['question'], prediction=answers[question_id],
                                                 golden_answers=refs['golden_answers'])
                response = complete(judge_client, judge_model, prompt, question_id, judge_usage, 64,
                                    stage='hgp-v2-judge')
                label = response.casefold().rstrip('.。!！')
                score = {'yes': 1, 'no': 0}.get(label)
                judgment = dict(question_id=question_id, question_type=refs['question_type'], response=response, score=score)
                judgments[question_id] = judgment
                append_row(judged_path, judgment)
            if position % 25 == 0 or position == len(retrievals):
                print(f'Eval {position}/{len(retrievals)}', flush=True)
    usage = dict(answer=summarize_construction_usage(read_jsonl(answer_usage), scope='answer'),
                 judge=summarize_construction_usage(read_jsonl(judge_usage), scope='judge'))
    write_json(directory / 'usage.json', usage)
    scores = [row['score'] for row in judgments.values() if row['score'] is not None]
    write_json(directory / 'summary.json', dict(questions=len(judgments), judge_model=judge_model,
                                               answer_model='deepseek-flash', unparsed=len(judgments) - len(scores),
                                               accuracy=sum(scores) / len(scores) if scores else None))


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
    target = args.run_root.resolve() / f'user-{args.user:02d}'
    if args.stage == 'construct':
        construct(args.dataset, args.user, target)
    elif args.stage == 'search':
        search(target, args.top_k)
    else:
        evaluate(target)


if __name__ == '__main__':
    main()
