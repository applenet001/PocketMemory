"""Server-side access to the same DeepSeek model used in evaluation."""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / 'config/local.env'
MODEL = 'deepseek-flash'


def api_key():
    key = os.getenv('DEEPSEEK_API_KEY', '').strip()
    if key:
        return key
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding='utf-8-sig').splitlines():
            field, _, value = line.partition('=')
            if field.strip() == 'DEEPSEEK_API_KEY':
                key = value.strip().strip('"').strip("'")
                if key:
                    return key
    raise RuntimeError('DeepSeek API key unavailable')


def complete(messages):
    from openai import OpenAI

    with OpenAI(api_key=api_key(), base_url='https://api.deepseek.com/v1',
                timeout=60, max_retries=0) as client:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.0, reasoning_effort='low'
        )
    if not response.choices:
        return ''
    return (response.choices[0].message.content or '').strip()
