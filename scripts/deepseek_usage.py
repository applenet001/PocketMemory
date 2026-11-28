"""Summarize construction tokens for the MobileMem Text protocol."""

from decimal import Decimal


PRICING_SOURCE = 'https://api-docs.deepseek.com/quick_start/pricing/'
# Public non-peak USD per million tokens, verified 2026-11-27.
OFF_PEAK_RATES = {
    'deepseek-flash': ('0.003', '0.15', '0.6'),
    'deepseek-v4-flash': ('0.003', '0.15', '0.6'),
    'deepseek-v4-flash-vision-exp': ('0.003', '0.15', '0.6'),
    'deepseek-v4-pro': ('0.022', '0.66', '1.98'),
}


def token_count(value):
    return type(value) is int and value >= 0


def summarize_construction_usage(records, *, scope='memory_construction'):
    """Keep token totals independent of whether a request can be priced."""
    input_tokens = output_tokens = missing_usage = unpriced = failed_requests = 0
    cache_hit_tokens = cache_miss_tokens = cache_usage_requests = 0
    minimum_usd = Decimal('0')
    models = {}
    monitor = dict(input_tokens=0, output_tokens=0, counted_requests=0)
    for record in records:
        model = record.get('model') or 'unknown'
        model_usage = models.setdefault(model, dict(requests=0, input_tokens=0, output_tokens=0))
        model_usage['requests'] += 1
        if record.get('error_type'):
            failed_requests += 1
        counted = record.get('membase_tokens') or {}
        if all(token_count(counted.get(key)) for key in ('input_tokens', 'output_tokens')):
            monitor['input_tokens'] += counted['input_tokens']
            monitor['output_tokens'] += counted['output_tokens']
            monitor['counted_requests'] += 1
        usage = record.get('usage') or {}
        prompt = usage.get('prompt_tokens')
        completion = usage.get('completion_tokens')
        if not token_count(prompt) or not token_count(completion):
            missing_usage += 1
            unpriced += 1
            continue
        input_tokens += prompt
        output_tokens += completion
        model_usage['input_tokens'] += prompt
        model_usage['output_tokens'] += completion

        cached = usage.get('prompt_cache_hit_tokens')
        if cached is None:
            cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        uncached = usage.get('prompt_cache_miss_tokens')
        if uncached is None and token_count(cached):
            uncached = prompt - cached
        if cached is None and token_count(uncached):
            cached = prompt - uncached
        has_cache_usage = token_count(cached) and token_count(uncached) and cached + uncached == prompt
        if has_cache_usage:
            cache_hit_tokens += cached
            cache_miss_tokens += uncached
            cache_usage_requests += 1
        rates = OFF_PEAK_RATES.get(model)
        if not rates or not has_cache_usage:
            unpriced += 1
            continue
        billed_tokens = (cached, uncached, completion)
        minimum_usd += sum(
            Decimal(price) * count for price, count in zip(rates, billed_tokens, strict=True)
        ) / 1_000_000

    cost = None
    if records and not unpriced:
        cost = dict(off_peak_usd=str(minimum_usd), peak_usd=str(minimum_usd * 2), source=PRICING_SOURCE,
                    rates_checked='2026-11-27')
    return dict(scope=scope, requests=len(records), input_tokens=input_tokens,
                output_tokens=output_tokens, total_tokens=input_tokens + output_tokens,
                cache_hit_tokens=cache_hit_tokens, cache_miss_tokens=cache_miss_tokens,
                cache_usage_requests=cache_usage_requests, failed_requests=failed_requests,
                missing_usage_requests=missing_usage, unpriced_requests=unpriced, models=models,
                membase_monitor=monitor, estimated_cost_range=cost)
