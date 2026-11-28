"""Render a self-contained HTML view of an HGP smoke replay report."""

import argparse
from html import escape
import json
from pathlib import Path

from mobilemem_source import mobilemem_source_app


STYLE = """
:root {
  color-scheme: light;
  --ink: #182523;
  --muted: #536763;
  --line: #d9e3dc;
  --paper: #f4f7f1;
  --white: #fff;
  --accent: #205f4b;
  --pale: #e7f0e9;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 15px/1.6 'Segoe UI', 'Microsoft YaHei', sans-serif;
}
a { color: var(--accent); text-underline-offset: 3px; }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 5px; }
header { background: #102f28; color: #effaf1; border-bottom: 5px solid #a7cf9d; }
header .wrap { padding-block: 36px 32px; }
.wrap { width: min(1120px, calc(100% - 36px)); margin: auto; }
.eyebrow { margin: 0; color: #c4e2c6; font-size: 13px; letter-spacing: .1em; }
h1 { font: 650 clamp(30px, 6vw, 50px)/1.14 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: .2em 0; }
.sub { margin: 0; color: #d8e8de; }
nav { display: flex; gap: 18px; flex-wrap: wrap; padding: 15px 0; border-bottom: 1px solid var(--line); }
nav a { text-decoration: none; font-weight: 650; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 10px; margin: 22px 0 14px; }
.metric { padding: 12px 16px; background: var(--white); border: 1px solid var(--line); border-radius: 10px; }
.metric span { display: block; color: var(--muted); font-size: 13px; }
.metric strong { font-size: 26px; color: var(--accent); }
main { padding-bottom: 55px; }
section { margin-top: 34px; scroll-margin-top: 18px; }
h2 { font-size: 23px; margin: 0 0 5px; }
h3 { font-size: 17px; margin: 0 0 5px; }
p { overflow-wrap: anywhere; }
.note, .meta { color: var(--muted); }
.note { margin: 0 0 14px; }
.meta { font-size: 13px; margin: 0; }
.panel, .event { background: var(--white); border: 1px solid var(--line); border-radius: 10px; padding: 15px 17px; margin-bottom: 10px; }
.timeline { padding-left: 16px; border-left: 2px solid #aac9b7; }
.event { position: relative; }
.event::before {
  content: '';
  position: absolute;
  left: -23px;
  top: 23px;
  width: 10px;
  height: 10px;
  background: var(--accent);
  border-radius: 50%;
  border: 2px solid var(--paper);
}
.event summary { display: grid; grid-template-columns: minmax(180px, auto) 1fr; gap: 12px; align-items: baseline; cursor: pointer; padding-right: 18px; }
.event summary::after { content: '+'; position: absolute; right: 15px; top: 14px; color: var(--accent); }
.event[open] summary::after { content: '−'; }
.summary-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.event .source { white-space: pre-wrap; margin: 12px 0; }
.state { border-left: 3px solid #74a384; background: var(--pale); padding: 9px 12px; margin: 10px 0 0; }
.state p { margin: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(265px, 1fr)); gap: 10px; }
.grid .panel { margin: 0; }
.tag { display: inline-block; font-size: 12px; font-weight: 650; padding: 1px 7px; margin: 0 5px 4px 0; border-radius: 5px; background: var(--pale); color: var(--accent); }
.tag.muted { background: #eff1ed; color: #54635d; }
.triplet { display: grid; grid-template-columns: minmax(0, 1fr); gap: 2px; }
.triplet dt { font: 650 13px/1.4 Consolas, monospace; color: var(--accent); overflow-wrap: anywhere; }
.triplet dd { margin: 0 0 9px; font-size: 17px; overflow-wrap: anywhere; }
ol { margin: .45em 0 .45em 1em; padding-left: 1em; }
footer { border-top: 1px solid var(--line); padding-top: 13px; color: var(--muted); margin-top: 26px; }
.empty { padding: 13px; border: 1px dashed #aabdb0; border-radius: 8px; color: var(--muted); }
.trace summary { color: var(--accent); cursor: pointer; font-weight: 650; }
.table-wrap { overflow: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
th, td { text-align: left; white-space: nowrap; padding: 5px 9px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); }
@media (max-width: 560px) {
  .event summary { grid-template-columns: 1fr; gap: 2px; }
  header .wrap { padding-block: 26px; }
  .wrap { width: calc(100% - 26px); }
}
"""


def html(value):
    return escape(str(value if value is not None else ''))


def app_badge(metadata):
    app = metadata.get('source_app') or mobilemem_source_app(metadata.get('role'), metadata.get('name'))
    return f'<span class="tag">来源：{html(app)}</span>' if app else ''


def render_token_usage(usage):
    if not usage or not usage['requests']:
        return '<p class="note">本次未记录 DeepSeek token 用量。</p>'
    missing = usage['missing_usage_requests']
    coverage = f'（{missing} 次响应缺少用量，以下为已记录合计）' if missing else ''
    tokens = (f'<p>DeepSeek 记忆构建用量 · {usage["requests"]} 次响应{coverage}<br>'
              f'Input {usage["input_tokens"]:,} · Output {usage["output_tokens"]:,} · '
              f'合计 {usage["total_tokens"]:,} tokens</p>')
    cost = usage['estimated_cost_range']
    if cost:
        tokens += (f'<p class="meta">公开低峰／高峰费率估算 '
                   f'US${html(cost["off_peak_usd"])}–{html(cost["peak_usd"])}。 '
                   f'<a href="{html(cost["source"])}">DeepSeek 费率</a></p>')
    return tokens


def render(data):
    turns = data['turns']
    states = {(track['session_id'], state['state_id']): state for track in data['working'] for state in track['states']}
    current = {(track['session_id'], track['current']['state_id']) for track in data['working'] if track['current']}
    provenance = {t['details'].get('episodic', {}).get('episode_id'): t for t in turns if 'episodic' in t['memory_types']}
    timeline = []
    for index, episode in enumerate(data['episodic']):
        turn = provenance.get(episode['id'], {})
        details = turn.get('details', {})
        state_id = details.get('working', {}).get('node_id')
        extracted = details.get('episodic', {}).get('extracted_state') or {}
        state_key = (turn.get('session_id'), state_id)
        state = states.get(state_key)
        if state_key in current:
            badge = '<span class="tag">会话当前状态</span>'
        elif state and not state.get('is_active', True):
            badge = '<span class="tag muted">已归档</span>'
        else:
            badge = '<span class="tag muted">轨迹状态</span>'
        confidence = extracted.get('confidence')
        confidence_label = f' · 状态置信度 {html(confidence)}' if confidence is not None else ''
        status = (f'<div class="state"><p>{badge}<strong>{html(state["state_name"])}</strong></p>'
                  f'<p class="meta">进度 {html(state["content"].get("progress"))} · 优先级 {html(state["priority"])}{confidence_label}</p>'
                  f'<p class="meta">状态 ID {html(state_id)}</p></div>') if state else ''
        content = html(episode['content'])
        previous = episode.get('prev_episode_id')
        link = f' · <a href="#episode-{html(previous)}">上一事件</a>' if previous else ''
        actors = '、'.join(episode.get('actors') or [])
        source = app_badge(episode.get('metadata') or {}) or app_badge(turn)
        initial_state = ' open' if index == 0 else ''
        timeline.append(f'<details class="event" id="episode-{html(episode["id"])}"{initial_state}><summary><time>{html(episode["timestamp"])}</time>'
                        f'<span class="summary-text">{content}</span></summary><p class="source">{content}</p>'
                        f'<p class="meta">{source}{html(turn.get("role"))} · {html(episode.get("event_type"))} · {html(actors)}{link}</p>'
                        f'<p class="meta">会话 {html(turn.get("session_id"))} · 事件 ID {html(episode["id"])}</p>{status}</details>')

    facts = [f'<article class="panel">{app_badge(item)}<dl class="triplet"><dt>{html(item["relation"])}</dt>'
             f'<dd>{html(item["entity"])}</dd></dl><p class="meta">置信度 {html(item.get("confidence"))}</p></article>'
             for item in data['semantic']]
    rules = []
    for rule in data['procedural']:
        steps = ''.join(f'<li>{html(step)}</li>' for step in rule['then_steps'])
        conditions = '、'.join(rule['if_conditions'])
        rules.append(f'<article class="panel"><span class="tag">{html(rule["status"])} · v{html(rule["version"])}</span>'
                     f'{app_badge(rule.get("meta") or {})}'
                     f'<h3>{html(rule["title"])}</h3><p class="meta">触发条件</p><p>{html(conditions)}</p>'
                     f'<p class="meta">执行步骤</p><ol>{steps}</ol></article>')

    metrics = ''.join(f'<div class="metric"><span>{label}</span><strong>{number}</strong></div>' for label, number in (
        ('输入消息', len(turns)), ('情节事件', len(data['episodic'])), ('Working 状态', len(states)),
        ('语义三元组', len(facts)), ('程序规则', len(rules))))
    adaptive = data['adaptive']
    evolution = data['evolution']
    failures = sum(not t['success'] or any(isinstance(detail, dict) and detail.get('success') is False
                                          for detail in t['details'].values()) for t in turns)
    trace = (f'<p>分类复判 {sum(t["source"] == "llm" for t in turns)} 次 · 增量缓冲 {adaptive["pending"]} 条 · '
             f'已更新 {adaptive["iterations"]} 轮 · 当前阈值 {adaptive["threshold"]}</p>'
             f'<p>记录到的消息级写入失败 {failures} 条 · Evolution 检测模式 {len(evolution["detected_patterns"])} 个</p>')
    trace_rows = ''.join(f'<tr><td>{index}</td><td>{html(turn["timestamp"])}</td>'
                         f'<td>{turn["max_confidence"]:.3f}</td><td>{turn["threshold"]:.3f}</td>'
                         f'<td>{html(turn["source"])}</td><td>{app_badge(turn) or "—"}</td>'
                         f'<td>{html(", ".join(turn["memory_types"])) or "—"}</td></tr>'
                         for index, turn in enumerate(turns, 1))
    empty = '<p class="empty">本次回放没有写入此类记忆。</p>'
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HGP v2 · MobileMem 回放</title><style>{STYLE}</style></head><body>
<header><div class="wrap"><p class="eyebrow">POCKETMEMORY / HGP v2</p><h1>MobileMem 回放</h1>
<p class="sub">{len(turns)} 条消息 · {len(set(turn['session_id'] for turn in turns))} 个会话 · {html(data['user_id'])}</p></div></header>
<main class="wrap"><nav aria-label="记忆分区"><a href="#timeline">情节与 Working</a><a href="#semantic">语义</a>
<a href="#procedural">程序规则</a><a href="#run">运行结果</a></nav><div class="metrics">{metrics}</div>
<section id="timeline"><h2>Episodic 与 Working</h2><p class="note">展开事件，查看它触发的 Working 状态。</p>
<div class="timeline">{''.join(timeline) or empty}</div></section>
<section id="semantic"><h2>Semantic 三元组</h2><p class="note">主体为本次回放用户。</p><div class="grid">{''.join(facts) or empty}</div></section>
<section id="procedural"><h2>Procedural 规则</h2><div class="grid">{''.join(rules) or empty}</div></section>
<section id="run"><h2>运行结果</h2><div class="panel">{trace}{render_token_usage(data.get('deepseek_construction'))}
<a href="report.json">查看原始报告 JSON</a></div>
<details class="panel trace"><summary>逐条路由</summary><div class="table-wrap"><table><thead><tr><th>序号</th><th>消息时间</th>
<th>Qwen 置信度</th><th>复判阈值</th><th>路由来源</th><th>App 来源</th><th>写入类型</th></tr></thead><tbody>{trace_rows}</tbody></table></div></details></section>
<footer>命名空间 {html(data['namespace'])}</footer></main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    output = args.report.with_name('index.html')
    output.write_text(render(report), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
