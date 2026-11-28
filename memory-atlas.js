'use strict';

const { facts, events } = userMemory;
const types = {
  SPORT: { label: '运动', color: '#c4a8ff' },
  LOCATION: { label: '地点', color: '#70dfd1' },
  PERSON: { label: '人物', color: '#ffb193' },
  HOBBY: { label: '兴趣', color: '#f2a9e4' },
};
const modes = {
  portrait: ['语义图谱', '按实体类型理解 User 1 的兴趣、地点和人际关系'],
  locations: ['地点轨迹', '按回放顺序看通勤起点、实测和运动计划'],
  people: ['人际网络', '按 PERSON 查看与 User 1 有关系记录的人物'],
  assistant: ['建议的依据', '回放到通勤、拼场或待办记录时，查看对应建议'],
};
const reactions = {
  portrait: '✨', locations: '📍', people: '🤝', assistant: '💡',
  浏览: '🏸', 计划: '📝', 实测: '🚗', 安排: '🏸', 待办: '🔔', 完成: '✅',
};
const adviceIndexes = events.flatMap((event, index) => event.advice ? [index] : []);
const eventById = new Map(events.map((event) => [event.id, event]));
const indexById = new Map(events.map((event, index) => [event.id, index]));
const $ = (selector) => document.querySelector(selector);
const create = (tag, text, className) => {
  const item = document.createElement(tag);
  if (text) item.textContent = text;
  if (className) item.className = className;
  return item;
};

const stage = $('#stage');
const canvas = $('#edges');
const nodes = $('#nodes');
const detail = $('#detail');
const eventPanel = $('#event');
const scrub = $('#scrub');
const play = $('#play');
const context = canvas.getContext('2d');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const modeButtons = [...document.querySelectorAll('[data-mode]')];
let mode = 'portrait';
let currentIndex = events.length - 1;
let selected = 'food-photo';
let timer;
let frame;
let reactionAnimation;
let positions = [];

function modeFacts(fact) {
  if (mode === 'locations') return fact.type === 'LOCATION';
  if (mode === 'people') return fact.type === 'PERSON';
  if (mode === 'assistant') return fact.evidence.includes(events[currentIndex].id);
  return true;
}

function visibleFacts() {
  return facts.filter((fact) => modeFacts(fact) && fact.evidence.some((id) => indexById.get(id) <= currentIndex));
}

function coordinates(index, count) {
  const angle = (index / count) * Math.PI * 2 - Math.PI / 2;
  return { x: 50 + Math.cos(angle) * 34, y: 50 + Math.sin(angle) * 35 };
}

function drawEdges() {
  if (!context) return;
  cancelAnimationFrame(frame);
  const { width, height } = stage.getBoundingClientRect();
  if (!width || !height) return;
  const ratio = Math.min(devicePixelRatio || 1, 2);
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const cx = width / 2;
  const cy = height / 2;
  const point = ({ x, y }) => [x * width / 100, y * height / 100];
  const start = performance.now();

  function draw(now) {
    const progress = reducedMotion.matches ? 1 : Math.min(1, (now - start) / 550);
    context.clearRect(0, 0, width, height);
    context.strokeStyle = '#a692dc24';
    context.lineWidth = 1;
    for (const radius of [0.35, 0.66, 1]) {
      context.beginPath();
      context.ellipse(cx, cy, width * 0.34 * radius, height * 0.35 * radius, 0, 0, Math.PI * 2);
      context.stroke();
    }
    for (const position of positions) {
      const [x, y] = point(position);
      const active = position.fact.id === selected;
      context.strokeStyle = types[position.fact.type].color;
      context.globalAlpha = active ? .8 : .32;
      context.lineWidth = active ? 2 : 1;
      const length = position.current ? 0.15 + 0.85 * progress : 1;
      context.beginPath();
      context.moveTo(cx, cy);
      context.lineTo(cx + (x - cx) * length, cy + (y - cy) * length);
      context.stroke();
    }
    context.globalAlpha = 1;
    if (mode === 'locations') drawCommute();
    if (progress < 1 && positions.some((position) => position.current)) frame = requestAnimationFrame(draw);
  }

  function drawCommute() {
    // Only this pair has a recorded origin/destination in the same trip plan.
    const origin = positions.find(({ fact }) => fact.id === 'origin');
    const destination = positions.find(({ fact }) => fact.id === 'business');
    if (!origin || !destination) return;
    const [x1, y1] = point(origin);
    const [x2, y2] = point(destination);
    context.lineWidth = 2;
    context.strokeStyle = '#70dfd188';
    context.setLineDash([6, 6]);
    context.beginPath();
    context.moveTo(x1, y1);
    context.bezierCurveTo(x1 + width * .36, y1, x2 + width * .15, y2, x2, y2);
    context.stroke();
    context.setLineDash([]);
  }

  draw(start);
}

function renderGraph(visible) {
  const event = events[currentIndex];
  const buttons = new Map([...nodes.children].map((button) => [button.dataset.id, button]));
  positions = visible.map((fact, index) => {
    const { x, y } = coordinates(index, visible.length);
    const current = fact.evidence.includes(event.id);
    let button = buttons.get(fact.id);
    if (!button) {
      button = create('button', null, 'orbiter');
      button.type = 'button';
      button.dataset.id = fact.id;
      button.style.setProperty('--tone', types[fact.type].color);
      button.append(create('small', types[fact.type].label), create('span', fact.entity));
      button.addEventListener('click', () => {
        selected = fact.id;
        render();
      });
      nodes.append(button);
    }
    buttons.delete(fact.id);
    button.style.left = `${x}%`;
    button.style.top = `${y}%`;
    button.classList.toggle('is-current', current);
    button.setAttribute('aria-pressed', String(fact.id === selected));
    return { x, y, fact, current };
  });
  for (const button of buttons.values()) button.remove();
  const legend = [...new Set(visible.map((fact) => fact.type))].map((type) => {
    const item = create('li', `${types[type].label} · ${type}`);
    item.style.setProperty('--tone', types[type].color);
    return item;
  });
  $('#legend').replaceChildren(...legend);
  $('#empty-graph').hidden = visible.length > 0;
  $('#empty-graph').textContent = '在这个时间点还没有该类语义关系。';
  $('#fact-count').textContent = `${visible.length} 条关系`;
  drawEdges();
}

function showDetails(fact) {
  const event = events[currentIndex];
  detail.replaceChildren();
  const adviceMode = mode === 'assistant';
  $('#inspector-title').textContent = adviceMode ? '助手建议' : '记忆详情';
  $('#inspector-kicker').textContent = adviceMode ? '以回放时间为准' : '当前选中';
  if (adviceMode) {
    detail.append(create('h3', event.title));
    detail.append(evidenceCard(event));
    if (event.advice) detail.append(adviceCard(event.advice));
    else detail.append(create('p', '此片段提供背景信息。可跳到有建议的片段。', 'meta'));
    const next = adviceIndexes.find((index) => index > currentIndex) ?? adviceIndexes[0];
    const button = create('button', '跳到下一条建议', 'text-button');
    button.type = 'button';
    button.addEventListener('click', () => chooseEvent(next));
    detail.append(button);
    return;
  }
  if (!fact) {
    detail.append(create('p', '播放生活轨迹，或选择其他时间点查看相关关系。', 'meta'));
    return;
  }
  detail.append(create('h3', fact.entity), create('p', fact.label));
  const relation = create('p', `Entity.type ${fact.type} · `, 'relation');
  relation.append(create('code', fact.relation));
  detail.append(relation);
  const evidence = fact.evidence.includes(event.id) ? event :
    eventById.get(fact.evidence.findLast((id) => indexById.get(id) <= currentIndex));
  const method = evidence.id === fact.databaseEvidence ? '数据库证据引用' : '与原始记录核对';
  detail.append(evidenceCard(evidence, method));
  if (evidence.advice) detail.append(adviceCard(evidence.advice));
  const button = create('button', '在时间轴查看依据', 'text-button');
  button.type = 'button';
  button.addEventListener('click', () => {
    chooseEvent(indexById.get(evidence.id));
    scrub.focus();
  });
  detail.append(button);
}

function evidenceCard(event, method = '来源记录') {
  const card = create('div', null, 'evidence');
  card.append(create('span', `${method} · ${event.date} · ${event.source}`), create('p', event.text));
  return card;
}

function adviceCard(advice) {
  const card = create('div', null, 'advice');
  card.append(create('strong', advice.label), create('p', advice.text));
  return card;
}

function render() {
  const event = events[currentIndex];
  const visible = visibleFacts();
  if (!visible.some((fact) => fact.id === selected)) {
    selected = (visible.find((fact) => fact.evidence.includes(event.id)) || visible[0])?.id;
  }
  $('#graph-title').textContent = modes[mode][0];
  $('#graph-description').textContent = modes[mode][1];
  renderGraph(visible);
  showDetails(visible.find((fact) => fact.id === selected));
  scrub.value = currentIndex;
  scrub.setAttribute('aria-valuetext', `${event.date} ${event.title}`);
  ticks.forEach((tick, index) => tick.setAttribute('aria-current', index === currentIndex ? 'step' : 'false'));
  eventPanel.replaceChildren(
    create('span', `${event.date} · ${event.source} · ${event.kind}`, 'meta'),
    create('strong', event.title), create('p', event.text),
  );
}

function react(emoji) {
  const badge = $('#reaction');
  badge.textContent = emoji;
  reactionAnimation?.cancel();
  if (reducedMotion.matches) return;
  reactionAnimation = badge.animate([
    { opacity: 0, transform: 'translate(-6px, 8px) scale(.55)' },
    { opacity: 1, transform: 'translate(0, -5px) scale(1.16)', offset: .55 },
    { opacity: 1, transform: 'translate(0, 0) scale(1)' },
  ], { duration: 560, easing: 'cubic-bezier(.21,.68,.35,1)' });
}

function stopPlayback() {
  clearInterval(timer);
  timer = undefined;
  play.textContent = currentIndex === events.length - 1 ? '▶ 播放轨迹' : '▶ 继续播放';
}

function chooseEvent(index, fromPlayback = false) {
  currentIndex = index;
  if (!fromPlayback) stopPlayback();
  selected = visibleFacts().find((fact) => fact.evidence.includes(events[index].id))?.id ?? selected;
  render();
  react(reactions[events[index].kind]);
}

const ticks = events.map((event, index) => {
  const tick = create('button', event.date.slice(5).replace('-', '/'));
  tick.type = 'button';
  tick.title = event.title;
  tick.setAttribute('aria-label', `${event.date} ${event.title}`);
  tick.addEventListener('click', () => chooseEvent(index));
  return tick;
});
$('#ticks').append(...ticks);
scrub.max = events.length - 1;
scrub.disabled = false;
scrub.addEventListener('input', () => chooseEvent(Number(scrub.value)));
play.disabled = false;
play.addEventListener('click', () => {
  if (timer !== undefined) {
    stopPlayback();
    return;
  }
  if (currentIndex === events.length - 1) chooseEvent(0, true);
  play.textContent = 'Ⅱ 暂停播放';
  timer = setInterval(() => {
    chooseEvent(currentIndex + 1, true);
    if (currentIndex === events.length - 1) stopPlayback();
  }, 1750);
});
for (const button of modeButtons) {
  button.addEventListener('click', () => {
    mode = button.dataset.mode;
    for (const option of modeButtons) option.setAttribute('aria-pressed', String(option === button));
    render();
    react(reactions[mode]);
  });
}

render();
react(reactions[mode]);
new ResizeObserver(drawEdges).observe(stage);
