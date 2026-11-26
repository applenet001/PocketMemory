const $ = id => document.getElementById(id);
let current = null, searchTimer, requestNumber = 0, authGeneration = 0, signedInRole = null;
let semanticQuery = '';
const processingNames = {extract: '原件已保存，正在后台提取文字', queued: '已保存，等待后台处理', classifying: '已保存，正在分类；低置信度时由 DeepSeek 复判', syncing: '已保存，正在同步记忆', failed: '已保存，后台处理失败，可重试'};
let processingTimer;
function notice(message) { $('status').textContent = message; $('status').hidden = false; clearTimeout(notice.timer); notice.timer = setTimeout(() => $('status').hidden = true, 6500); }
async function api(path, options = {}) {
  const generation = authGeneration;
  const response = await fetch(path, options);
  const data = await response.json();
  if (generation !== authGeneration) throw new Error('会话已变更，请重新操作');
  if (!response.ok) { if (response.status === 401) clearPrivateUI(); throw new Error(data.error || '请求失败'); }
  return data;
}
function clearPrivateUI() {
  clearTimeout(processingTimer);
  clearTimeout(searchTimer);
  semanticQuery = '';
  $('memory-search').reset();
  $('memory-search-status').hidden = true;
  authGeneration++; requestNumber++; current = null; signedInRole = null;
  $('detail').close(); $('capture-form').reset(); clearPhoto();
  for (const id of ['items', 'profile-groups', 'graph-canvas', 'graph-list', 'users-list', 'detail-links', 'graph-scope']) $(id).replaceChildren();
  for (const id of ['detail-title', 'detail-text', 'detail-note', 'detail-date', 'detail-status', 'graph-detail', 'graph-status', 'account-name']) $(id).textContent = '';
  $('new-user-key').value = ''; $('user-credential').hidden = true; $('tab-admin').hidden = true;
  $('tab-experiment').hidden = true;
  $('tab-experiment-search').hidden = true;
  $('tab-workflows').hidden = true;
  for (const id of ['experiment-profile', 'experiment-user', 'experiment-status', 'workflow-user',
    'workflow-graphs', 'workflow-status', 'experiment-search-user', 'experiment-search-results',
    'experiment-search-status']) $(id).replaceChildren();
  $('experiment-search-form').reset();
  for (const id of ['experiment-portrait', 'personal-portrait']) {
    $(id).hidden = true;
    $(id).textContent = '';
  }
  $('experiment-answer').hidden = true;
  $('experiment-answer-text').textContent = '';
  $('graph-database').replaceChildren();
  $('graph-source').hidden = true;
  $('graph-source').onclick = null;
  $('graph-technical').hidden = true;
  $('graph-technical').open = false;
  $('graph-technical-data').textContent = '';
  $('login').hidden = false; $('workspace').hidden = true; tab('library');
}
const tabs = ['library', 'capture', 'graph', 'guide', 'admin', 'experiment', 'experiment-search', 'workflows'];
function tab(name) { for (const item of tabs) { $(item).hidden = item !== name; $('tab-' + item).classList.toggle('selected', item === name); } }
for (const name of tabs) $('tab-' + name).onclick = () => {
  tab(name);
  if (name === 'admin') loadUsers();
  if (name === 'graph' && signedInRole === 'admin') loadExperimentUsers('graph-scope', 'graph-status');
  if (name === 'experiment') loadExperimentUsers('experiment-user', 'experiment-status');
  if (name === 'experiment-search') loadExperimentUsers('experiment-search-user', 'experiment-search-status');
  if (name === 'workflows') {
    if (signedInRole === 'admin') loadExperimentUsers('workflow-user', 'workflow-status');
    else loadPersonalWorkflows();
  }
};

async function openWorkspace() {
  try {
    const { user } = await api('/api/me');
    signedInRole = user.role;
    const admin = user.role === 'admin';
    for (const name of ['admin', 'experiment', 'experiment-search']) $('tab-' + name).hidden = !admin;
    $('tab-workflows').hidden = false;
    $('workflow-user').hidden = !admin;
    $('workflow-user-label').hidden = !admin;
    for (const name of ['library', 'capture', 'guide']) $('tab-' + name).hidden = admin;
    const graphViews = admin
      ? [['实验记录 · 事件', 'episodic'], ['实验记录 · 偏好与概念', 'semantic']]
      : [['我的事件', 'personal-episodic'], ['我的偏好与概念', 'personal-semantic']];
    $('graph-database').replaceChildren(...graphViews.map(([label, value]) => new Option(label, value)));
    $('graph-scope-label').textContent = admin ? '评测用户' : '数据空间';
    $('account-name').textContent = user.name + (admin ? ' · 管理员' : ' · 个人记忆库');
    $('login').hidden = true;
    $('workspace').hidden = false;
    if (admin) $('tab-experiment').click();
    else {
      tab('library');
      await reload();
    }
  } catch (error) {
    if (error.message !== '请使用自己的访问密钥登录') notice(error.message);
  }
}
function kind(item) { return item.url ? '链接' : item.mime.startsWith('image/') ? '图片' : item.name ? '文件' : '文字'; }
const routeNames = { semantic: '语义', episodic: '事件', procedural: '方法与流程' };
function renderProfile(profile = {}) {
  const groups = $('profile-groups');
  groups.replaceChildren();
  const sections = [
    ['semantic', '喜欢什么、了解什么'],
    ['episodic', '最近做过什么'],
    ['procedural', '记下了哪些做法'],
    ['working', '任务进行到哪一步'],
  ];
  for (const [kind, title] of sections) {
    const section = document.createElement('section');
    section.className = 'profile-group';
    const heading = document.createElement('h3');
    heading.textContent = title;
    section.append(heading);
    const records = profile[kind] || [];
    if (!records.length) {
      const empty = document.createElement('p');
      empty.className = 'hint';
      empty.textContent = '暂无相关记录';
      section.append(empty);
    }
    for (const item of records) {
      const source = [item.note, item.text].filter(Boolean).join(' ').replace(/\s+/g, ' ').trim();
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'profile-source';
      button.onclick = () => detail(item);
      const preview = document.createElement('span');
      const characters = Array.from(source || item.title);
      preview.textContent = characters.slice(0, 100).join('') + (characters.length > 100 ? '…' : '');
      const date = document.createElement('small');
      const uncertainty = item.classification.uncertain ? ' · 分类置信度低' : '';
      date.textContent = new Date(item.created).toLocaleDateString() + uncertainty + ' · 查看原文';
      button.append(preview, date);
      section.append(button);
    }
    groups.append(section);
  }
}
function classificationText(result) {
  if (!result) return '尚未分类';
  if (result.status === 'failed') return '本地分类暂不可用，已保留原文，可重试';
  if (result.status === 'skipped') return '没有可分类的正文';
  const route = routeNames[result.label] || '无长期记忆类别';
  const working = result.working_state ? ' · 工作状态' : '';
  if (result.source === 'deepseek') return `${route}${working} · DeepSeek 复判${result.uncertain ? ' · 仍不确定，暂不自动同步' : ''}`;
  const confidence = Math.round(result.confidence * 100);
  return `${route}${working} · ${confidence}%${result.uncertain ? ' · 低置信度' : ''}${result.fallback_status === 'failed' ? ' · DeepSeek 复判失败，可重试' : ''}`;
}
async function reload() {
  if (signedInRole !== 'user') return;
  clearTimeout(processingTimer);
  const number = ++requestNumber;
  const query = $('search').value.trim();
  const semantic = Boolean(query) && semanticQuery === query;
  const searchStatus = $('memory-search-status');
  searchStatus.textContent = semantic ? '正在按意思搜索…' : '';
  searchStatus.hidden = !semantic;
  try {
    const params = new URLSearchParams({ q: query });
    if (semantic) params.set('mode', 'semantic');
    const data = await api('/api/memories?' + params);
    if (number !== requestNumber) return;
    searchStatus.hidden = !data.search_fallback;
    searchStatus.textContent = data.search_fallback ? '按意思搜索暂不可用，以下是关键词结果。' : '';
    $('login').hidden = true; $('workspace').hidden = false; $('count').textContent = data.items.length;
    renderProfile(data.profile);
    $('items').replaceChildren();
    if (!data.items.length) { const empty = document.createElement('div'); empty.className = 'empty'; empty.textContent = query ? '没有找到相关记忆。试试换个问法或关键词。' : '还没有记忆。点击「记下一条」，留下第一份内容。'; $('items').append(empty); }
    for (const item of data.items) {
      const button = document.createElement('button'); button.className = 'card';
      const badge = processingNames[item.processing] || (item.classification?.status === 'classified' ? kind(item) + ' · ' + classificationText(item.classification) : kind(item));
      for (const [tag, text, cls] of [['span', badge, 'kind'], ['strong', item.title, ''], ['p', item.note || item.text || item.status, ''], ['time', new Date(item.created).toLocaleString(), '']]) {
        const element = document.createElement(tag); element.textContent = text; element.className = cls; button.append(element);
      }
      button.onclick = () => detail(item); $('items').append(button);
    }
    if ($('detail').open && current) {
      const updated = data.items.find(item => item.id === current.id);
      if (updated) detail(updated);
    }
    if (!semantic) processingTimer = setTimeout(() => { if (!document.hidden) reload(); }, 3000);
  } catch (error) {
    if (number !== requestNumber) return;
    searchStatus.hidden = true;
    notice(error.message);
  }
}
function detail(item) {
  current = item;
  $('detail-title').textContent = item.title;
  $('detail-date').textContent = '保存时间 ' + new Date(item.created).toLocaleString();
  $('detail-status').textContent = processingNames[item.processing] || item.status; $('detail-note').textContent = item.note; $('detail-note').hidden = !item.note;
  $('detail-classification').textContent = 'HGP 分类 · ' + classificationText(item.classification);
  const retryReview = item.classification?.uncertain && item.classification?.fallback_status !== 'reviewed';
  $('classify').hidden = item.processing !== 'failed' && (['extract', 'queued', 'classifying', 'syncing'].includes(item.processing) || Boolean(item.classification && item.classification.status !== 'failed' && !retryReview));
  const result = item.classification;
  const eligible = result?.status === 'classified' && !result.uncertain && ['semantic', 'episodic'].includes(result.label);
  $('detail-graph-status').textContent = item.graph_state === 'synced' ? '已写入个人 Neo4j 图'
    : item.graph_state === 'pending' ? 'Neo4j 同步待重试'
    : result?.uncertain ? '分类置信度低，暂留在记忆库'
    : eligible ? '可同步到个人 Neo4j 图' : '';
  $('graph-sync').hidden = !eligible || item.graph_state === 'synced';
  $('detail-text').textContent = item.text || '暂无可检索正文，可打开原件查看。'; $('detail-links').replaceChildren();
  if (item.url && /^https?:\/\//i.test(item.url)) { const a = document.createElement('a'); a.href = item.url; a.target = '_blank'; a.rel = 'noopener noreferrer'; a.textContent = '打开来源链接 ↗'; $('detail-links').append(a); }
  if (item.name) { const a = document.createElement('a'); a.href = '/api/files/' + item.id; a.download = item.name; a.textContent = '下载原件 ↗'; $('detail-links').append(a); }
  if (!$('detail').open) $('detail').showModal();
}
$('close').onclick = () => $('detail').close();
$('graph-sync').onclick = async () => {
  if (!current) return;
  const id = current.id;
  $('graph-sync').disabled = true;
  try {
    const { item } = await api('/api/memories/' + id + '/graph-sync', { method: 'POST' });
    if ($('detail').open && current?.id === id) detail(item);
  } catch (error) { notice(error.message); }
  finally { $('graph-sync').disabled = false; }
};
$('classify').onclick = async () => {
  if (!current) return;
  const id = current.id;
  $('classify').disabled = true;
  try {
    const { item } = await api('/api/memories/' + id + '/classify', { method: 'POST' });
    if ($('detail').open && current?.id === id) detail(item);
    await reload();
    resetPersonalPortrait();
  } catch (error) { notice(error.message); }
  finally { $('classify').disabled = false; }
};
$('delete').onclick = async () => { if (!current || !confirm('删除这条记忆和保存的原件？此操作无法撤销。')) return; try { await api('/api/memories/' + current.id, { method: 'DELETE' }); $('detail').close(); notice('已删除'); await reload(); resetPersonalPortrait(); } catch (error) { notice(error.message); } };
$('login-form').onsubmit = async event => { event.preventDefault(); clearPrivateUI(); try { await api('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token: $('token').value.trim() }) }); $('token').value = ''; await openWorkspace(); } catch (error) { notice(error.message); } };
$('logout').onclick = async () => { clearPrivateUI(); try { await api('/api/logout', { method: 'POST' }); location.reload(); } catch (error) { notice('退出请求未完成，请重试或关闭此浏览器窗口。'); } };
$('capture-form').onsubmit = async event => {
  event.preventDefault(); const file = $('file').files[0];
  if (file && file.size > 20 * 1024 * 1024) return notice('单个文件不能超过 20 MB');
  if (!file && !$('text').value.trim()) return notice('请填写文字或选择文件');
  const form = new FormData(); for (const key of ['title', 'text', 'note']) form.set(key, $(key).value);
  if (file) form.set('file', file);
  $('save').disabled = true; $('save').textContent = '正在上传并保存…';
  try { const result = await api('/api/memories', { method: 'POST', body: form }); $('capture-form').reset(); clearPhoto(); tab('library'); $('search').value = ''; semanticQuery = ''; await reload(); resetPersonalPortrait(); notice(result.message); detail(result.item); }
  catch (error) { notice(error.message + '，输入已保留，可以重试'); }
  finally { $('save').disabled = false; $('save').textContent = '保存记忆'; }
};
$('search').oninput = () => {
  clearTimeout(searchTimer);
  clearTimeout(processingTimer);
  requestNumber++;
  semanticQuery = '';
  searchTimer = setTimeout(reload, 250);
};
$('memory-search').onsubmit = event => {
  event.preventDefault();
  clearTimeout(searchTimer);
  semanticQuery = $('search').value.trim();
  reload();
};
async function refreshPersonalMemory() {
  if (signedInRole !== 'user') return;
  await reload();
}
$('refresh').onclick = refreshPersonalMemory;
$('endpoint').value = location.origin + '/api/memories';
$('image-endpoint').value = location.origin + '/api/memories/image';
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && !$('workspace').hidden) refreshPersonalMemory();
});
openWorkspace();

let graphRequest = 0;
async function loadGraph(reset = false) {
  const request = ++graphRequest;
  const admin = signedInRole === 'admin';
  const scope = reset && !admin ? '' : $('graph-scope').value;
  if (admin && !scope) {
    $('graph-status').textContent = '请选择一位评测用户。';
    $('graph-refresh').disabled = false;
    $('graph-canvas').replaceChildren();
    $('graph-list').replaceChildren();
    $('graph-source').hidden = true;
    $('graph-technical').hidden = true;
    $('graph-detail').textContent = '';
    return;
  }
  $('graph-refresh').disabled = true;
  $('graph-status').textContent = '正在读取本机 Neo4j…';
  const database = $('graph-database').value;
  $('graph-source').hidden = true;
  $('graph-source').onclick = null;
  $('graph-technical').hidden = true;
  $('graph-technical').open = false;
  $('graph-technical-data').textContent = '';
  try {
    const params = new URLSearchParams({ database, [admin ? 'user' : 'namespace']: scope });
    const data = await api('/api/graph?' + params);
    if (request !== graphRequest) return;
    if (!admin) {
      $('graph-scope').replaceChildren();
      const placeholder = new Option('选择数据空间', '');
      $('graph-scope').append(placeholder);
      for (const item of data.scopes) $('graph-scope').add(new Option(item.name + ' · ' + item.count + ' 节点', item.name));
      $('graph-scope').value = scope;
    }
    $('graph-status').textContent = scope
      ? `${data.nodes.length} 个节点 · ${data.edges.length} 条关系（截取视图）`
      : data.scopes.length ? '请选择数据空间。' : '这个视图还没有已同步的记录。';
    $('graph-canvas').replaceChildren(); $('graph-list').replaceChildren(); $('graph-detail').textContent = '点一条记忆，查看内容和关联。';
    const ns = 'http://www.w3.org/2000/svg'; const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', '0 0 700 500'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Neo4j 节点关系图，节点详情也可从下方列表选择');
    const positions = new Map(data.nodes.map((node, index) => { const angle = 2 * Math.PI * index / Math.max(data.nodes.length, 1); const radius = 100 + 100 * (index % 2); return [node.id, { x: 350 + Math.cos(angle) * radius, y: 250 + Math.sin(angle) * radius }]; }));
    for (const edge of data.edges) { const a = positions.get(edge.source), b = positions.get(edge.target); if (!a || !b) continue; const line = document.createElementNS(ns, 'line'); for (const [key, val] of Object.entries({ x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#adbfae','stroke-width':1 })) line.setAttribute(key, val); const title = document.createElementNS(ns, 'title'); title.textContent = edge.type; line.append(title); svg.append(line); }
    for (const node of data.nodes) {
      const point = positions.get(node.id); const group = document.createElementNS(ns, 'g');
      const circle = document.createElementNS(ns, 'circle'); circle.setAttribute('cx', point.x); circle.setAttribute('cy', point.y); circle.setAttribute('r', 8); circle.setAttribute('fill', node.labels.some(label => label === 'Episode' || label === 'ImportedEpisode') ? '#b47737' : '#396a50');
      const label = document.createElementNS(ns, 'text'); label.setAttribute('x', point.x + 11); label.setAttribute('y', point.y + 4); label.setAttribute('font-size', '11'); label.textContent = node.label.slice(0, 12); group.append(circle, label);
      const show = () => {
        const related = data.edges.filter(edge => edge.source === node.id || edge.target === node.id).map(edge => {
          const outbound = edge.source === node.id;
          const other = data.nodes.find(candidate => candidate.id === (outbound ? edge.target : edge.source));
          const label = graphRelationNames.get(edge.type) || semanticRelationNames.get(edge.type) || '相关记忆';
          return `${outbound ? '→' : '←'} ${label} · ${other?.label || ''}`;
        });
        const fields = node.properties;
        const content = fields.content || fields.text || fields.value;
        const date = fields.saved_at ? '保存时间 ' + fields.saved_at.slice(0, 10)
          : fields.timestamp ? '记录时间 ' + fields.timestamp.slice(0, 10) : '';
        $('graph-detail').textContent = [
          fields.title || fields.name || node.label,
          date,
          content && content !== node.label ? content : '',
          related.length ? '关联记忆\n' + related.join('\n') : '',
        ].filter(Boolean).join('\n\n');
        $('graph-technical').hidden = false;
        $('graph-technical').open = false;
        $('graph-technical-data').textContent = JSON.stringify(fields, null, 2);
        const source = node.properties.source_id;
        $('graph-source').hidden = !source;
        $('graph-source').onclick = source ? async () => {
          try {
            const { item } = await api('/api/memories/' + encodeURIComponent(source));
            detail(item);
          } catch (error) { notice(error.message); }
        } : null;
      };
      group.onclick = show; svg.append(group);
      const button = document.createElement('button'); button.className = 'secondary graph-node'; button.textContent = node.label; button.onclick = show; $('graph-list').append(button);
    }
    if (data.nodes.length) $('graph-canvas').append(svg);
  } catch (error) {
    if (request === graphRequest) {
      $('graph-status').textContent = error.message;
      $('graph-canvas').replaceChildren();
      $('graph-list').replaceChildren();
    }
  } finally {
    if (request === graphRequest) $('graph-refresh').disabled = false;
  }
}
$('graph-refresh').onclick = () => loadGraph();
$('graph-database').onchange = () => loadGraph(true);
$('graph-scope').onchange = () => loadGraph();

let photoURL = null;
function clearPhoto() { if (photoURL) URL.revokeObjectURL(photoURL); photoURL = null; $('photo-preview').removeAttribute('src'); $('photo-preview').hidden = true; $('photo-description').textContent = ''; }
$('file').onchange = () => {
  clearPhoto();
  const file = $('file').files[0];
  if (!file) return;
  $('photo-description').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB。点击保存后才会上传。`;
  if (!file.type.startsWith('image/')) return;
  photoURL = URL.createObjectURL(file); $('photo-preview').src = photoURL; $('photo-preview').hidden = false;
};
$('photo-preview').onerror = () => { $('photo-preview').hidden = true; $('photo-description').textContent += ' 此格式无法预览，仍可保存原件；建议使用 JPEG/PNG 进行 OCR。'; };

function showCredential(name, key) {
  $('new-user-key').value = `用户：${name}\n访问地址：${location.origin}\n个人密钥：${key}`;
  $('user-credential').hidden = false;
}
async function loadUsers() {
  try {
    const data = await api('/api/admin/users'); $('users-list').replaceChildren();
    for (const user of data.users) {
      const card = document.createElement('div'); card.className = 'memory-record';
      const title = document.createElement('p'); title.textContent = `${user.name} · ${user.role === 'admin' ? '管理员' : '普通用户'} · ${user.active ? '已启用' : '已停用'}`; card.append(title);
      const id = document.createElement('p'); id.className = 'hint'; id.textContent = '用户 ID：' + user.id; card.append(id);
      if (user.role !== 'admin') {
        for (const [action, label] of [['rotate', '重置个人密钥'], [user.active ? 'disable' : 'enable', user.active ? '停用用户' : '启用用户']]) {
          const button = document.createElement('button'); button.className = 'secondary'; button.textContent = label;
          button.onclick = async () => {
            if (!confirm(`${label}「${user.name}」？该用户现有登录会失效；重置密钥还需要更新手机快捷指令。`)) return;
            button.disabled = true;
            try { const result = await api(`/api/admin/users/${user.id}/${action}`, { method: 'POST' }); if (result.key) showCredential(user.name, result.key); await loadUsers(); }
            catch (error) { notice(error.message); button.disabled = false; }
          }; card.append(button);
        }
      }
      $('users-list').append(card);
    }
  } catch (error) { notice(error.message); }
}
$('create-user').onsubmit = async event => {
  event.preventDefault(); $('create-user-button').disabled = true;
  try {
    const result = await api('/api/admin/users', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: $('new-user-name').value }) });
    showCredential(result.user.name, result.key); $('create-user').reset(); await loadUsers(); notice('已创建独立用户，请保存个人密钥');
  } catch (error) { notice(error.message); }
  finally { $('create-user-button').disabled = false; }
};
$('users-refresh').onclick = loadUsers;
$('hide-user-key').onclick = () => { $('new-user-key').value = ''; $('user-credential').hidden = true; };

const experimentRequests = new WeakMap();
const portraitRequests = new WeakMap();
async function loadPortrait(path, target, options = {}) {
  const request = (portraitRequests.get(target) || 0) + 1;
  const generation = authGeneration;
  portraitRequests.set(target, request);
  target.hidden = !path;
  target.textContent = path ? '正在整理近况…' : '';
  if (!path) return;
  try {
    const { portrait } = await api(path, options);
    if (request !== portraitRequests.get(target) || generation !== authGeneration) return;
    if (portrait.status === 'ready' && portrait.items?.length) {
      const list = document.createElement('ul');
      list.className = 'portrait-items';
      for (const item of portrait.items) {
        const row = document.createElement('li');
        const headline = document.createElement('strong');
        headline.textContent = item.headline;
        row.append(headline, ' · ' + item.detail);
        list.append(row);
      }
      target.replaceChildren(list);
    } else if (portrait.status === 'ready') target.textContent = portrait.text;
    else if (portrait.status === 'empty') target.textContent = '保存几条记忆后，就能看到你的近况速览。';
    else target.textContent = '近况速览暂时无法生成，可以先查看下方的记忆记录。';
  } catch (error) {
    if (request === portraitRequests.get(target) && generation === authGeneration) target.textContent = error.message;
  }
}
function resetPersonalPortrait() {
  loadPortrait('', $('personal-portrait'));
}
$('generate-personal-portrait').onclick = async () => {
  const button = $('generate-personal-portrait');
  button.disabled = true;
  try {
    await loadPortrait('/api/portrait', $('personal-portrait'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ consent: true }),
    });
  } finally {
    button.disabled = false;
  }
};
const semanticRelationNames = new Map(Object.entries({
  LIKES: '喜欢',
  DISLIKES: '不喜欢',
  PREFERS: '偏好',
  ENJOYS: '喜爱',
  WORKS_AS: '职业',
  LIVES_IN: '居住地',
  HAS_SKILL: '技能',
  GOOD_AT: '擅长',
  STUDIES: '学习',
  HAS: '拥有',
  IS: '是',
}));
const graphRelationNames = new Map(Object.entries({
  HAS_SOURCE: '来自原始记录',
  INVOLVES: '涉及的人',
  HAPPENS_AT: '发生时间',
  OCCURS_AT: '发生地点',
  MENTIONS: '提到的人',
  ABOUT: '相关主题',
  NEXT: '后续事件',
}));

async function loadExperimentUsers(selectId, statusId) {
  const select = $(selectId);
  if (select.options.length > 1) return;
  $(statusId).textContent = '正在读取实验用户…';
  try {
    const { users, names = {} } = await api('/api/admin/experiment-profile');
    select.replaceChildren(new Option('请选择评测用户', ''));
    users.forEach((user, index) => select.add(new Option(names[user] || '评测用户 ' + (index + 1), user)));
    $(statusId).textContent = users.length ? '' : '未找到评测记录';
    if (selectId !== 'experiment-search-user' && users.length) {
      select.value = selectId === 'workflow-user' && users.includes($('experiment-user').value)
        ? $('experiment-user').value : users[0];
      select.onchange();
    }
  } catch (error) {
    if (signedInRole === 'admin') $(statusId).textContent = error.message;
  }
}

function flowElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function relatedEpisodeDetails(episode, currentText) {
  const related = flowElement('details', 'flow-related');
  const sameText = episode.text?.trim() === currentText?.trim();
  related.append(
    flowElement('summary', '', '查看对应事件'),
    flowElement('p', sameText ? 'hint' : '', sameText ? '对应事件与上方状态来自同一条记录。' : episode.text),
  );
  if (episode.date) related.append(flowElement('time', 'hint', episode.date));
  return related;
}

function workingFlow(states, transitions) {
  const timeline = flowElement('ol', 'flow-sequence working-flow');
  const chronological = [...states].sort((a, b) => (a.timestamp || '').localeCompare(b.timestamp || ''));
  const nodes = new Map(states.map(state => [state.node_id, state]));
  for (const state of chronological) {
    const item = flowElement('li', 'flow-state');
    const incoming = transitions.filter(edge => edge.to_node_id === state.node_id && nodes.has(edge.from_node_id));
    for (const edge of incoming) {
      const previous = nodes.get(edge.from_node_id);
      const label = ['从记录状态流转', previous?.state_name, edge.reason].filter(Boolean).join(' · ');
      item.append(flowElement('p', 'flow-transition', '↓ ' + label));
    }
    const box = flowElement('div', 'flow-box');
    if (state.timestamp) box.append(flowElement('time', 'hint', state.timestamp.replace('T', ' ')));
    const stateText = (typeof state.content === 'string' ? state.content : state.content?.text) || state.state_name || '未命名状态';
    box.append(flowElement('p', '', stateText));
    if (state.upload_id) box.append(uploadSourceButton(state.upload_id));
    if (state.related_episode) box.append(relatedEpisodeDetails(state.related_episode, stateText));
    item.append(box);
    timeline.append(item);
  }
  return timeline;
}

function procedureSteps(steps) {
  const list = flowElement('ol', 'flow-sequence');
  for (const step of steps || []) {
    if (!step || typeof step !== 'object') continue;
    const item = flowElement('li', 'flow-step');
    if (step.type === 'sequence') item.append(procedureSteps(step.steps));
    else if (step.type === 'branch') {
      item.append(flowElement('p', 'flow-box flow-decision', '判断：' + (step.condition || '按情况选择')));
      const branches = flowElement('div', 'flow-branches');
      for (const branch of step.cases || []) {
        const section = flowElement('section', 'flow-case');
        section.append(flowElement('p', 'flow-case-title', branch.when || '其他情况'), procedureSteps(branch.steps));
        branches.append(section);
      }
      item.append(branches);
    } else if (step.text) {
      item.append(flowElement('p', 'flow-box', step.text));
    }
    if (item.childElementCount) list.append(item);
  }
  return list;
}

function procedureFlow(rule) {
  const flow = flowElement('div', 'procedure-flow');
  const conditions = [].concat(rule.if_conditions || rule.if_condition || []).filter(Boolean);
  if (conditions.length) flow.append(flowElement('p', 'hint', '适用情况：' + conditions.join('、')));
  const tree = rule.tree ? procedureSteps([rule.tree]) : null;
  if (tree?.childElementCount) flow.append(tree);
  else {
    const steps = [].concat(rule.then_steps || []).filter(Boolean).map(text => ({ type: 'action', text }));
    flow.append(steps.length ? procedureSteps(steps) : flowElement('p', 'hint', '这条规则没有步骤记录。'));
  }
  return flow;
}

function uploadSourceButton(identity) {
  const button = flowElement('button', 'secondary', '查看原始记忆');
  button.type = 'button';
  button.onclick = async () => {
    button.disabled = true;
    try {
      const { item } = await api('/api/memories/' + encodeURIComponent(identity));
      detail(item);
    } catch (error) { notice(error.message); }
    finally { button.disabled = false; }
  };
  return button;
}

let personalWorkflowRequest = 0;
async function loadPersonalWorkflows() {
  if (signedInRole !== 'user') return;
  const request = ++personalWorkflowRequest;
  const generation = authGeneration;
  const target = $('workflow-graphs');
  target.replaceChildren();
  $('workflow-status').textContent = '正在读取你的任务与流程…';
  try {
    const data = await api('/api/workflows');
    if (request !== personalWorkflowRequest || generation !== authGeneration) return;
    $('workflow-status').textContent = '显示已保存的任务状态与方法记录。记录当时的状态不代表任务今天仍未完成。';
    const working = flowElement('section', 'experiment-flow');
    working.append(flowElement('h3', '', '任务状态 · Working'),
      flowElement('p', 'hint', '按记录时间展示；只有已记录的状态变化才显示连接箭头。'));
    if (!data.working.length) working.append(flowElement('p', 'hint', '暂无任务状态记录。新内容分类完成并确认后会显示在这里。'));
    for (const group of data.working) {
      const entry = flowElement('details', 'flow-related');
      entry.open = true;
      entry.append(flowElement('summary', '', group.nodes.at(-1)?.state_name || '任务记录'),
        workingFlow(group.nodes, group.transitions));
      working.append(entry);
    }
    const procedures = flowElement('section', 'experiment-flow');
    procedures.append(flowElement('h3', '', '方法与流程 · Procedural'));
    if (!data.procedural.length) procedures.append(flowElement('p', 'hint', '暂无方法与流程记录。保存的可复用方法经分类确认后会显示在这里。'));
    for (const rule of data.procedural) {
      const entry = flowElement('details', 'flow-related');
      entry.open = true;
      entry.append(flowElement('summary', '', rule.title || '方法记录'));
      if (rule.valid_from) entry.append(flowElement('time', 'hint', '记录时间 ' + rule.valid_from.replace('T', ' ')));
      if (rule.status === 'source_only') entry.append(flowElement('p', 'hint', '已保存方法原文，尚未提取条件和步骤。'));
      else entry.append(procedureFlow(rule));
      if (rule.source_text) entry.append(flowElement('pre', 'workflow-original', rule.source_text));
      if (rule.upload_id) entry.append(uploadSourceButton(rule.upload_id));
      procedures.append(entry);
    }
    target.append(working, procedures);
  } catch (error) {
    if (request === personalWorkflowRequest && generation === authGeneration) $('workflow-status').textContent = error.message;
  }
}
$('workflow-refresh').onclick = () => signedInRole === 'admin' ? $('workflow-user').onchange() : loadPersonalWorkflows();

function renderExperimentFlows(target, profile) {
  if (profile.working.length) {
    const section = flowElement('details', 'experiment-flow');
    section.open = true;
    section.append(flowElement('summary', '', '工作状态流转图 · Working'),
      flowElement('p', 'hint', '箭头表示原始数据里记录的状态变化。'));
    const content = flowElement('div');
    const latest = () => content.replaceChildren(workingFlow(profile.working, profile.working_transitions || []));
    if (profile.working_linked?.length) {
      const label = flowElement('label', '', '显示范围');
      const select = flowElement('select');
      select.add(new Option('最近记录的状态', 'latest'));
      select.add(new Option('有对应事件的状态', 'linked'));
      select.onchange = () => {
        if (select.value === 'linked') {
          content.replaceChildren(workingFlow(profile.working_linked, profile.working_transitions || []));
        } else latest();
      };
      label.append(select);
      section.append(label);
    }
    latest();
    section.append(content);
    target.append(section);
  }
  if (profile.procedural.length) {
    const section = flowElement('details', 'experiment-flow');
    section.open = true;
    const label = flowElement('label', '', '选择一条做法');
    const select = flowElement('select');
    profile.procedural.forEach((rule, index) => select.add(new Option(rule.title || '做法 ' + (index + 1), index)));
    const graph = flowElement('div');
    const show = () => graph.replaceChildren(procedureFlow(profile.procedural[Number(select.value)]));
    select.onchange = show;
    show();
    label.append(select);
    section.append(flowElement('summary', '', '操作流程图 · Procedural'), label, graph);
    target.append(section);
  }
}

async function displayExperiment(path, target, status, loading) {
  const request = (experimentRequests.get(target) || 0) + 1;
  experimentRequests.set(target, request);
  target.replaceChildren();
  const search = target.id === 'experiment-search-results';
  if (search) {
    $('experiment-answer').hidden = true;
    $('experiment-answer-text').textContent = '';
  }
  status.textContent = path ? loading : '';
  if (!path) return;
  try {
    const { profile, answer_ticket } = await api(path);
    if (request !== experimentRequests.get(target)) return;
    status.textContent = profile.semantic_error ? '偏好与概念暂时无法读取，其他内容已加载。' : '';
    const episodeTargets = new Map();
    const episodes = search ? profile.episodic.map((episode, index) => {
      const anchor = 'experiment-result-episode-' + index;
      if (episode.source) episodeTargets.set(episode.source, anchor);
      return { ...episode, anchor };
    }) : profile.episodic;
    const sections = [
      ['喜欢什么、了解什么', profile.semantic.map(fact => ({
        text: (semanticRelationNames.get(fact.relation) || fact.relation) + ' · ' + fact.entity,
        date: fact.source_timestamp,
        graphDate: fact.date,
        source: (fact.evidence || []).join('、'),
        original: fact.source_text,
      }))],
      ['做过什么', episodes],
      ['记下了哪些做法', profile.procedural.map(rule => ({
        title: rule.title,
        text: [
          ...([].concat(rule.if_conditions || rule.if_condition || []).filter(Boolean).map(condition => '适用情况：' + condition)),
          ...([].concat(rule.then_steps || []).filter(Boolean).map((step, index) => (index + 1) + '. ' + step)),
        ].join('\n'),
        date: rule.source_timestamp, showTimestamp: true, source: rule.meta?.source_message_id,
        original: rule.source_text,
      }))],
      ['任务进行到哪一步', profile.working.map(state => ({
        text: state.content?.text || state.state_name, date: state.timestamp, source: state.source_episode_id,
        relatedEpisode: state.related_episode,
      }))],
    ];
    if (search) {
      const count = sections.reduce((total, [, entries]) => total + entries.length, 0);
      const result = count ? '显示 ' + count + ' 条匹配记录。' : '没有找到匹配记录。试试补充一点上下文。';
      const fallback = profile.query_extraction && profile.query_extraction !== 'succeeded'
        ? 'DeepSeek 查询提取暂不可用，本次事件使用本地检索。' : '';
      status.textContent = [result, fallback, status.textContent].filter(Boolean).join(' ');
      if (!count) return;
    }
    const visible = target.id === 'experiment-profile' ? sections.slice(0, 2) : sections;
    for (const [label, entries] of visible) {
      const heading = document.createElement('h3');
      heading.textContent = label;
      target.append(heading);
      if (!entries.length) {
        const empty = document.createElement('p');
        empty.textContent = '暂无可展示的记录';
        target.append(empty);
      }
      for (const entry of entries) {
        const article = document.createElement('article');
        article.className = 'memory-record';
        if (entry.anchor) article.id = entry.anchor;
        if (entry.title) {
          const name = document.createElement('strong');
          name.textContent = entry.title;
          article.append(name);
        }
        const body = document.createElement('pre');
        const eventAnchor = episodeTargets.get(entry.relatedEpisode?.source);
        const duplicate = eventAnchor && entry.text?.trim() === entry.relatedEpisode.text?.trim();
        body.textContent = duplicate ? '这条事件也记录了当时的任务状态。' : entry.text || '';
        const metadata = document.createElement('p');
        metadata.className = 'hint';
        metadata.textContent = entry.showTimestamp && entry.date
          ? '原始记录时间 ' + entry.date.replace('T', ' ')
          : entry.date?.slice(0, 10) || '';
        article.append(body);
        if (entry.date) article.append(metadata);
        if (eventAnchor) {
          const link = flowElement('a', 'event-reference', '查看对应的事件结果');
          link.href = '#' + eventAnchor;
          article.append(link);
        } else if (entry.relatedEpisode) {
          article.append(relatedEpisodeDetails(entry.relatedEpisode, entry.text));
        }
        if (entry.original || entry.source || entry.graphDate) {
          const evidence = document.createElement('details');
          const summary = document.createElement('summary');
          summary.textContent = entry.original ? '查看原始记录' : '查看记录详情';
          evidence.append(summary);
          if (entry.original) {
            const original = document.createElement('pre');
            original.textContent = entry.original;
            evidence.append(original);
          }
          if (entry.source) {
            const source = document.createElement('p');
            source.className = 'hint';
            source.textContent = '记录编号 ' + entry.source;
            evidence.append(source);
          }
          if (entry.graphDate) {
            const graphDate = document.createElement('p');
            graphDate.className = 'hint';
            graphDate.textContent = '图谱入库或最近更新时间 ' + entry.graphDate.slice(0, 10);
            evidence.append(graphDate);
          }
          article.append(evidence);
        }
        target.append(article);
      }
    }
    if (search && answer_ticket) loadExperimentAnswer(answer_ticket, target, request);
  } catch (error) {
    if (request === experimentRequests.get(target) && signedInRole === 'admin') status.textContent = error.message;
  }
}

async function loadExperimentAnswer(ticket, target, request) {
  const generation = authGeneration;
  const currentSearch = () => request === experimentRequests.get(target) && generation === authGeneration;
  $('experiment-answer').hidden = false;
  const text = $('experiment-answer-text');
  text.textContent = '正在根据检索到的记忆回答…';
  try {
    const { answer } = await api('/api/admin/experiment-answer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticket }),
    });
    if (!currentSearch()) return;
    text.textContent = answer.status === 'ready' ? answer.text : '暂时无法生成回答，可以查看下方检索结果。';
  } catch (error) {
    if (currentSearch()) text.textContent = error.message;
  }
}

$('experiment-user').onchange = () => {
  const user = $('experiment-user').value;
  const path = user ? '/api/admin/experiment-profile?user=' + encodeURIComponent(user) : '';
  displayExperiment(path, $('experiment-profile'), $('experiment-status'), '正在读取实验记忆…');
  const portraitPath = user ? '/api/admin/experiment-portrait?user=' + encodeURIComponent(user) : '';
  loadPortrait(portraitPath, $('experiment-portrait'));
};
$('workflow-user').onchange = async () => {
  const target = $('workflow-graphs');
  const status = $('workflow-status');
  const user = $('workflow-user').value;
  const request = (experimentRequests.get(target) || 0) + 1;
  experimentRequests.set(target, request);
  target.replaceChildren();
  status.textContent = user ? '正在读取任务与流程…' : '';
  if (!user) return;
  try {
    const { profile } = await api('/api/admin/experiment-profile?user=' + encodeURIComponent(user));
    if (request !== experimentRequests.get(target)) return;
    renderExperimentFlows(target, profile);
    status.textContent = target.childElementCount ? '' : '这位评测用户暂时没有任务与流程记录。';
  } catch (error) {
    if (request === experimentRequests.get(target) && signedInRole === 'admin') status.textContent = error.message;
  }
};
$('experiment-search-user').onchange = () => displayExperiment('', $('experiment-search-results'), $('experiment-search-status'));
$('experiment-search-form').onsubmit = event => {
  event.preventDefault();
  const user = $('experiment-search-user').value;
  const query = $('experiment-query').value.trim();
  if (!user || !query) {
    $('experiment-search-status').textContent = '请选择评测用户，并输入问题。';
    return;
  }
  const params = new URLSearchParams({ user, q: query });
  displayExperiment('/api/admin/experiment-search?' + params, $('experiment-search-results'),
    $('experiment-search-status'), '正在搜索…');
};
