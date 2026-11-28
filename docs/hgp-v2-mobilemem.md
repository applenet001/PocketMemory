# HGP v2 · MobileMem 复现方案

各存储的字段来源、时间含义、操作和跨库追溯见 [HGP v2 存储字段与操作](hgp-v2-mobilemem-schema.md)。

以仓库里的 [HGP-](../HGP-/README.md) 为行为基准，用 `outputs/four-head-500` 初始化本地分类器。冻结 Qwen 编码器和基础 checkpoint，每个用户的四个分类头可从各自新增数据做增量更新。先取单用户样本的前 20 条做 smoke，再考虑扩大回放。

v2 的记忆和分类器增量只使用本次回放的 MobileMem 消息，以及之后用户实际新增的记忆。直接写入、情节归纳、Working 更新和 Knowledge Evolution 都在这批数据和相应的用户存储上运行。`four-head-500` 的原训练样本无需导入 v2。

上游源码以 [Ouan6/HGP- 的 `265c7c2`](https://github.com/Ouan6/HGP-/tree/265c7c28137e583bf79a15295b74720feece50e6) 为基准。四类 `memory/` 模块、CLI 和 LLM 客户端本地已有。补齐了原版 `train/` 的 5 个源码文件。上游示例 `data/evaluate/` 面向其自带评测，`pictures/` 是文档图片。MobileMem 回放使用自己的 text split 和 `four-head-500`，所以上游示例数据、图片以及未随上游发布的 BERT 权重和训练数据都无需加入 v2。

## Smoke20 样本和结果

- [20 条输入消息](../outputs/v2/smoke20/messages.jsonl) 来自 `person_CNDg9N6aMzcSxgkmAwDk43`，跨 5 个会话，从 `2024-06-01 09:03:22` 到 `2024-06-10 19:15:20`。角色分别为 8 条 `user`、8 条 `assistant` 和 4 条 `system`。
- 按原始 `timestamp` 排序，同一时间按 session ID、message ID 排序，选取首个有足够消息的用户的前 20 条非空消息。样本保留源时间字符串。事件写入和 evolution 以中国标准时间（UTC+08:00）解释这些时间。
- 样本 SHA-256 为 `3f8845f8f0ea471651d4caf89fa520a2a232b7f32b6783abf5125d5b2e4ac7a3`。数据源固定到 [MobileMem 官方 text split](https://huggingface.co/datasets/zjunlp/MobileMem/blob/989ba00900bddf4e0d4b0c7dd52f0840a7ee3a54/text/mobilemem_data.json)。
- [本次运行 HTML](../outputs/v2/smoke20/efdf5a601563/index.html) 和 [原始结果 JSON](../outputs/v2/smoke20/efdf5a601563/report.json) 包含 10 条事件、10 条 Working 状态、35 条语义关系、19 条程序规则。20 条触发分类复判并进入缓冲，更新轮次为 0，阈值为 `0.9`。

## 按 HGP- 写入和检索

让 v2 适配器把 MobileMem 消息和模型路由交给 `Memory.add(..., fast_mode=False)`。前三类调用 `Memory.search`，Working 直接从对应会话调用原版 `WorkingMemory.get_current_state()`。补齐分类器和存储对象的注入点，让直接写入、情节抽取和 evolution 共用同一套按用户隔离的存储。原始消息时间传给 HGP- 的 `session_date_time`，同时记录原始消息 ID 和会话 ID。将消息角色作为数据处理，来源为 `system` 的正文进入 LLM 时同样放在数据字段中。

`source_app` 取自 MobileMem 中 `role=system` 且 `name` 是已知 App 的消息。它作为来源元数据写入日志、事件和规则。直接写入的语义关系保留证据消息 ID，明确指向同一个 App 时可显示来源。QA 上下文可以为已有来源的证据加上 `[来源：App]`。向量继续编码正文，问题没有指定 App 时依然可以跨来源检索。Working 轨迹仍按会话保存，HTML 可在触发事件上显示 App 来源。 [来源提取](../scripts/mobilemem_source.py)

没有显式会话 ID 的 QA 先走原版 Episodic 检索。v2 用已召回 `episode_id` 在回放记录里定位触发的 `(session_id, state_id)`，将当时的 Working 状态和进度附到该事件正文之后。未关联状态的事件照常展示。关联只使用回放中的 ID，无额外 LLM 请求。要检查实际生成的问答上下文，可用 `run_hgp_v2_smoke.py --search-run <运行目录> --query <问题>`，Episodic 检索的用量另计入 `deepseek-search-usage.jsonl`。 [v2 查询入口](../scripts/run_hgp_v2_smoke.py)

| v2 存储位置 | 复现行为 |
| --- | --- |
| Neo4j `v2-episodic`，另保留用户目录下的 episode JSONL | 复用 `EpisodeStore` 写事件和关联图。按原版普通模式以最近 5 条 episode 抽取状态并更新 Working。检索复用图和向量混合搜索。 |
| Neo4j `v2-semantic` | 复用语义画像抽取、三元组、来源和原版合并逻辑。从直接路由、情节归纳及 evolution 写入同一用户的画像。 |
| `<运行目录>/<用户>/v2-procedural/procedural_rules.jsonl` | 复用原版规则抽取和 `ProceduralMemoryStore`，保留条件、步骤及向量检索。直接写入、情节抽取和 evolution 使用同一规则文件。 |
| `<运行目录>/<用户>/v2-working/<会话>/state_trajectory.json` | 写入一条 episodic 后，原版 `StateExtractor` 用最近 5 条 episode 推断当前状态和进度，再用 `start_state()` 或 `transition_to()` 写入会话轨迹。读取时对同一会话取当前状态。 |

Neo4j 数据按运行和用户设命名空间，文件也按运行分开。Smoke 从 `outputs/v2/smoke20/messages.jsonl` 读取前 20 条，每次执行写到 `outputs/v2/smoke20/` 下的新目录和独立命名空间。保留每条路由和各类写入结果。 [HGP- 写入与检索](../HGP-/memory/__init__.py)

## 记忆更新和删减规则

| 类型 | 触发条件 | 原版处理 |
| --- | --- | --- |
| Episodic | 一条新消息被路由到 episodic | 追加事件并建立与上一事件的 `NEXT` 链接。相似内容可以分别保留。情节归纳会标记已抽取事件，原事件仍可检索。 [事件写入](../HGP-/memory/episodic_memory/episode_store.py#L517) |
| Semantic | 写入三元组时，同一用户在同一命名空间已有同类关系 | DeepSeek 判断 `add`、`skip`、`update` 或 `delete_and_add`。可共存则追加，重复则跳过，更具体则替换指定旧关系，冲突则删除指定旧关系再写入新关系。删除关系后，仅在旧实体没有其他关系时移除该实体。没有同类关系，或合并请求失败时，按原版走新增路径。 [三元组写入](../HGP-/memory/semantic_memory/semantic_store.py#L66) |
| Procedural | 写入新规则时先查同条件同步骤，再查整体向量相似度，门槛为 `0.85` | 命中就复用已存规则 ID。通过去重后，若某条活跃规则的条件相似度至少 `0.8`、步骤相似度至多 `0.45`，旧规则标为 `deprecated`，新规则升为同组下一版本。旧规则仍留在 JSONL，向量检索会降低其分数并按版本组择优。其余规则作为新组写入。 [规则写入与检索](../HGP-/memory/procedural_memory/store.py#L289) |
| Working | 写入 episodic 后，从最近 5 条事件推断会话状态 | 首次调用 `start_state()`，之后用 `transition_to()` 更新当前状态并保留轨迹。状态和转换使用触发事件的原始时间，以 UTC+08:00 保存。加入新状态前，活跃项达到 9 条时，原版从历史项里优先归档已完成项，其次暂停项，再其次最早一项。归档将其标为非活跃，读取仍用同一会话的 `get_current_state()`。 [状态写入](../HGP-/memory/episodic_memory/episodic_memory.py#L951) · [归档逻辑](../HGP-/memory/working_memory/state_trajectory.py#L93) |

原版批量语义归纳的提示词要求至少 3 个相似项才提议 `merge`。执行器只处理 `add` 列表，`merge` 的结果会作为一条新增内容写入。批量返回的 `update` 和 `delete` 指令没有独立执行路径。新增内容写入三元组时，仍可能触发上表的关系合并。 [批量归纳与执行](../HGP-/memory/episodic_memory/episodic_memory.py#L504)

v2 加一条独立的提前归档规则。对 MobileMem 的 Todo 系统消息，保留原始任务 `metadata.id` 和布尔 `is_completed`。Todo 触发 Working 写入时，用该任务 ID 标注新节点。后续遇到该用户同一任务明确完成的消息时，只将匹配的活跃 Working 节点设为 `is_active=false` 并写回轨迹。`archive_trigger` 保留触发消息 ID、事件时间和可用的 Episodic ID。事件时间表示记录显示已完成的时间。找不到匹配任务时保留现有状态。每条消息仍按原版路由写入，原版状态提取和会话容量规则继续执行。QA 仍使用事件当时的状态名和进度，HTML 用活跃标记显示是否已归档。 [v2 提前归档](../scripts/run_hgp_v2_smoke.py)

Smoke 用受控输入核对语义替换、规则升版和 Working 归档。原版 Working 在内存中修改旧节点的归档标记，追加新节点时只重写新节点，重启后旧节点可能恢复为活跃。验收时记录重载后的状态。 [轨迹保存](../HGP-/memory/working_memory/state_trajectory.py#L357)

## 分类、fallback 和 adaptive threshold

- v2 的本地分类器用冻结的 Qwen3-Embedding-0.6B 生成归一化的 1,024 维向量，再用 [four-head-500](../outputs/four-head-500/config.json) 的 `head.pt` 输出四个独立 sigmoid 分数。HGP- 原版直接用 `memory_bert` checkpoint 预测同一组四标签。v2 按配套配置的 `0.5` 判定各标签命中，用 strongest-hit 处理 `none` 冲突，再用原版自适应阈值决定是否复判。`none` 表示没有前三类记忆，最终用空标签集合表达无记忆。零命中进入复判。 [原版 BERT 推理](../HGP-/train/infer.py#L15)
- HGP- 原版存储和检索继续使用 512 维 `distiluse-base-multilingual-cased-v2`。Qwen 的 1,024 维向量只用于本地分类。 [HGP- 图向量配置](../HGP-/memory/core/episodic_graph_builder.py#L10)
- 复用 HGP- 原版 fallback prompt 和 `labels` 解析，沿用本地零命中或最高置信度低于阈值时复判的条件。原版 prompt 描述了 Working，但最终允许列表和解析都只接受 `episodic`、`semantic`、`procedural`、`none`。因此原样复现时，Working 由 episodic 状态抽取产生。 [原版分类与 fallback](../HGP-/train/main.py)
- 按每个用户的时间顺序逐条回放。先用当时的分类头路由该条消息，再将通过解析和校验的 LLM 复判标签加入增量缓冲。沿用原版累计 100 条触发更新尝试并去重的方式。适配到 Qwen 时，从当前的四个分类头继续训练，训练输入使用当前用户已处理消息中的合格样本。 [原版缓冲与复判](../HGP-/train/main.py) · [原版增量更新](../HGP-/train/infer.py)
- 使用 HGP- 原版 `AdaptiveThresholdManager` 和独立用户状态。公式为 `round(max(0.5, 0.9 × exp(-0.05 × t)), 4)`。从第 0 轮开始，每次成功训练并保存新分类头后同步推进一轮，随后用新头和新阈值处理下一条。数据时间控制处理顺序，成功更新次数控制阈值。 [原版阈值](../HGP-/train/adaptive_threshold.py)

## DeepSeek 调用

- 分类复判调用原版 `train.main.call_llm_for_classification()`。通过 OpenAI 兼容客户端向 DeepSeek 发一条当前消息和原版分类 prompt，温度为 `0.2`，从响应解析 `labels`。原版通用客户端默认模型为 `deepseek-chat`，运行时记录实际模型。 [原版分类](../HGP-/train/main.py#L19) · [客户端](../HGP-/llm_api.py#L11)
- Working 推断由 episodic 写入触发。原版 `StateExtractor` 在缓存未命中时把最近的 episode 发给 DeepSeek，温度为 `0.3`，解析阶段和进度等状态信息。读取当前 Working 状态只访问本地轨迹。 [原版状态抽取](../HGP-/memory/episodic_memory/state_extractor.py) · [当前状态读取](../HGP-/memory/working_memory/working_memory.py#L94)
- 原版状态抽取在同一次响应里返回 `priority`（1–10，表示紧急程度）和 `confidence`（0–1，表示对状态推断的自评）。解析失败时使用原版回退值 `5` 和 `0.5`。Working 的归档仍按完成状态、暂停状态和时间选择。分类复判只返回标签，路由的置信度取 Qwen 分类头的最高 sigmoid 分数。 [原版状态字段](../HGP-/memory/episodic_memory/state_extractor.py#L184) · [原版分类触发](../HGP-/train/main.py#L99)
- 原版语义画像、事件结构抽取、程序性规则和情节归纳沿用各自的 LLM 调用。Smoke 记录每个 DeepSeek 响应的模型和用量，并把可观察的写入失败记入消息日志。 [语义抽取](../HGP-/memory/semantic_memory/semantic_memory.py) · [事件抽取](../HGP-/memory/core/llm_extractor.py) · [规则抽取](../HGP-/memory/procedural_memory/procedural_memory.py) · [情节归纳](../HGP-/memory/episodic_memory/episodic_memory.py#L356)

### 与 MobileMem Text 论文对比的 token 口径

按论文表 1 的记忆构建口径，v2 用 MemBase 原生 token monitor 对每次请求的消息和返回正文计数，使用本仓库 RAG 和 Long Context 配置采用的本地 Qwen tokenizer。构建包括分类复判、记忆抽取、合并、Working 状态推断和回放结束后的归纳与 evolution。问答、裁判、检索阶段和本地向量编码不计入构建用量。每次请求同时保存 DeepSeek API 的 `usage` 原始字段，包括输入、输出及缓存明细，供账单核对。两套计数各自汇总到 `report.json` 的 `deepseek_construction`。 [MobileMem Text 论文，表 1](https://arxiv.org/html/2608.13606#S4.T1) · [DeepSeek 响应用量字段](https://api-docs.deepseek.com/api/create-chat-completion/)

Smoke20 报告只覆盖 20 条消息。全量对比时先按用户合计完整轨迹的构建用量，再做逐用户等权平均，以千 token 展示。对比时同时标注 LLM backbone。美元金额单列为公开低峰／高峰费率估算区间。缺少实际用量或费率时，报告保留已知的 token 计数。 [DeepSeek 官方定价](https://api-docs.deepseek.com/quick_start/pricing/)

全量每个用户独立写入 Construct、Search、Eval 回答和 Eval Judge 的请求日志。失败或缺少用量时保留请求记录，汇总把无法定价的请求单列。两个用户的 Construct 都完成后，运行目录的 `construct-token-comparison.json` 提供按 MemBase monitor 计算的 `In. Tok.` 与 `Out. Tok.` 逐用户等权平均，包括原始 token 数和千 token。若有请求缺少 monitor 计数，文件标识缺失并提供已观测用量的平均值。Companion 的 v2 上传 API 用量按账号另存，使用上传 ID 对应请求。

## 情节归纳与 Knowledge Evolution

1. **自动语义归纳**。沿用 `Memory.add` 在成功写入每 20 条 episodic 后触发 `extract_to_semantic` 的行为。Smoke 同步执行，保留原版至少 5 条待抽取事件和合并逻辑，记录取到的事件、抽取和写回结果。 [自动归纳入口](../HGP-/memory/__init__.py#L372)
2. **情节转程序性规则**。回放结束时单独调用原版 `extract_to_procedural`，从成功任务序列提取、去重、写回规则。记录缺少成功状态或步骤的情况。 [原版程序性抽取](../HGP-/memory/episodic_memory/episodic_memory.py#L647)
3. **Knowledge Evolution**。在回放结束后显式调用 `run_knowledge_evolution`。沿用原版频次门槛 3 和 30 天窗口，复用偏好、习惯、成功流程的模式检测与写入。把窗口时钟锚定在该用户最后一条已回放消息，并统一时间比较，避免历史数据按机器当前日期被过滤。把原实现中的默认用户、默认命名空间和全局规则 store 接到 v2 当前用户，并记录检测结果、写回和去重情况。原版过程规则检测读取 `status`／`execution_status` 和 `how`／`steps`，回放时核对这些字段是否来自实际存入的 episode，避免用预期规则补齐缺失证据。 [原版 evolution](../HGP-/memory/knowledge_evolution.py)

## Smoke 验收与全量

1. 按顺序处理 20 条，记录每条的路由、原版模块返回的阈值、LLM 复判、写入状态和错误。核对两个 Neo4j 库和两个文件存储的实际内容，按直接写入、情节归纳和 evolution 汇总。
2. 验证 episodic 状态更新写入对应会话的 Working 轨迹，并能从同一会话读回当前状态。验证 evolution 实际读取了窗口内的本用户事件。若数据没有足够重复证据，报告零新增和原因，并用受控例子单独验证 evolution 能写回两类目标。
3. 用这 20 条覆盖的事实、事件和操作规则检查实际存储。Working 展示使用会话当前状态，并在 HTML 中将状态关联到触发它的事件。官方题只有在其证据都落在样本中时才用于 smoke 评分，避免把截断带来的缺失当成检索失败。
4. 核对重新加载和重试结果，并验证成功更新后，下一条消息使用新分类头和新阈值。20 条 smoke 用于验证复判和增量缓冲，原版 100 条的更新门槛沿用。全量回放使用独立命名空间。

回放入口为 [run_hgp_v2_smoke.py](../scripts/run_hgp_v2_smoke.py)，默认选取前 20 条消息。完成后用 [render_hgp_v2_smoke.py](../scripts/render_hgp_v2_smoke.py) 将运行目录的 `report.json` 渲染为 `index.html`。

全量入口 [run-hgp-v2-mobilemem.ps1](../scripts/run-hgp-v2-mobilemem.ps1) 按 Construct → Search → Eval 调度，每个阶段分别处理两条用户轨迹。[HGP v2 layer](../scripts/hgp_v2_membase_layer.py) 注册到本仓库的 MemBase，复用原有 runner 和已应用的 traced patch。Search 仍对每类最多取 5 条，命中的 Episodic 随附事件当时的 Working 状态。Eval 使用 MemBase 的默认问答和 MobileMem 判分提示，由 DeepSeek Flash 回答，DeepSeek V4 Pro 判分。原生 monitor 在各用户的 `construct/token_cost.json` 输出计数。Eval 的 patch 日志写到各用户的 `eval/deepseek-patch-usage.jsonl`，与按回答和 Judge 分开的 API 日志一道保留。

两用户全量使用顺序适配。用户 2 的 Construct 从用户 1 完成时的分类头和阈值开始，继承训练轮次计数以延续阈值曲线。用户 2 的复判缓冲、训练样本、更新历史和各类记忆单独存放。初始 checkpoint、轮次和阈值写入用户 2 的 `construct/adaptive_start.json`。用户 1 若没有成功更新，用户 2 使用初始权重和阈值。这种设置的用户 2 结果包含跨用户适配。
