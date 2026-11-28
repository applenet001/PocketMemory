# HGP v2 存储字段与操作

本文记录 MobileMem Text 全量经 MemBase 调度时的字段映射和操作。HGP v2 使用两个 Neo4j 数据库，Working 和 Procedural 写入本地文件。每个用户有独立的 Construct 目录。Neo4j 的两个数据库共用同一个用户运行命名空间。 [全量入口](../scripts/run_hgp_v2_membase.py) · [存储绑定](../scripts/run_hgp_v2_smoke.py#L248)

| 记忆或记录 | 存储位置 | 主键或查找键 |
| --- | --- | --- |
| 输入及路由 | `construct/turns.jsonl` | `message_id`，会话用 `session_id` |
| Episodic | Neo4j `v2-episodic` 和 `construct/episodes.jsonl` | 图节点 `(namespace, episode_id)`；文件 `id` |
| Semantic | Neo4j `v2-semantic` | `(namespace, user_id, 关系类型, 实体名)` |
| Procedural | `construct/v2-procedural/procedural_rules.jsonl` | `id`；版本组用 `version_group_id` |
| Working | `construct/v2-working/<session_id>/state_trajectory.json` | `(session_id, node_id)` |

`namespace` 是 `v2_<当前用户 construct 绝对路径的 SHA-256 前 12 位>_<person.id>`。同一用户的全量、Smoke 和重跑各有自己的范围。构建完成后可在 `construct/report.json` 的顶层读取 `namespace`。

## MobileMem 字段进入 v2 的位置

| MobileMem Text 原字段 | v2 字段和用途 |
| --- | --- |
| `person.id` | `turns.user_id`、记忆元数据的 `user_id`、Semantic 的 `User.user_id`、`namespace` 后缀。MemBase 用 `mobilemem-<person.id>` 作为轨迹 ID。 |
| `sessions[].id` | 每条消息的 `session_id`，用于 Working 文件路径、事件/规则元数据和回溯。 |
| `sessions[].messages[].id` | `turns.message_id`、`episodes.metadata.message_id`、直接提取规则的 `meta.message_id`；已知 App 的直接 Semantic 写入可以保存这个 ID 作为证据。 |
| `messages[].content` | 原文进入 `turns.content`、分类编码器，并交给命中的记忆写入路径。Episodic 保存事件正文；Semantic 和 Procedural 从正文抽取结构。空白正文跳过。 |
| `messages[].timestamp` | 原始字符串进入 `turns.timestamp` 和相关元数据的 `timestamp`。带时区的规范值存为 `session_date_time`，供事件、Working 状态与转换使用。无时区按中国标准时间 UTC+08:00，显式时区转为 UTC+08:00。 |
| `messages[].role`、`name` | 保留在消息及相关元数据。只有 `role=system` 且 `name` 为已知 App 时，才派生 `source_app`。当前名单含 Bill、Calendar、Document、Note、Screen Memo、Todo、Voice Memo。 |
| Todo 的 `messages[].metadata.id`、`is_completed` | Todo ID 为非空字符串且完成值为布尔时，分别映射成 `todo_id`、`is_completed`。它们用于定位同一个任务和触发归档。 |
| `messages[].created_at`、`side_note`、其他 App 原始 `metadata` | 完整记录仍在输入数据。MemBase 读取消息元数据，v2 回放层只提取本表列出的消息字段和 Todo 字段。Todo 的 `due_date`、`priority`、`is_finished` 也没有直接映射为 Working 的时间、优先级或活跃状态。 |
| `sessions[].event_id`、`started_at`、`ended_at`，`person` 的画像字段，`graphs` | 保留在数据集/标准化输入中。v2 记忆构建使用消息回放，Working 的 `event_id` 由会话键生成。 |
| `question_type_toolbook` 中的问题、`golden_answers`、`source_evidences` 等 | MemBase 负责 Search/Eval 的 `qa_pair`。Search 的查询是 `question`，QA 根据召回记忆回答，Judge 才看标准答案。问题和标准答案不参与 Construct。 |

消息按规范事件时间排序，同刻以会话 ID、消息 ID 打破平局。`turns.jsonl` 每条还保存 `labels`、`local_labels`、`scores`、分类来源 `source`、`max_confidence`、`threshold`、`queued`、`memory_types`、`details`、`success`，以及发生时的 `working_archived`。处理完成的非空消息即使归为 `none` 也保留日志。 [数据读取](../MemBase/membase/datasets/mobilemem.py) · [回放映射](../scripts/hgp_v2_membase_layer.py#L32) · [App 与 Todo 映射](../scripts/mobilemem_source.py)

## Episodic

| 字段或结构 | 来源及含义 |
| --- | --- |
| `Episode.episode_id` / JSONL `id` | HGP 新生成的事件 ID；MobileMem 原 ID 在 JSONL 的 `metadata.message_id`。 |
| `Episode.content`、JSONL `content` | MobileMem 原始消息正文。 |
| `Episode.timestamp`、JSONL `timestamp` | 消息发生时间精确到秒及 UTC+08:00，例如 `2024-06-01T09:03:22+08:00`。原始时间仍可在 `metadata.timestamp` 查到。 |
| `Episode.source_app` | 从已知系统发送者派生的来源。图上有 `(namespace, source_app)` 复合索引；JSONL 保存为 `metadata.source_app`。 |
| `Episode.event_type`、`execution_status` | 普通回放入口通常写入 `user_message` 和 `unknown`；原消息的角色保存在 `metadata.role`。结构化任务成功状态和动作序列属于 HGP 的其他写入路径。 |
| `Episode.embedding`、JSONL `embedding` | 本地 DistilUSE 的 512 维存储向量。来源 App 是元数据。 |
| `Episode.updated_at` | 最近写入图的数据库时间。 |
| `Actor`、`INVOLVES` | Actor 字段为 `namespace`、`name`、`role`、`embedding`。普通回放由 HGP 给事件设置 `actors=["user"]`；MobileMem 实际发送人和角色还可查文件元数据。 |
| `Time`、`HAPPENS_AT` | 日期节点的 `timestamp` 是日期键，例如 `2024-06-01`，另有 `namespace`、`value`、`year`、`month`、`day`、`granularity`。事件的时分秒在 `Episode.timestamp`。 |
| `Person`、`Topic`、`Location` | DeepSeek 从事件正文提取的人物、主题、地点。节点使用 `namespace`、`name`、`embedding`，地点另有 `type`。事件分别用 `MENTIONS`、`ABOUT`、`OCCURS_AT` 连接。出现相应提取结果时才建立。 |
| `NEXT`、JSONL `prev_episode_id` / `next_episode_id` | 连接同一用户运行中前后写入的事件，可跨原始会话。JSONL 追加新行；前一条的 `next_episode_id` 在运行中的缓存可更新，图上的 `NEXT` 和后条文件记录的 `prev_episode_id` 更适合追溯。 |
| JSONL `actors`、`context`、`action_sequence`、`extracted_semantic`、`extracted_procedural` | HGP 的事件结构及抽取标记。抽取标记会更新运行内的事件缓存；初次写出的 JSONL 行没有随标记重写，不能据它核对最终抽取状态。 |

**写入和去重。** 图按 `(episode_id, namespace)` 执行 `MERGE`，然后写入属性；重复使用同一键可以更新同一个图节点。正文相近的不同事件 ID 各自保留。Semantic 归纳会保留来源事件。**检索。** 原版混合搜索结合图候选和本地向量候选，按事件 ID 融合，最终取最多 5 个。图时间筛选使用 `Episode.timestamp`。 [图结构与索引](../HGP-/memory/core/episodic_graph_builder.py) · [事件文件与写入](../HGP-/memory/episodic_memory/episode_store.py)

## Semantic

| 字段或结构 | 来源及含义 |
| --- | --- |
| `User.user_id`、`User.namespace` | MobileMem `person.id` 和该用户在本次运行中的命名空间。 |
| `User.created_at`、`User.updated_at` | 用户节点首次和最近写入的数据库时间。 |
| `Entity.name`、`Entity.type`、`Entity.namespace`、`Entity.created_at` | 抽取后的实体名、实体类型、该用户运行范围和实体首次入库时间。实体向量 `embedding` 在首次写入有可用向量时保存。 |
| `(User)-[r]->(Entity)` 的关系类型 | 由文本抽取的关系，如 `LIKES`；存储时转大写，空格转下划线。 |
| `r.confidence` | 语义抽取得到的三元组置信度，与分类头的最高 sigmoid 分数各自独立。 |
| `r.evidence_episode_ids` / 查询输出 `evidence_ids` | 来源引用。直接路由且已知 App 时使用原始 **消息 ID**，从 `turns.jsonl` 回查。实际有 **Episode ID** 时可在 Episodic 查 `Episode.timestamp`。未传证据的路径可能使用 `manual_<user_id>`。批量归纳和 Knowledge Evolution 当前不写入逐条证据 ID。 |
| `r.created_at`、`r.last_seen_at` | 首次和后续匹配写入关系的数据库时间，不提供证据发生时间。新建关系可以没有 `last_seen_at`。 |
| 检索输出 `source_app`、`similarity` | v2 从证据 ID 和消息日志还原可确认的单一来源 App；相似度是查询时计算的结果。 |

**更新和删除。** 同一用户、同一命名空间内，按同类关系找旧项。DeepSeek 可返回 `add`、`skip`、`update` 或 `delete_and_add`。后两种可以删除指定旧边再写入新边，无其他关系的旧实体可被清理。相同三元组通过 Neo4j `MERGE` 复用同一条边，匹配更新会写入这次的证据列表与置信度。合并提示只传新旧关系文本；直接回放的先后顺序提供处理顺序，证据时间没有进入冲突决策。不同关系类型如 `LIKES` 和 `DISLIKES` 在这一操作中各自比较。 [合并、去重与写回](../HGP-/memory/semantic_memory/semantic_store.py#L66) · [来源引用](../HGP-/memory/semantic_memory/semantic_memory.py#L191)

**归纳与演化。** 每累计写入 20 条 Episodic，HGP 尝试从最多 20 条待抽取事件归纳长期知识，需要至少 5 条候选。批量归纳的执行器处理 `add` 列表，具体入图仍会触发三元组合并。整条轨迹结束后，Knowledge Evolution 以最后一条消息为时钟，回看 30 天，从事件正文找出现至少 3 次的偏好或习惯，筛查重复后尝试写回。**检索。** 从该用户的候选关系中读取最多 100 条，再比较问题和 `relation + entity` 的向量，相似度门槛 `0.15`，最多 5 条。 [自动归纳](../HGP-/memory/__init__.py#L389) · [Evolution](../HGP-/memory/knowledge_evolution.py) · [Semantic 向量检索](../HGP-/memory/semantic_memory/semantic_memory.py#L261)

## Procedural

| 字段 | 来源及含义 |
| --- | --- |
| `id`、`title`、`tags` | HGP 规则 ID 和从正文提取的标题、标签。 |
| `if_conditions`、`then_steps`、`tree` | 适用条件、执行步骤及规则树。提取结果缺少可用树时，可以从步骤生成顺序树。 |
| `source_text`、`meta` | 提取输入与来源信息。直接路由的 `meta` 可包含消息/会话 ID、原始时间、规范事件时间、发送者、角色和 `source_app`。由事件或 Evolution 归纳的规则，其来源形状随路径而异。 |
| `embedding` | 由条件和步骤拼成规范文本再编码为 512 维。用于相似规则查找和 QA 检索。 |
| `version_group_id`、`version`、`status`、`supersedes`、`conflict_type` | 规则版本组、版本号、状态、被替代规则 ID 和冲突类型。 |
| `valid_from`、`valid_to`、`last_hit_at` | 版本生效/结束和使用统计的记录时间。源事件时间查 `meta.timestamp` 或 `meta.session_date_time`。 |
| `usage_count`、`success_count`、`priority` | HGP 的规则使用统计和优先级。 |

**去重和升版。** 先按相同条件和步骤复用已有规则 ID，再以整体向量相似度 `≥0.85` 识别同义规则。条件相似度至少 `0.8`、步骤相似度至多 `0.45` 时可触发冲突升版，旧规则标为 `deprecated`，新规则成为同组下一版本，规则文件保留历史版。回放结束还尝试从有成功状态和动作序列的 Episodic 归纳程序规则。**检索。** 查询相似度门槛为 `0.3`，历史版本降权，每个版本组取一条得分最高的规则，最多返回 5 条。 [规则数据结构和操作](../HGP-/memory/procedural_memory/store.py) · [从事件提取](../HGP-/memory/episodic_memory/episodic_memory.py#L647)

## Working

| 字段 | 来源及含义 |
| --- | --- |
| `event_id`、`node_id` | `event_id` 为 `session_<session_id>`，状态节点 ID 如 `state_0000`。状态 ID 在各会话各自编号。 |
| `state_name`、`priority`、`content.progress`、`content.text` | 原版从最近 5 条该用户的 Episodic 推断状态、优先级和进度，再写入当前消息所在会话。进度形如 `1/3`；`content.text` 取状态抽取器使用的一条事件正文。 |
| `timestamp`、转换的 `timestamp` | 触发状态更新的事件时间，保存到秒及 UTC+08:00。 |
| `source_episode_id` | 原版 MobileMem 回放常写固定值 `episodic`。实际的事件到状态关联来自 `turns.jsonl` 的 `details.episodic.episode_id`、`session_id`、`details.working.node_id`。 |
| `is_active`、`duration_seconds`、`access_count`、`last_access` | 活跃标记、可选时长及访问统计。访问时间是运行时间。 |
| `content.todo_id`、`content.archive_trigger` | Todo 任务 ID。由完成消息触发提前归档时记录 `message_id`、该消息的 `event_timestamp`，以及可用的 `episode_id`。 |
| `transitions[]` | `from_node_id`、`to_node_id`、`timestamp`、`reason`、`context`，连接会话相邻状态。 |

**更新和归档。** 首次调用 `start_state()`，之后用 `transition_to()`。每个会话的活跃状态达到 9 条时，原版尝试归档旧状态，依次优先已完成、已暂停、最早的历史状态。历史节点仍留在轨迹。原版容量归档在内存标记旧节点，写新节点时可能没有一并保存旧标记，重载后旧节点可能仍显示活跃。v2 的 Todo 提前归档则显式保存同一任务匹配状态的 `is_active=false` 和触发证据。**检索。** QA 先命中 Episodic，再依照回放记录定位该事件对应的历史 Working，附上状态名和进度。 [状态推断与写入](../HGP-/memory/episodic_memory/episodic_memory.py#L865) · [轨迹存储](../HGP-/memory/working_memory/state_trajectory.py) · [Todo 归档与 Working 关联](../scripts/run_hgp_v2_smoke.py#L182)

**推断缓存。** 原版 `StateExtractor` 在进程内缓存最多 100 份状态分析结果。键取传入事件列表最后一项的 ID。命中可以复用分析、跳过 DeepSeek 推断；超出容量时移除最早放入的分析结果。持久化轨迹及其活跃状态有各自的规则。 [状态分析缓存](../HGP-/memory/episodic_memory/state_extractor.py#L85)

## 验证与追溯

- **从消息追到事件与状态。** 用 `turns.jsonl.message_id` 定位原文，读 `details.episodic.episode_id` 查 Neo4j 或 `episodes.jsonl.id`。若同一条 turn 有 `details.working.node_id`，用 `session_id` 和该状态 ID 打开对应轨迹。
- **从语义关系查来源时间。** `evidence_episode_ids` 中的值如果是消息 ID，在 `turns.jsonl` 查原始时间；如果是事件 ID，在 `v2-episodic` 查 `Episode.timestamp`。只有 `manual_<user_id>` 时无法定位单条来源证据。
- **区分 Search 和 QA 内容。** MemBase 的 Search 结果包括 `user_id`、`qa_pair` 和 `retrieved_memories`，每条检索结果的 `metadata.memory` 保存 HGP 结果。QA 将 Episodic 正文最多放入前 200 字符、Semantic 放入抽取关系和实体最多 100 字符、Procedural 格式化为规则标题以及前两个条件和前两个步骤。可用的来源 App 和事件相关的 Working 随结果进入上下文。 [检索适配](../scripts/hgp_v2_membase_layer.py#L69) · [QA 上下文](../HGP-/memory/__init__.py#L52)
- **查看用量和模型状态。** `construct/deepseek-usage.jsonl` 保留 API 用量与 MemBase monitor 计数，`construct/token_cost.json` 为 MemBase 汇总。增量权重与阈值在 `head-round-*.pt` 和 `threshold_state.json`。用户 2 的 `adaptive_start.json` 记录所继承的用户 1 定稿权重和阈值。Search、回答和 Judge 单独记录用量。 [调度与记录路径](../scripts/run_hgp_v2_membase.py)
- **删除范围。** MemBase 的 v2 layer 没有通用 `delete(memory_id)` 或 `update(memory_id)` 实现。语义替换、规则升版和状态归档由各自存储处理。若管理上清理某次运行，图数据与本地文件分别存在，需用该次运行的精确 `namespace` 核对范围。 [Layer 接口](../scripts/hgp_v2_membase_layer.py#L88)

查看事件时间时，在 Neo4j Browser 切到 `v2-episodic`，用实际命名空间设置参数：

```cypher
MATCH (e:Episode {namespace: $ns})
RETURN e.episode_id, e.timestamp, e.source_app, e.updated_at
ORDER BY datetime(e.timestamp)
LIMIT 25
```

查看语义及证据时，切到 `v2-semantic` 并使用同一个 `$ns`：

```cypher
MATCH (u:User {namespace: $ns})-[r]->(e:Entity {namespace: $ns})
RETURN u.user_id, type(r) AS relation, e.name AS entity,
       r.evidence_episode_ids, r.created_at, r.last_seen_at
LIMIT 25
```
