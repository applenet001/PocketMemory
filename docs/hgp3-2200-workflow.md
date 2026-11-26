# PocketMemory HGP3-2200 数据、训练与端到端评测

HGP3-2200 用 2,200 条带标签的消息训练记忆路由器。下游实验用该路由器逐条处理 MobileMem 对话，写入 HGP 的四类记忆，再按问题检索、生成答案并评分。

```mermaid
flowchart LR
    A[PerLTQA、proScript、MobileMem 消息] --> B[分组划分与标注]
    B --> C[冻结 Qwen 编码器并训练两个分类头]
    C --> D[Construction 构建用户记忆]
    M[MobileMem 对话与问题] --> D
    D --> E[Search 按问题检索]
    E --> F[Evaluation 生成答案并评分]
```

## 1. 构建分类数据

| 来源 | Train | Val | Test | 样本和标签 |
| --- | ---: | ---: | ---: | --- |
| 中文 PerLTQA | 1,000 | 97 | 84 | 人物描述和关系映射为 `semantic`，事件映射为 `episodic` |
| proScript 中文改写 | 400 | 54 | 58 | 将英文流程改写成中文方法回答，映射为 `procedural` |
| MobileMem | 800 | 192 | 192 | 逐条会话消息，由 DeepSeek 标注主类别和 working |
| **合计** | **2,200** | **343** | **334** | 总计 2,877 条 |

准备脚本读取 `data/external/PerLTQA/Dataset/zh/perltmem.json`、`outputs/proscript-dialogue-600/synthetic.jsonl` 和 `data/text/mobilemem_data.json`。proScript 上游先生成中文改写，再单独调用模型检查步骤、顺序和事实保真。2200 配方只接受自动复核通过、无需人工复核且未标记源问题的记录，并沿用上游划分。要复现历史数据，应使用当时的合成结果和标注，重新调用模型可能改变数据及其哈希。[数据准备脚本](../scripts/prepare_hgp3_2200.py) · [proScript 改写脚本](../scripts/synthesize_proscript.py)

划分使用种子 42 和基于 ID 的确定性哈希排序。PerLTQA 按人物分组，各划分的 semantic 样本依次为 500、50、50。proScript 按场景分组。MobileMem 从每位用户保留约五分之一的 session，再把完整保留 session 分给 val 和 test，各 192 条消息。从其余 session 的候选池中保留 800 条训练消息。准备和训练均检查重复 ID、跨划分分组及规范化正文重叠。MobileMem 的隔离单位是 session，同一用户可以出现在多个划分。

### 标签与人工复核

- 主类别四选一，`semantic` 对应稳定事实、属性或关系，`episodic` 对应具体事件和经历，`procedural` 对应可复用方法或规则，`null` 表示没有合适的主记忆。
- `working_state` 独立判断当前任务、进展、阻碍或待跟进事项，可与任一主类别同时成立。PerLTQA 和 proScript 的 `working_state=null` 表示缺少监督。Working 头的 800 条训练监督全部来自 MobileMem。
- MobileMem 标注使用 `deepseek-flash`、关闭 thinking、温度为 0。每项请求包括匿名编号、角色和目标消息正文，返回主类别、working、是否不确定和理由。默认每批 8 条、12 个 worker，批次可从 `.checkpoint.json` 继续。分类器本身只编码消息正文。[标注脚本](../scripts/label_hgp3_2200.py) · [标注提示词](../scripts/hgp-message-label-prompt.txt)
- `uncertain=true` 的记录进入人工复核。训练读取 `human_review.reviewed=true` 的标签覆盖原标注，遇到未解决的不确定记录或待标注的 MobileMem 记录会停止。历史记录中 train、val、test 分别有 8、1、1 条人工复核。[训练加载器](../scripts/train_hgp3_2200.py) · [历史标签统计](classifier-training-and-weights.md)

| 生效标签 | Train | Val | Test |
| --- | ---: | ---: | ---: |
| Semantic | 577 | 64 | 65 |
| Episodic | 705 | 82 | 72 |
| Procedural | 755 | 145 | 151 |
| Null | 163 | 52 | 46 |
| Working 为 true | 358 | 105 | 83 |
| Working 为 false | 442 | 87 | 109 |

## 2. 训练 HGP3-2200 分类器

输入为 `annotations-unified/dataset.json`，顶层包含 `train`、`val`、`test`。在 CPU 上冻结 Qwen3-Embedding-0.6B，将每条正文编码为 L2 归一化的 1,024 维向量。训练只拟合两个 sklearn `LogisticRegression` 头，共 5,125 个权重与偏置参数。

| 分类头 | 训练输入 | 推理输出 | 存档超参数 |
| --- | --- | --- | --- |
| 主类别 | 全部 2,200 条训练消息 | 4 个 logits 经温度缩放和 softmax，取概率最大的类别 | `C=10`，`class_weight=balanced` |
| Working | 800 条有布尔监督的训练消息 | 1 个 logit 经 sigmoid，概率达到 0.26 时输出 true | `C=100`，`class_weight=balanced` |

每个头比较 `C ∈ {0.1, 1, 10, 100}` 与 `class_weight ∈ {None, balanced}` 的 8 种组合，按 MobileMem 验证集的 macro-F1 选模型。主类别在同一验证集上按 log-loss 选择温度，在至少接受 20 条且接受部分标签一致率达到 90% 的条件下最大化覆盖率。存档配置的温度为 **1.214195**，离线回退阈值为 **0.824191**。Working 阈值独立按验证集 macro-F1 选出。测试集用于最终评分。[训练脚本](../scripts/train_hgp3_2200.py) · [分类器配置](../outputs/hgp3-train-2200/classifier/config.json)

训练输出 `head.pt`、`config.json`、`metrics.json` 和 `embeddings.npz`。向量缓存按数据 SHA-256 和本地模型文件信息校验。当前工作副本包含权重、配置和指标。原项目还保存了最终标注与向量缓存。原项目的 `dataset.json` 已核对与当前 checkpoint 的 `data_sha256` 一致，SHA-256 为 `563fc61028ddfae33f5467084fc94d3bc62bf71bde5bd9b4b99ff41edaebc522`。

### 存档的分类器结果

| 评测集合 | 消息数 | 主类标签一致率 | 主类 Macro-F1 | 本地接受数 | 接受部分一致率 | Working Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 Test | 334 | 82.93% | 0.7987 | 191 | 93.72% | — |
| MobileMem Val | 192 | 72.40% | 0.6456 | 64 | 90.63% | 0.8135 |
| MobileMem Test | 192 | 70.31% | 0.6514 | 61 | 80.33% | 0.8436 |

上表使用离线固定阈值，分数衡量与数据标签的一致程度。MobileMem 测试集只有 1 条有人工复核标记。验证集的接受部分一致率达到约 90%，测试集为 80.33%。端到端问答使用第 5 节的评分指标。[分类器指标](../outputs/hgp3-train-2200/classifier/metrics.json)

## 3. Construction 构建记忆

入口为 `run-mobilemem-comparison.ps1 -Method hgp -Stage construction`。启动器向 MemBase 注册 `OriginalHGPLayer`。默认输入为完整的 `data/text/mobilemem_data.json`，样本数取文件中的轨迹数，种子为 42。每条轨迹有自己的用户目录和图命名空间，标准化后的轨迹及问答写到 `MobileMem_stage_1.json`。[HGP 启动器](../scripts/run_hgp_membase.py) · [MemBase construction 入口](https://github.com/zjunlp/MemBase/blob/a9ba9495d48541d3e68588ff097294cc7a1201bc/memory_construction.py)

每条非空消息由 Qwen 分类器输出主类别及 working 状态。主类别置信度低于本轮阈值时，由 DeepSeek 复核两个输出。`null + false` 没有写入目标，`null + true` 单独写入 working，有主类别且 working 为 true 时可以同时写入两类记忆。Qwen 的 1,024 维向量用于消息路由，原版 HGP 存储和向量检索使用独立的本地 512 维 `distiluse-base-multilingual-cased-v2`。

| 类型 | 写入内容 | 存储位置 |
| --- | --- | --- |
| Semantic | 从消息提取属性、关系等事实三元组，关联来源消息 ID | Neo4j `semantic` 库 |
| Episodic | 带时间戳、参与者和来源 ID 的事件及其关联图 | Neo4j `episodic` 库及用户目录 `episodes.jsonl` |
| Procedural | 规则标题、适用条件和执行步骤 | 用户目录 `procedural_rules.jsonl` |
| Working | 按消息时间记录工作状态及转移轨迹 | 用户目录 `working/state_trajectory.json` |

`ingestion.sqlite3` 记录路由及每类写入。重跑时按消息 ID 复用已记录的路由，跳过已经完成的写入。构建完成后记录 `ready`，供 Search 确认可加载。回退结果中的 `uncertain` 会阻止该样本进入增量缓冲，桥接层仍按回退给出的类别和 working 状态写入。模板默认关闭 LLM 回退及抽取，完整实验入口启用两者。Semantic 和 Procedural 写入需要启用抽取。[HGP 原版存储桥接](../scripts/hgp_original_layer.py) · [实验配置模板](../config/templates/hgp_neo4j.json)

### 构建过程中的增量学习

模板开启 `adaptive_updates`。接受阈值从 0.9 起步，按成功更新次数 `k` 计算 `max(0.5, 0.9 × exp(-0.05 × k))`。增量样本来自低置信度消息经 DeepSeek 复核后的确定标签。代码排除 MobileMem 分类验证和测试消息的 ID、原始数据或增量缓冲中已有的规范化正文，以及与基础训练或累计样本最大余弦相似度达到 0.90 的正文。

积累 100 条合格样本后，在原始 2,200 条训练消息和该用户所有累计合格样本上重新拟合两个头。Working 仍只使用有相应标签的样本。更新沿用离线确定的 `C`、类别权重、温度和 working 阈值。权重、轮次和缓冲写入用户目录 `adaptive-router.pt`，基础 checkpoint 供所有用户初始化。关闭 `adaptive_updates` 时，实验采用固定的 `tau0`。[路由与更新代码](../scripts/hgp3_router.py)

## 4. Search 按问题检索

`-Stage search` 读取 `MobileMem_stage_1.json`，遍历 `[0, 轨迹数)`。MemBase 按用户加载已有记忆，把每题的 `question` 文本和 `TopK` 交给 HGP。参考答案随题目写入结果，供下一阶段评分。批量实验通过 `OriginalHGPLayer` 的普通构造入口加载该运行的配置。[MemBase search runner](https://github.com/zjunlp/MemBase/blob/a9ba9495d48541d3e68588ff097294cc7a1201bc/membase/runners/search.py)

| 记忆 | 检索方式 |
| --- | --- |
| Semantic | 从该用户的语义图取候选三元组，按问题的向量相似度筛选 |
| Episodic | 使用原版 HGP 混合检索，结合问题结构解析、事件图、向量相似度和 BM25 重排 |
| Procedural | 按规则 embedding 的余弦相似度检索 |
| Working | 对状态文本和时间戳建 BM25 索引，中文采用相邻字对，同分时较新的状态优先 |

HGP 将同一个 `k` 传给各路，各取至多 `k` 条，再按类别拼接。比如 `TopK=5` 时，一题可以返回多于 5 条。每题结果包含 `user_id`、`qa_pair` 和 `retrieved_memories`，写入 `<TopK>_0_<轨迹数>.json`。记忆元数据可包含类型、来源 ID、分数、事件时间等。[HGP 检索适配](../scripts/hgp_original_layer.py) · [原版 HGP 检索](../HGP-/memory/__init__.py)

**时间协议**采用先构建整段会话历史、再逐题检索的方式。MobileMem 将 `effective_timestamp` 保存在题目元数据中，当前批量检索只传入问题正文和 `k`，没有按该时间戳裁剪每题可见记忆。分类数据的 session 留出与下游的完整轨迹建库分别用于各自的评测。[MobileMem 数据适配](https://github.com/zjunlp/MemBase/blob/a9ba9495d48541d3e68588ff097294cc7a1201bc/membase/datasets/mobilemem.py) · [MemBase search runner](https://github.com/zjunlp/MemBase/blob/a9ba9495d48541d3e68588ff097294cc7a1201bc/membase/runners/search.py)

## 5. Evaluation 生成答案并评分

1. `-Stage evaluation` 读取与 Search 相同 `TopK`、相同轨迹范围的检索 JSON。`build_hgp_qa_context` 将结果按语义关系、带时间的事件、条件和步骤规则、working 历史组装成 QA 上下文。
2. MemBase 用默认问答提示词，把问题和检索上下文交给 `deepseek-flash` 生成回答。评分阶段对照 `golden_answers`。显式传 `-SkipLlmJudge` 时算 F1 和 BLEU，仍会调用 QA 模型。通用入口省略该开关时，MemBase 启用默认 F1、BLEU 和 LLM Judge，Judge 模型默认为 `deepseek-flash`。
3. 逐题预测、指标、检索内容和用户 ID 写入 `<TopK>_0_<轨迹数>_evaluation.json`，终端输出整体及按题型的平均指标。[HGP 上下文组装](../scripts/hgp_original_layer.py) · [MemBase evaluation runner](https://github.com/zjunlp/MemBase/blob/a9ba9495d48541d3e68588ff097294cc7a1201bc/membase/runners/evaluation.py) · [指标入口](https://github.com/zjunlp/MemBase/blob/a9ba9495d48541d3e68588ff097294cc7a1201bc/membase/datasets/base.py)

### 历史端到端结果

原项目存档包含两次双用户评测。已核对两份 Judge 记录覆盖相同的 1,319 个题目 ID。两次均先生成回答并算 F1、BLEU，再由独立脚本读取已保存的答案，用 `deepseek-flash` 和 `default-exact-match` 输出 Yes 或 No。表中的 F1、BLEU 是逐题分数的平均值，Judge 正确率从原报告读取。

| 原项目运行目录 | 题数 | F1 | BLEU | DeepSeek Judge |
| --- | ---: | ---: | ---: | ---: |
| `hgp-original-nojudge` | 1,319 | 0.0905 | 0.1041 | 641 / 1,319，48.60% |
| `hgp-history-time-qa` | 1,319 | 0.0941 | 0.1078 | 878 / 1,319，66.57% |

原项目的时间修复脚本按来源消息的时间校正事件和 working 状态轨迹。历史分数汇报的是各次运行的完整结果，现有记录不足以把分数差异归因于某一项改动。两次运行的 Stage 1、Top-5 检索、QA 输出以及独立 Judge 文件位于相邻的原项目目录。[原项目复跑说明](../../../PocketMemory/MOBILEMEM-RUN.md) · [时间修复脚本](../../../PocketMemory/scripts/repair_hgp_timestamps.py) · [独立 Judge 脚本](../../../PocketMemory/scripts/judge_mobilemem.py) · [原始运行报告](../../../PocketMemory/outputs/mobilemem-qwen/hgp-original-nojudge/5_0_2_evaluation_judge_deepseek.json) · [历史时间运行报告](../../../PocketMemory/outputs/mobilemem-qwen/hgp-history-time-qa/5_0_2_evaluation_judge_deepseek.json)

## 6. 复跑命令与产物

以下在当前项目根目录 `PocketMemory-v1/pocketmemory` 的 PowerShell 执行。命令使用该目录的 `Scripts/python.exe`，可将 `$py` 改为其他可用的 Python 环境。依赖参照实验环境快照。当前副本已包含 `models/` 下的 Qwen 和 512 维存储模型。端到端运行前还需恢复固定版本的 `MemBase/` 与原始 `data/text/mobilemem_data.json`，启动支持两个独立数据库的 Neo4j 实例，并在 `config/local.env` 填写 `NEO4J_USERNAME`、`NEO4J_PASSWORD` 和 `DEEPSEEK_API_KEY`。Neo4j 地址和数据库名取自实验配置。通用实验入口生成的 OpenAI 兼容配置使用 `https://api.deepseek.com/v1`。[依赖快照](../requirements-experiment.txt) · [MemBase 版本](../patches/membase/base-commit.txt) · [API 配置逻辑](../scripts/run-mobilemem-comparison.ps1)

### 重建分类数据和权重

先恢复第 1 节的三个输入文件。下列命令写入 `outputs/hgp3-train-2200-rebuilt`。`$qwen` 指向当前副本内的模型目录。

```powershell
$py = (Resolve-Path ./Scripts/python.exe).Path
$rebuilt = Join-Path $PWD 'outputs/hgp3-train-2200-rebuilt'
$annotations = Join-Path $rebuilt 'annotations-unified'
$trainData = Join-Path $annotations 'dataset.json'
$classifierDir = Join-Path $rebuilt 'classifier'
$qwen = (Resolve-Path ./models/qwen3-embedding-0.6b).Path

& $py scripts/prepare_hgp3_2200.py --out $rebuilt
& $py scripts/label_hgp3_2200.py --data $rebuilt --out $annotations
# 复核 dataset.json 中标为 uncertain 的条目，再运行训练。
& $py scripts/train_hgp3_2200.py --data $trainData --model $qwen --out $classifierDir
```

已有历史标注时，可在准备阶段传 `--annotations <历史 dataset.json 路径>`。脚本验证各划分内 MobileMem 的 ID、正文、session 和角色后，直接写出带原标注的 `dataset.json`，可以跳过重新调用 DeepSeek。重建的权重和数据应成对使用。运行原 `head.pt` 且开启增量学习时，需要恢复与 `config.json` 中 `data_sha256` 匹配的 `dataset.json` 和对应的 `embeddings.npz`。基础权重配置保存了训练机上的 Qwen 绝对路径。迁移历史产物时，在运行副本的 `classifier/config.json` 中将 `encoder` 指到本机的模型目录。

### 构建、检索和评测

下例使用上一段的 `$classifierDir` 和 `$trainData`。使用已恢复的历史产物时，将变量设为历史分类器目录和标注文件的绝对路径。每次独立实验使用自己的输出目录和图命名空间。

```powershell
$runName = 'hgp3-2200-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$runDir = Join-Path $PWD "outputs/mobilemem-qwen/$runName"
New-Item -ItemType Directory -Path $runDir | Out-Null
$cfgPath = Join-Path $runDir 'live-config.json'
$cfg = Get-Content -Raw -Encoding UTF8 ./config/templates/hgp_neo4j.json | ConvertFrom-Json
$cfg.save_dir = $runDir
$cfg.graph_namespace = $runName
$cfg.allow_llm_fallback = $true
$cfg.allow_llm_extraction = $true
$cfg | Add-Member -NotePropertyName classifier_dir -NotePropertyValue $classifierDir
$cfg | Add-Member -NotePropertyName training_data -NotePropertyValue $trainData
[IO.File]::WriteAllText($cfgPath, ($cfg | ConvertTo-Json -Depth 8))

& $py scripts/run_hgp_membase.py init-graph --config-path $cfgPath
if ($LASTEXITCODE -ne 0) { throw 'Neo4j initialization failed.' }
$run = @{
    Method = 'hgp'
    Python = $py
    ConfigPath = $cfgPath
    TopK = 5
    Workers = 2
    QaBatchSize = 8
    UsageLogPath = (Join-Path $runDir 'deepseek-qa-usage.jsonl')
}
& ./scripts/run-mobilemem-comparison.ps1 @run -Stage check
& ./scripts/run-mobilemem-comparison.ps1 @run -Stage construction
& ./scripts/run-mobilemem-comparison.ps1 @run -Stage search
& ./scripts/run-mobilemem-comparison.ps1 @run -Stage evaluation -SkipLlmJudge
```

通用入口的 `check` 是静态检查，固定检查 `outputs/hgp3-train-2200/classifier/head.pt` 是否存在。Construction 和 Search 通过配置里的 `classifier_dir` 与 `training_data` 加载本次分类器。Search 和 Evaluation 使用同一个 `TopK`。

复用已保存的答案跑历史版独立 Judge 时，先从原项目恢复 `scripts/judge_mobilemem.py`。对双用户、`TopK=5` 的运行，在第 5 节所述的 QA 文件上运行 `& $py ./scripts/judge_mobilemem.py (Join-Path $runDir '5_0_2_evaluation.json')`，得到 `5_0_2_evaluation_judge_deepseek.json`。

历史完整入口是 `./scripts/run_hgp_full.ps1 -Stage all`，脚本要求项目根目录下有 `pocketmemory/Scripts/python.exe`。当前工作副本的解释器位于 `Scripts/python.exe`，可用上面的逐阶段命令。此入口使用默认分类器与数据路径、`TopK=5`、`Workers=2`、`QaBatchSize=8`，输出到 `outputs/mobilemem-qwen/hgp-original-nojudge/`，并将 Evaluation 限定为 F1 和 BLEU。它也接受 `construction`、`search`、`evaluation` 和 `search-and-evaluation`，后几项依赖前序产物。[逐阶段入口](../scripts/run-mobilemem-comparison.ps1) · [完整实验入口](../scripts/run_hgp_full.ps1)

| 产物 | 后续用途 | 当前工作副本 |
| --- | --- | --- |
| `annotations-unified/dataset.json` | 训练及增量学习回放 | 需恢复或重建 |
| `classifier/head.pt`、`config.json`、`metrics.json` | 基础权重、配置和离线指标 | 已保存 |
| `classifier/embeddings.npz` | 向量缓存及增量学习初始化 | 需恢复或用匹配输入重建 |
| `MobileMem_stage_1.json`、用户记忆目录及 Neo4j 图 | Search 的标准化数据与记忆 | 由 Construction 生成 |
| `<TopK>_0_<轨迹数>.json` | Evaluation 的逐题检索输入 | 由 Search 生成 |
| `<TopK>_0_<轨迹数>_evaluation.json` | 逐题答案和问答指标 | 由 Evaluation 生成 |
