# PocketMemory：MobileMem traced smoke20

本仓库保存当前正在使用的实验适配：基于 MemBase，运行 NaiveRAG、Long Context 和 Mem0 的构建、检索、回答与评分，并导出执行图。当前实验已暂停；仓库不包含运行结果，不代表全部方法已完成。

## 上传范围

外层仓库采用白名单，只上传以下文件：

| 路径 | 用途 |
|---|---|
| `README.md`、`LICENSE`、`.gitignore`、`.gitattributes` | 说明、许可证及上传规则 |
| `scripts/run-traced-smoke20.sh` | 依次运行三种方法的三个阶段；支持从 RAG evaluation 恢复 |
| `scripts/serve_qwen_embedding.py` | CPU Qwen embedding HTTP 服务，端口 8008 |
| `config/templates/*_traced_smoke20.json` | 三种方法的无密钥配置模板 |
| `requirements-experiment.txt` | 当前统一环境的依赖快照 |
| `patches/membase/base-commit.txt` | MemBase 上游基准提交 |
| `patches/membase/traced-mobilemem.patch` | 对上游已跟踪文件的修改 |
| `patches/membase/mobilemem_profile.sh` | 新增的共享实验配置；补丁本身不包含未跟踪文件 |


## 相对 MemBase 原版改了哪里

以下路径均相对于 `MemBase/`：

| 文件 | 修改 |
|---|---|
| `examples/trace_memory_lifecycle_with_membase/run_traced_construction.sh` | 默认 MobileMem/RAG；加载共享配置；指定 Python；日志进入 save_dir；可等待后台进程完成 |
| `examples/trace_memory_lifecycle_with_membase/run_traced_search.sh` | 同上；MobileMem 显式传 start/end，使结果文件名与 evaluation 一致 |
| `examples/trace_memory_lifecycle_with_membase/run_traced_evaluation.sh` | 同上；共享配置覆盖 QA/judge 模型及 batch |
| `examples/trace_memory_lifecycle_with_membase/mobilemem_profile.sh` | 新文件：数据路径、统一 Python、模型、范围、top-k、API 配置和 usage 日志 |
| `membase/configs/naive_rag.py` | 去掉 GPT tokenizer 默认值；llm_model 必须显式填非空字符串 |
| `membase/configs/long_context.py` | 同上；该字段只选择计数 tokenizer，不代表推理模型 |
| `membase/utils/token_monitor.py` | 支持直接加载本地 tokenizer.json |
| `membase/inference_utils/base_operator.py` | 给 DeepSeek 用量标注调用阶段 |
| `membase/inference_utils/backends.py` | 记录 DeepSeek API usage、重试及缺失 usage；流式请求启用 usage 返回 |
| `membase/baselines/mem0/llms/openai.py` | DeepSeek reasoning_effort=low；记录 Mem0 构建调用的 usage |

未修改 MobileMem loader 的时间协议、F1/BLEU 算法及检索算法。`--tracing`、construction 的 `--rerun` 原版已有。

## 当前参数

- QA 和 judge：deepseek-flash；两种调用均计费。
- QA/judge batch：1（原版 4）；workers：1；seed：42。
- RAG/Mem0 top-k：30（原版 10）；Long Context：1。
- 轨迹数自动取输入数组长度；本地 smoke20 为 2 个用户、1600 条消息、20 个问题。
- Qwen3-Embedding-0.6B，1024 维，本地 CPU；RAG 分块预算 1200 tokens；Long Context 窗口 200000 tokens。
- Qwen tokenizer 用于本地计数；DeepSeek 费用使用 API 返回的 usage。

## 复现准备（Windows + Git Bash）

1. 在仓库根目录下载固定版本的 MemBase。新克隆时才应用补丁；本机现有 MemBase 已修改，不要重复应用。

```bash
git clone https://github.com/zjunlp/MemBase.git MemBase
git -C MemBase checkout "$(cat patches/membase/base-commit.txt | tr -d '\r')"
git -C MemBase apply --check ../patches/membase/traced-mobilemem.patch
git -C MemBase apply ../patches/membase/traced-mobilemem.patch
cp patches/membase/mobilemem_profile.sh MemBase/examples/trace_memory_lifecycle_with_membase/
```

2. 使用 Python 3.13 建立唯一实验环境。在 PowerShell 中执行：

```powershell
py -3.13 -m venv pocketmemory
.\pocketmemory\Scripts\python.exe -m pip install -r requirements-experiment.txt
.\pocketmemory\Scripts\python.exe -m pip check
```

依赖文件是当前 Windows 环境快照，不承诺其他系统兼容。脚本直接指定 Python 路径，不依赖 activate。

3. 自行准备有权限使用的 MobileMem 数据，放在 `data/text/smoke20/mobilemem_data.json`。这里不随仓库分发数据，也未提供可复现的 smoke20 抽样脚本；换一份 20 题样本不保证与本地结果相同。

4. 将配置模板复制到 `config/mobilemem-qwen/`，文件名保持不变。将 RAG 和 Long Context 的 `llm_model` 占位符替换为本机 Qwen `tokenizer.json` 绝对路径，确保它和 embedding 服务使用同一模型版本。

5. 下载 Qwen3-Embedding-0.6B 完整模型。本地服务默认从 `~/.cache/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master` 加载；其他位置需修改 `serve_qwen_embedding.py` 的模型路径。只有 safetensors 文件不够，还需要 tokenizer 和模型配置。

6. 本地创建 `outputs/private/deepseek-api.json`（禁止提交）：

```json
{"api_keys":["YOUR_DEEPSEEK_KEY"],"base_urls":["https://api.deepseek.com/v1"]}
```

当前 traced 脚本直接读取该 JSON，不自动读取 `config/local.env`。

## 运行

先在 PowerShell 的仓库根目录启动 embedding 服务：

```powershell
.\pocketmemory\Scripts\python.exe scripts/serve_qwen_embedding.py
```

等日志出现 ready，再在另一个 Git Bash 窗口、仓库根目录执行：

```bash
bash scripts/run-traced-smoke20.sh
```

只从 RAG evaluation 恢复并继续后续方法：

```bash
RESUME_RAG_EVALUATION=1 bash scripts/run-traced-smoke20.sh
```

这不是通用断点续跑：恢复模式会跳过 RAG construction/search，但会重跑 RAG 回答和评分，并继续从头运行 Long Context、Mem0。重复运行会额外消耗 API token；启动前确认没有另一份任务。保留旧结果时先复制输出目录或调整 save_dir。

三个官方阶段脚本仍可单独运行；共享 `METHOD`、`DATASET` 和 `WAIT_FOR_COMPLETION=1`，先等待前一阶段成功。

## 输出与限制

- 各方法结果：`outputs/mobilemem-qwen/traced-smoke20/{rag,long_context,mem0}/`。
- `logs/traced_*.log`：阶段日志；`traced_data/<user>/graph_*.json`：执行图，不是知识图谱。
- `deepseek_usage.jsonl`：调用返回的原始 token usage；包括保留日志中的失败后重跑调用。不是人民币账单，没有 80 元自动停止机制，SDK 内部失败重试不保证都有独立 usage。
- embedding 服务目前将日志写入旧路径 `outputs/mobilemem-qwen/smoke20/embedding_usage.jsonl`；它是本地计算，不是 DeepSeek API 费用。
- NaiveRAG 保存记忆文本而不持久化向量，search 会重算已有记忆的 embedding；Mem0 使用持久化 Qdrant；Long Context 不建向量索引。
- 当前先构建完整历史再问答；MobileMem loader 把问题时间设为输入最后一个 session 结束后 7 天，effective_timestamp 只存 metadata，没有逐题历史隔离。
- 默认 F1/BLEU 没有中文专用分词；不要将低文本匹配分数直接等同于语义错误。
- 额外的统一延迟、session 快照等 metrics 尚未全部接入；开启 tracing 不代表这些指标都已实现。

## 提交前检查

```bash
git status --short
git add README.md LICENSE .gitignore .gitattributes requirements-experiment.txt scripts/run-traced-smoke20.sh scripts/serve_qwen_embedding.py config/templates patches/membase
git diff --cached --stat
git diff --cached --check
```

检查暂存区不含密钥、数据、模型、输出或虚拟环境后再自行 commit/push。本文档整理工作没有执行 add、commit 或 push。

以后修改 MemBase 后，重新导出 `git -C MemBase diff --binary --output=../patches/membase/traced-mobilemem.patch`，并同步新增的 profile 文件，否则外层仓库不会包含最新修改。