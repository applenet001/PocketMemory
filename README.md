# PocketMemory · MobileMem Tracing

基于 [MemBase](https://github.com/zjunlp/MemBase) 的 MobileMem 实验适配，运行 **NaiveRAG、Long Context、Mem0** 的构建、检索与评测，保留原版生命周期 tracing。

## 修改内容

下面的代码路径均相对于 `MemBase/`。

| 文件 | 修改 |
|---|---|
| `examples/trace_memory_lifecycle_with_membase/run_traced_{construction,search,evaluation}.sh` | 增加 MobileMem 分支；加载共享配置；显式选择 Python；统一日志目录；支持等待阶段完成。Search 显式传入轨迹范围。 |
| `examples/trace_memory_lifecycle_with_membase/mobilemem_profile.sh`（新增） | 集中管理数据路径、模型、batch、top-k、API 配置和用量日志；自动选择 Windows/macOS 虚拟环境路径。 |
| `membase/configs/{naive_rag,long_context}.py` | 移除 GPT tokenizer 默认值，要求显式填写 `llm_model`；该字段仅用于 token 计数。 |
| `membase/utils/token_monitor.py` | 支持读取本地 `tokenizer.json`。 |
| `membase/inference_utils/{backends,base_operator}.py` | 记录 DeepSeek API 返回的 token usage、调用阶段及缺失用量；支持流式 usage。 |
| `membase/baselines/mem0/llms/openai.py` | 设置 DeepSeek `reasoning_effort=low`，记录记忆构建调用的 usage。 |

仓库新增两个入口：

- `scripts/run-traced-smoke20.sh`：串联三种方法的三个阶段。
- `scripts/serve_qwen_embedding.py`：本地 CPU Qwen embedding 服务，支持 `QWEN_MODEL_PATH`。

**未修改**检索算法、MobileMem 时间协议、F1/BLEU 算法。`--tracing` 和 construction 的 `--rerun` 为原版已有参数。

## 参数差异

| 参数 | 原版 traced 示例 | 当前配置 |
|---|---|---|
| 数据集 | LoCoMo | MobileMem smoke20 |
| QA / Judge | GPT-4.1 mini / Claude Opus 4.5 | DeepSeek Flash / DeepSeek Flash |
| QA / Judge batch | 4 / 4 | 1 / 1 |
| Workers / Seed | 1 / 42 | 1 / 42 |
| Top-k | 10 | RAG、Mem0：30；Long Context：1 |
| 轨迹数 | 1 | 从输入自动读取；smoke20 为 2 条、20 题 |

Embedding 使用 **Qwen3-Embedding-0.6B（1024 维）**。RAG 分块预算为 1200 tokens，Long Context 窗口为 200000 tokens。配置模板位于 `config/templates/`。

## 安装与运行

MemBase 改动以固定上游版本和补丁保存。在仓库根目录使用 Bash（Windows 使用 Git Bash，macOS 使用终端）：

```bash
git clone https://github.com/zjunlp/MemBase.git MemBase
git -C MemBase checkout "$(cat patches/membase/base-commit.txt | tr -d '\r')"
git -C MemBase apply ../patches/membase/traced-mobilemem.patch
cp patches/membase/mobilemem_profile.sh MemBase/examples/trace_memory_lifecycle_with_membase/
```

创建统一环境：

```bash
# Windows Git Bash 使用 python；macOS 使用 python3
python -m venv pocketmemory          # Windows
# python3 -m venv pocketmemory       # macOS

# Windows
source pocketmemory/Scripts/activate
# macOS
# source pocketmemory/bin/activate

python -m pip install -r requirements-experiment.txt
```

启动脚本兼容两种环境路径；依赖快照来自 Windows Python 3.13，macOS 尚未实机验证，平台 wheel 可用性需在安装时确认。

运行前准备：

1. 数据放入 `data/text/smoke20/mobilemem_data.json`。数据不随仓库分发；20 题抽样应固定后用于所有方法。
2. 将 `config/templates/*.json` 复制到 `config/mobilemem-qwen/`，将 RAG/Long Context 的 tokenizer 占位符改为本机 Qwen `tokenizer.json` 绝对路径。
3. 设置 `QWEN_MODEL_PATH` 为完整 Qwen 模型目录，包含权重、模型配置和 tokenizer。
4. 本地创建 `outputs/private/deepseek-api.json`：

```json
{"api_keys":["YOUR_DEEPSEEK_KEY"],"base_urls":["https://api.deepseek.com/v1"]}
```

在已激活环境的 Bash 终端启动服务：

```bash
export QWEN_MODEL_PATH="/path/to/Qwen3-Embedding-0.6B"
python scripts/serve_qwen_embedding.py
```

服务 ready 后，在另一个 Bash 终端、仓库根目录运行：

```bash
bash scripts/run-traced-smoke20.sh
```

只跳过已完成的 RAG 构建、检索，并继续后续实验：

```bash
RESUME_RAG_EVALUATION=1 bash scripts/run-traced-smoke20.sh
```

恢复入口会重新生成 RAG 回答和评分，并从头运行后续方法；不是通用断点续跑，会产生额外调用费用。

## 结果与解释

输出位于 `outputs/mobilemem-qwen/traced-smoke20/<method>/`：`logs/` 保存阶段日志，`traced_data/` 保存执行图，`deepseek_usage.jsonl` 保存 API 用量。用量日志不是账单，没有自动预算停止机制。Embedding 用量当前另写入 `outputs/mobilemem-qwen/smoke20/embedding_usage.jsonl`。

- **时间协议**：完整历史构建后问答，不按每题 `effective_timestamp` 隔离历史。
- **指标**：默认 F1、BLEU、LLM judge；原版 F1/BLEU 没有中文专用分词。额外统一延迟、session 快照等指标尚未完整接入。
- **持久化**：NaiveRAG 保存文本，检索阶段重算向量；Mem0 使用持久化 Qdrant；Long Context 不使用向量索引。

继续修改 MemBase 后，更新补丁并同步新增 profile 文件：

```bash
git -C MemBase diff --binary --output=../patches/membase/traced-mobilemem.patch
cp MemBase/examples/trace_memory_lifecycle_with_membase/mobilemem_profile.sh patches/membase/
```