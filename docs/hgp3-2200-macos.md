# HGP3-2200 macOS 运行指南

在 Apple Silicon Mac 的「终端」中，从 `PocketMemory-v1` 项目根目录依次运行所需命令。只使用 iPhone 收件箱时，按安装教程准备本地模型和现成分类头即可。 [macOS 安装教程](../README.md#macos-终端) · [使用现成分类器](hgp3-2200-workflow.md#使用现成分类器)

## 1. 准备 Python 和 Qwen

安装 Python 3.13 后，在本机创建 `.venv`。已有按安装教程创建的环境时，从 `py=…` 那一行继续。分类器使用 CPU 上的 Qwen3-Embedding-0.6B。 [PyTorch macOS 安装说明](https://pytorch.org/get-started/locally/#macos)

```bash
python3.13 -m venv .venv
py="$PWD/.venv/bin/python"
"$py" -m pip install --upgrade pip
"$py" -m pip install 'sentence-transformers>=5.2.3,<6' scikit-learn
"$py" -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-Embedding-0.6B", local_dir="models/qwen3-embedding-0.6b")'
```

## 2. 恢复或重建分类器

使用现成权重时，从归档恢复 `outputs/`。HGP3-2200 的实验权重、配套标注和向量缓存应保留原有目录结构。在恢复的 `outputs/hgp3-train-2200/classifier/config.json` 中把 `encoder` 改成本机 `models/qwen3-embedding-0.6b` 的绝对路径，再跳到第 3 节。 [下载实验产物和文件路径](hgp3-2200-workflow.md#下载实验产物)

重建历史划分时，从数据 ZIP 解压 PerLTQA 和 MobileMem，并从实验产物 ZIP 恢复历史 `dataset.json`。缺少 `synthetic.jsonl` 时，准备脚本用历史 `dataset.json` 中的 proScript 样本和场景分组。 [下载源数据](hgp3-2200-workflow.md#下载源数据) · [数据配方和标签说明](hgp3-2200-workflow.md#1-构建分类数据)

- `data/external/PerLTQA/Dataset/zh/perltmem.json`
- `data/text/mobilemem_data.json`
- `outputs/hgp3-train-2200/annotations-unified/dataset.json`

要从完整候选池重新选样并标注 MobileMem，从训练备份恢复 `outputs/proscript-dialogue-600/synthetic.jsonl`。若只有存档的 proScript 候选文件，在 `config/local.env` 中填写 `DEEPSEEK_API_KEY`，然后指定候选文件路径重新改写。重新调用 DeepSeek 可能改变复核通过数量和数据哈希。

```bash
"$py" scripts/synthesize_proscript.py --data '/路径/candidates.jsonl'
```

准备和标注数据。标注从 `config/local.env` 读取 `DEEPSEEK_API_KEY`。示例将重建结果写到独立目录。

```bash
build_dir="$PWD/outputs/hgp3-train-2200-rebuilt"
annotations="$build_dir/annotations-unified"
training_data="$annotations/dataset.json"
classifier_dir="$build_dir/classifier"
qwen="$PWD/models/qwen3-embedding-0.6b"

"$py" scripts/prepare_hgp3_2200.py --out "$build_dir"
"$py" scripts/label_hgp3_2200.py --data "$build_dir" --out "$annotations"
```

复用已恢复的历史 `dataset.json` 时，将准备和标注两条命令替换为下面一条，随后可直接训练。

```bash
"$py" scripts/prepare_hgp3_2200.py --out "$build_dir" --annotations 'outputs/hgp3-train-2200/annotations-unified/dataset.json'
```

重新标注后，在 `dataset.json` 中人工复核 `uncertain=true` 的条目，记录 `human_review.reviewed=true` 及复核后的 `label` 和 `working_state`，再训练。训练目录已有权重和 `metrics.json` 时，可加 `--evaluate-only` 刷新离线指标。

```bash
"$py" scripts/train_hgp3_2200.py --data "$training_data" --model "$qwen" --out "$classifier_dir"
```

## 3. 准备 MobileMem 实验环境

在同一个虚拟环境安装 HGP 实验依赖并下载 512 维存储模型。 [MemBase 环境说明](https://github.com/zjunlp/MemBase/blob/a9ba9495d48541d3e68588ff097294cc7a1201bc/README.md#installation) · [SmartComment 安装说明](https://github.com/zjunlp/smartcomment#installation)

```bash
"$py" -m pip install neo4j rank-bm25 python-dotenv openai litellm smartcomment tiktoken nltk
"$py" -c 'from huggingface_hub import snapshot_download; snapshot_download("sentence-transformers/distiluse-base-multilingual-cased-v2", local_dir="models/hgp-distiluse-base-multilingual-cased-v2")'
```

源码包内已有 `MemBase/membase/`。首次运行时，从项目锁定的 MemBase 提交补齐缺少的三个命令入口。 [固定版本](../patches/membase/base-commit.txt)

```bash
(
  cd MemBase || exit 1
  ref=$(cat ../patches/membase/base-commit.txt)
  for entry in memory_construction.py memory_search.py memory_evaluation.py; do
    if [ ! -f "$entry" ]; then
      curl -fsSL "https://raw.githubusercontent.com/zjunlp/MemBase/$ref/$entry" -o "$entry" || exit 1
    fi
  done
)
```

启动支持多数据库的 Neo4j，使用有建库权限的账号。默认连接地址取自 `config/templates/hgp_neo4j.json`，为 `neo4j://127.0.0.1:7687`。在 `config/local.env` 中填入以下值，下一节的 `init-graph` 会创建或检查 `semantic` 与 `episodic` 数据库。 [Neo4j 多数据库说明](https://neo4j.com/docs/operations-manual/current/database-administration/standard-databases/create-databases/)

```dotenv
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=填入你的Neo4j密码
DEEPSEEK_API_KEY=填入你的DeepSeek密钥
```

## 4. 构建、检索和评测

本节使用默认的分类器和标注位置。如果第 2 节训练到了独立目录，把下面的 `classifier_dir` 和 `training_data` 改成该次训练的绝对路径。每次独立实验创建自己的输出目录和图命名空间。

```bash
dataset="$PWD/data/text/mobilemem_data.json"
classifier_dir="$PWD/outputs/hgp3-train-2200/classifier"
training_data="$PWD/outputs/hgp3-train-2200/annotations-unified/dataset.json"
run_name="hgp3-2200-$(date +%Y%m%d-%H%M%S)"
run_dir="$PWD/outputs/mobilemem-qwen/$run_name"
config="$run_dir/live-config.json"
top_k=5

"$py" - "$config" "$run_name" "$classifier_dir" "$training_data" <<'PY'
import json
import sys
from pathlib import Path

config_path, run_name, classifier_dir, training_data = sys.argv[1:]
config_path = Path(config_path)
config = json.loads(Path("config/templates/hgp_neo4j.json").read_text(encoding="utf-8"))
config.update(
    save_dir=str(config_path.parent), graph_namespace=run_name,
    classifier_dir=classifier_dir, training_data=training_data,
    allow_llm_fallback=True, allow_llm_extraction=True,
)
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY
```

**构建记忆。** 在完整 MobileMem 数据上运行，写出标准化数据 `MobileMem_stage_1.json` 和用户记忆。

```bash
count=$("$py" -c 'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$dataset")
"$py" scripts/run_hgp_membase.py init-graph --config-path "$config" &&
  "$py" scripts/run_hgp_membase.py construction \
    --dataset-type MobileMem --dataset-path "$dataset" --config-path "$config" \
    --num-workers 2 --seed 42 --sample-size "$count" \
    --token-cost-save-filename "$run_dir/token_cost_hgp"
```

**按问题检索。** 使用上一阶段的标准化数据，输出 `<TopK>_0_<轨迹数>.json`。

```bash
stage1="$run_dir/MobileMem_stage_1.json"
count=$("$py" -c 'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["trajectories"]))' "$stage1")
"$py" scripts/run_hgp_membase.py search \
  --dataset-type MobileMem --dataset-path "$stage1" --dataset-standardized \
  --config-path "$config" --num-workers 2 --top-k "$top_k" --start-idx 0 --end-idx "$count"
```

**生成答案并评分。** 使用 DeepSeek 生成答案，计算 F1、BLEU 和 LLM Judge。结果写在检索文件旁，文件名为 `<TopK>_0_<轨迹数>_evaluation.json`。

```bash
api_key="${DEEPSEEK_API_KEY:-$("$py" -c 'from dotenv import dotenv_values; print(dotenv_values("config/local.env").get("DEEPSEEK_API_KEY") or "", end="")')}"
OPENAI_API_KEY="$api_key" OPENAI_API_BASE=https://api.deepseek.com/v1 \
  MEMORY_METHOD=hgp DEEPSEEK_USAGE_LOG="$run_dir/deepseek-qa-usage.jsonl" \
  "$py" scripts/run_hgp_membase.py evaluation \
    --dataset-type MobileMem --search-results-path "$run_dir/${top_k}_0_${count}.json" \
    --qa-model deepseek-flash --qa-batch-size 8 \
    --judge-model deepseek-flash --judge-batch-size 1 --metrics f1 bleu llm_judge
```

评测协议和历史结果见 [Evaluation 生成答案并评分](hgp3-2200-workflow.md#5-evaluation-生成答案并评分)。
