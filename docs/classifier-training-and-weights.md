# HGP3-2200 分类器训练

HGP3-2200 用冻结的 Qwen3-Embedding-0.6B 编码单条消息，再训练两个独立的逻辑回归头。它负责决定消息写入哪类记忆，以及是否同时记录 Working State。iPhone Companion 在电脑端使用固定权重和验证集校准过的阈值。 [Companion Quick Start](../README.md#quick-start) · [和原版 HGP 的区别](hgp3-2200-workflow.md#相对原版-hgp-的实现)

## 1. 数据和标签

| 来源 | 训练 | 验证 | 测试 | 怎么标 |
| --- | ---: | ---: | ---: | --- |
| PerLTQA 中文 | 1,000 | 97 | 84 | 人物描述与关系映射为语义记忆，事件映射为情节记忆。 |
| proScript 中文改写 | 400 | 54 | 58 | 把场景流程改写成中文方法回答，复核步骤、顺序和原始事实后标为程序记忆。 |
| MobileMem | 800 | 192 | 192 | DeepSeek 标注单条消息的主类别和 `working_state`。已确认的人工复核覆盖模型标注。 |
| 合计 | **2,200** | **343** | **334** | 共 **2,877** 条。 |

主类别在 `semantic`、`episodic`、`procedural` 和 `null` 中选一个。`working_state` 独立判断当前待办、进展或阻碍，可以和任一主类别同时成立。主头使用全部训练消息，Working 头只使用 800 条有布尔标签的 MobileMem 训练消息。缺少 Working 标注的 PerLTQA 和 proScript 消息跳过 Working 损失。

划分使用种子 42，PerLTQA 按人物、proScript 按场景、MobileMem 按会话分组。训练前检查跨集合的分组、ID 和规范化文本。MobileMem 评测覆盖留出的会话，全新用户和语义近重复仍需要单独验证。DeepSeek 标注输入含角色和消息正文，本地分类器只编码正文。 [数据准备](../scripts/prepare_hgp3_2200.py) · [标注脚本](../scripts/label_hgp3_2200.py) · [标签说明](../scripts/hgp-message-label-prompt.txt)

## 2. 实际怎么训练

1. 冻结 Qwen 编码器，在 CPU 上生成归一化的 1,024 维向量。训练前检查最长输入，超限时报错。
2. 拟合两个 `LogisticRegression` 头。主头用四类 softmax，Working 头用 sigmoid。总计训练 **5,125 个权重和偏置**。两个头各尝试 `C ∈ {0.1, 1, 10, 100}` 和 `class_weight ∈ {None, balanced}`，以 MobileMem 验证集 Macro-F1 选配置。
3. 用 MobileMem 验证集 log-loss 选主头温度，校准主类别最大概率。选择至少接受 20 条、接受部分标签一致率达到 90% 且覆盖率最高的回退阈值。Working 阈值按验证集 Macro-F1 选择。
4. 固定权重和阈值，在留出测试集评分，保存 `head.pt`、`config.json`、`metrics.json` 和 `embeddings.npz`。 [训练入口](../scripts/train_hgp3_2200.py)

发布权重的主头和 Working 头都用 `balanced`，`C` 分别为 10 和 100。主头温度为 **1.2142**，iPhone 采用的主头回退阈值为 **0.8242**，Working 阈值为 **0.26**。iPhone 低于主头阈值时用 DeepSeek 复判两个输出，采用 `adaptive=False`。 [iPhone 分类入口](../iphone-companion/classifier.py)

| 留出集合 | 样本数 | 主头标签一致率 | 主头 Macro-F1 | 本地接受数 | 接受部分一致率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部测试 | 334 | 82.93% | 0.7987 | 191 | 93.72% |
| MobileMem 验证 | 192 | 72.40% | 0.6456 | 64 | 90.63% |
| MobileMem 测试 | 192 | 70.31% | 0.6514 | 61 | 80.33% |

Working 头在 MobileMem 测试集上的标签一致率为 **84.38%**，Macro-F1 为 **0.8436**。这些指标衡量对现有标签的一致程度。MobileMem 测试集中只有 1 条人工复核记录，使用 DeepSeek 复判后的端到端分类表现需要另测。

## 3. 用存档标签重新训练

从项目根目录运行。先按 Quick Start 创建 `.venv` 并解压预置的 `outputs/`，确认有 `outputs/hgp3-train-2200/annotations-unified/dataset.json`。下面命令将新权重写到独立目录。用于实验增量回放的 `head.pt`、`config.json`、`embeddings.npz` 和 `dataset.json` 应成套使用。 [下载预置数据](hgp3-2200-workflow.md#下载实验产物)

### Windows PowerShell

```powershell
$trainingPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $trainingPython -m pip install "sentence-transformers>=5.2.3,<6" scikit-learn
& $trainingPython -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/qwen3-embedding-0.6b')"

$dataset = "outputs/hgp3-train-2200/annotations-unified/dataset.json"
$trained = "outputs/hgp3-2200-retrain-$(Get-Date -Format yyyyMMdd-HHmmss)/classifier"
& $trainingPython scripts/train_hgp3_2200.py --data $dataset --model models/qwen3-embedding-0.6b --out $trained
```

### macOS 终端

以下命令面向 Apple Silicon。Windows 和 macOS 都在本机 CPU 上生成向量。

```bash
training_python="$PWD/.venv/bin/python"
"$training_python" -m pip install 'sentence-transformers>=5.2.3,<6' scikit-learn
"$training_python" -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-Embedding-0.6B", local_dir="models/qwen3-embedding-0.6b")'

dataset="outputs/hgp3-train-2200/annotations-unified/dataset.json"
trained="outputs/hgp3-2200-retrain-$(date +%Y%m%d-%H%M%S)/classifier"
"$training_python" scripts/train_hgp3_2200.py --data "$dataset" --model models/qwen3-embedding-0.6b --out "$trained"
```

## 4. 从原始数据重建标签

从数据 ZIP 解压 PerLTQA 的 `data/external/PerLTQA/Dataset/zh/perltmem.json` 和 MobileMem 的 `data/text/mobilemem_data.json`。传入历史 `dataset.json` 可复用 MobileMem 标注，缺少 `synthetic.jsonl` 时还会复用历史 proScript 样本和场景分组。 [下载源数据](hgp3-2200-workflow.md#下载源数据) · [复用历史标注的命令](hgp3-2200-workflow.md#重建分类数据和权重)

要重新标注 MobileMem，可从训练备份恢复 `outputs/proscript-dialogue-600/synthetic.jsonl`，再运行下面的命令。如果备份只保留了候选文件，可以用 `scripts/synthesize_proscript.py --data <candidates.jsonl>` 重新改写，生成数据可能随模型响应变化。改写和标注读取 Quick Start 中的 DeepSeek 配置。

**Windows PowerShell**

```powershell
$build = "outputs/hgp3-2200-data-$(Get-Date -Format yyyyMMdd-HHmmss)"
& $trainingPython scripts/prepare_hgp3_2200.py --out $build
& $trainingPython scripts/label_hgp3_2200.py --data $build --out "$build/annotations-unified"
```

**macOS 终端**

```bash
build="outputs/hgp3-2200-data-$(date +%Y%m%d-%H%M%S)"
"$training_python" scripts/prepare_hgp3_2200.py --out "$build"
"$training_python" scripts/label_hgp3_2200.py --data "$build" --out "$build/annotations-unified"
```

人工复核 `dataset.json` 中的 `uncertain=true` 记录，填写 `human_review.reviewed=true` 及确认后的 `label`、`working_state`。用第 3 节的训练命令，把 `--data` 改成新目录下的 `annotations-unified/dataset.json`。
