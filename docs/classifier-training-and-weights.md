# PocketMemory Classifier：训练数据、分类设计与权重更新

核对日期：2026-11-24。本文以当前 iPhone 入口使用的 `outputs/hgp3-train-2200/classifier`、对应数据集和实际训练代码为准。

当前方案是 **冻结 Qwen3-Embedding-0.6B 编码器，共享 1024 维向量，分别训练主记忆分类头和 working 状态头**。主分类输出 semantic、episodic、procedural 或 null；working 独立判断。实验路由器支持积累 LLM 回退样本后重拟合两个分类头；目前 iPhone 入口关闭了这一更新功能。

## 1. 当前版本与旧版本

| 版本 | 实现 | 与当前部署的关系 |
|---|---|---|
| 早期四标签分类 | `scripts/train_hgp_classifier.py`：四个 sigmoid 输出，AdamW + BCE | 旧方案，不是当前 checkpoint 的训练方式 |
| 3,000 条三分类方案 | `scripts/train_hgp3.py`：三类 softmax，AdamW；见 `docs/hgp3-training.md` | 旧方案，数据配方和训练参数不能套用到当前版本 |
| 当前 HGP3-2200 | `scripts/train_hgp3_2200.py`：冻结 embedding + 两个 LogisticRegression 头 | 当前 iPhone 入口加载的模型 |

“2200”指训练集数量；加上验证与测试，当前数据总量是 **2,877 条**。虽然目录叫 HGP3，主分类头实际有四个输出位置，第四个表示“没有合适的主记忆类别”，不是 working。

## 2. 训练数据与标签来源

### 2.1 数据组成

| 数据来源 | Train | Val | Test | 标签来源与用途 |
|---|---:|---:|---:|---|
| PerLTQA 中文 | 1,000 | 97 | 84 | 人物描述、关系映射 semantic；事件映射 episodic |
| proScript 中文改写 | 400 | 54 | 58 | 场景流程改写成中文方法回答，映射 procedural；经过自动复核 |
| MobileMem | 800 | 192 | 192 | 单条真实会话消息；DeepSeek 标注主类别与 working，部分人工复核 |
| 合计 | **2,200** | **343** | **334** | |

PerLTQA 的 semantic 数量分别为 500、50、50，其余为 episodic。proScript 采用通过自动检查且无已标记源问题的样本；自动复核不等于人工 gold。

MobileMem 使用消息正文，不使用 QA 答案作为分类输入。当前标注请求传递 `role` 和目标消息 `target`；本地分类器输入仅为 `text`，不把角色或多轮上下文显式编码进去。这会造成 teacher 和本地分类器可见信息的差异。

### 2.2 最终生效的标签分布

统计遵循训练加载器：若 `human_review.reviewed=true`，采用人工复核中的 `label` 和 `working_state` 覆盖原标注。

| 主标签 | Train | Val | Test |
|---|---:|---:|---:|
| semantic | 577 | 64 | 65 |
| episodic | 705 | 82 | 72 |
| procedural | 755 | 145 | 151 |
| null / no_primary_memory | 163 | 52 | 46 |
| 合计 | 2,200 | 343 | 334 |

| Working 标签 | Train | Val | Test |
|---|---:|---:|---:|
| true | 358 | 105 | 83 |
| false | 442 | 87 | 109 |
| 未标注，不参与 working 训练/评估 | 1,400 | 151 | 142 |

当前 working 监督全部来自 MobileMem。外部数据的 `working_state=null` 表示缺少监督，不能当成 false。

当前数据中标记已人工复核的样本数为 Train 8、Val 1、Test 1。不能将旧方案中的“20% 人工审核计划”当作当前数据的实际审核比例。

### 2.3 划分和质量控制

- 准备脚本使用种子 42 和基于 ID 的确定性哈希排序。
- PerLTQA 按人物分组；proScript 按场景分组，并保留上游 train/val/test 归属；MobileMem 按 session 分组。
- 检查 ID、分组和规范化后的重复文本，禁止跨集合泄漏。离线划分没有实施全量语义近重复检测。
- MobileMem 按会话隔离，不按用户隔离，因此这些指标不能证明对全新用户的泛化能力。
- MobileMem 待标注样本和未解决的不确定标注会阻止训练；人工复核记录优先。
- 编码前检查 token 长度，超过编码器限制时报错，不静默截断。

## 3. 分类任务的定义

| 输出 | 判断依据 | 示例（仅说明标签含义） |
|---|---|---|
| semantic | 稳定属性、偏好、关系、一般知识 | “我喜欢清淡饮食。” |
| episodic | 有具体情境的事件、经历或明确安排 | “昨天我给小李打了电话。” |
| procedural | 已给出可复用的方法、策略、规则或步骤 | “如果文件过大，就先压缩再发送。” |
| null | 没有合适的主标签 | “好的，谢谢。” |
| working_state=true | 当前任务、待办、进度、阻碍、短期约束 | “还没通知组员，今天下班前得发邮件。” |

主标签最多选一个，working 可以与主标签同时成立。`null + true` 可以表达只有当前待办的消息；`null + false` 可以表达无需进入这些记忆类别的消息。

`null` 与 `uncertain` 含义不同：前者是模型实际学习的类别，后者是对结果是否可靠的标记。明确无主标签可以是确定的；任意主标签也可能置信度不足。

## 4. 模型结构与推理

```text
消息 text
   │
   ▼
冻结 Qwen3-Embedding-0.6B → L2 归一化向量 h，维度 1024
   ├─ 主分类头：4 个 logits → 温度缩放 → softmax → 主标签及置信度
   └─ Working 头：1 个 logit → sigmoid → 与 working 阈值比较
                                  │
主分类置信度不足 ──────────────────┴─→ DeepSeek 复核两个输出
```

数学形式：

```text
h = Encoder(text) / ||Encoder(text)||₂
z = W_route h + b_route
p = softmax(z / T)
label = [semantic, episodic, procedural, null][argmax(p)]
confidence = max(p)

q = sigmoid(W_working h + b_working)
working_state = (q ≥ θ_working)
```

| 参数 | 形状 | 是否训练 |
|---|---|---|
| Qwen encoder 参数 | 0.6B 级编码器 | 冻结，只生成向量 |
| `route.weight` | 4 × 1024 | 是 |
| `route.bias` | 4 | 是 |
| `working.weight` | 1 × 1024 | 是 |
| `working.bias` | 1 | 是 |

两个头总计 **5,125 个可训练参数**。轻量的是分类头；完整部署仍需要 Qwen 编码器。这里的“本地”是 companion 服务运行机器，不表示模型已在 iPhone 上运行。

## 5. 离线训练过程

1. **读取最终标注并校验划分。** 主分类使用所有训练样本；working 只使用具有布尔监督的训练样本。
2. **冻结编码器并生成 embedding。** 在 CPU 上以推理模式执行，输出归一化向量。缓存校验包含数据文件 SHA-256、模型路径以及模型文件大小/修改时间等信息；模型标识并非所有权重文件内容的完整哈希。
3. **分别拟合两个逻辑回归头。** 每个头尝试 `class_weight ∈ {None, balanced}`、`C ∈ {0.1, 1, 10, 100}`，共 8 种组合。以 MobileMem 验证集的 macro-F1 选择最优模型。
4. **校准主分类概率。** 在 `[0.25, 4]` 上取 101 个对数间隔温度，以 MobileMem 验证集 log-loss 最低者为温度 T。
5. **选择回退阈值。** 在 MobileMem 验证集上按置信度排序，要求至少接受 20 条，且已接受部分标签一致率达到 90%，在满足要求的候选中最大化覆盖率。
6. **选择 working 阈值。** 在 0.05～0.95 间以 0.01 步长搜索，最大化 MobileMem 验证集 working macro-F1；并列时优先接近 0.5 的阈值。
7. **保存 checkpoint、配置和指标。** 测试集只评估，不用于上述模型或阈值选择。

训练代码调用 sklearn `LogisticRegression(..., max_iter=1000)`；未指定 solver，使用安装版本的默认值。当前本地实现默认是 `lbfgs`（可见本地 sklearn 源码）。`1000` 是求解迭代上限，不是记录了 1,000 个 epoch。当前训练不使用旧脚本中的 AdamW 学习率。

当前保存配置：

| 项目 | 结果 |
|---|---:|
| 主分类 C | 10.0 |
| 主分类 class_weight | balanced |
| Working C | 100.0 |
| Working class_weight | balanced |
| 主分类温度 T | 1.2141948844 |
| 主分类回退阈值 τ | 0.8241907954 |
| Working 判断阈值 θ | 0.26 |

温度调节概率的尖锐程度，不改变同一 logits 下的 argmax 类别。上述 τ 是离线验证选出的固定阈值，与实验模式的动态阈值分开使用。

## 6. Weight 到底如何变化

### 6.1 模型权重 W、b：由数据拟合

主分类优化带正则的多分类交叉熵，working 优化带正则的二分类交叉熵。可用下式概括目标（具体归一化由 sklearn 实现决定）：

```text
L_route ∝ Σᵢ α[yᵢ] · (−log softmax(W_route hᵢ + b_route)[yᵢ])
          + λ · ||W_route||²

L_working ∝ Σᵢ α[wᵢ] · BCE(sigmoid(W_working hᵢ + b_working), wᵢ)
            + λ · ||W_working||²
```

`C` 越大，正则约束越弱。求解器依据损失和梯度调整 W、b，使正确标签获得更合适的分数。编码器没有参与反向更新。两个头独立拟合，没有共享训练损失或人为指定的 route/working 损失混合系数。

### 6.2 类别权重 α：补偿类别数量差异

选中的 `balanced` 使用：

```text
α[c] = N / (K × n[c])
```

N 为当前拟合数据总量，K 为类别数，n[c] 为该类数量。它是损失中的类别系数，与 `route.weight` 矩阵是两回事。

当前离线主分类中 N=2200、K=4；例如 null 只有 163 条，其权重为 `2200/(4×163)≈3.3742`，semantic 权重为 `2200/(4×577)≈0.9532`。Working 使用自身的 800 条有监督数据：true 权重约 1.1173，false 约 0.9050。

一次拟合期间这些系数由样本分布确定，不按消息时间衰减。在线重新拟合时，如果数据分布改变，`balanced` 会重新计算系数。

### 6.3 实验路由器的在线更新

实现：`scripts/hgp3_router.py` 的 `observe()` 和 `update()`。

只有同时满足下列要求的样本才进入更新缓冲：

- 启用 `adaptive=True`，且结果来自 LLM 回退。
- LLM 标注 `uncertain=false`。
- 不属于已有 MobileMem 验证或测试消息 ID。
- 规范化文本不在原始数据或已累计样本中。
- 与原训练向量及已累计样本的最大余弦相似度低于 0.90。

默认积累 **100 条新的合格样本**后触发更新。缓冲数量不等于全部消息数，也不等于全部 LLM 调用次数。

```text
D_update = 原始 2200 条训练数据 + history + pending

主分类：在 D_update 上重新拟合 LogisticRegression
Working：在 D_update 中有 working 标签的子集上重新拟合

替换两个头的 W、b
iteration += 1
history = history + pending
pending = []
```

这是**保留基础数据并回放累计样本的全量重拟合**。代码每次创建新的 LogisticRegression，没有 warm-start，也没有逐条 SGD 更新。旧样本和新样本没有额外的时间权重或新样本加权倍数；除类别平衡外，使用相同的样本处理方式。

更新继续使用离线选出的 C 和 `class_weight` 配置；编码器、温度 T、working 阈值保持不变，没有重新执行验证集校准。实验适配层将更新后的权重、iteration、history、pending 保存到各用户实验目录中的 `adaptive-router.pt`，不覆盖基础 `head.pt`。

### 6.4 动态阈值：随更新次数退火

实验路由器默认公式：

```text
τ(k) = max(0.5, 0.9 × exp(−0.05 × k))
```

k 是成功更新分类头的次数。

| 更新次数 k | 动态阈值 τ(k)，约值 |
|---:|---:|
| 0 | 0.9000 |
| 1 | 0.8561 |
| 5 | 0.7009 |
| 10 | 0.5459 |
| 12 及以后 | 0.5000 |

阈值降低后，在概率不变的情况下更多消息可以由本地分类器直接接受。**阈值退火与模型权重训练是两个独立机制**；退火不是根据实测准确率自动升降，也不证明模型已经变好。

### 6.5 当前 iPhone 入口的实际行为

`iphone-companion/classifier.py` 使用 `adaptive=False`、`state=None`，并将初始阈值改为保存配置中的 `0.8241907954`。iteration 保持为 0，因此实际阈值固定为该值。

- 高置信度：返回本地主类别和 working 判断。
- 低置信度：调用 DeepSeek 复核主类别与 working，补充“用户本人信息归属”和“是否仍需跟进”的约束。
- 复核失败：保留本地结果并标记 `uncertain=true`、`fallback_status=failed`。
- 此入口不调用 `observe()`，因此 DeepSeek 复核不会自动改变分类头权重。

目前触发复核只依据**主分类**最大概率，没有另设 working 置信度门控。因此主类别很确定、working 很不确定的消息，仍可能直接采用本地 working 结果。

## 7. 已保存的训练与评估结果

以下来自当前 `metrics.json`，使用固定校准阈值 0.8241907954；不代表开启动态适配之后的结果，也不包含 LLM 回退后的最终混合系统成绩。

| 评估集合 | n | 主分类标签一致率 | Macro-F1 | 本地接受数量 | 覆盖率 | 接受部分一致率 |
|---|---:|---:|---:|---:|---:|---:|
| 全部 Train | 2,200 | 95.91% | 0.9348 | 1,638 | 74.45% | 99.88% |
| 全部 Val | 343 | 83.97% | 0.8136 | 200 | 58.31% | 97.00% |
| 全部 Test | 334 | 82.93% | 0.7987 | 191 | 57.19% | 93.72% |
| MobileMem Train | 800 | 89.00% | 0.8741 | 287 | 35.88% | 99.30% |
| MobileMem Val | 192 | 72.40% | 0.6456 | 64 | 33.33% | 90.63% |
| MobileMem Test | 192 | 70.31% | 0.6514 | 61 | 31.77% | 80.33% |

| Working 评估集合 | n | 标签一致率 | Macro-F1 |
|---|---:|---:|---:|
| MobileMem Train | 800 | 96.38% | 0.9636 |
| MobileMem Val | 192 | 81.77% | 0.8135 |
| MobileMem Test | 192 | 84.38% | 0.8436 |

这些标签主要来自外部映射、合成数据和 DeepSeek。MobileMem 测试集只有 1 条标记为人工复核，因此应称为标签一致率或 proxy 指标，不能称为完整人工 gold 准确率。

验证集上的 90% 接受目标没有在 MobileMem 测试集保持：测试中只接受 61/192 条，其中一致率为 80.33%。这说明需要独立测试校准后的实际效果；不能把验证目标写成部署质量保证。

### 7.1 目前有哪些训练过程记录

已保存最终权重 `head.pt`、配置 `config.json`、向量缓存 `embeddings.npz` 和分集合指标 `metrics.json`。训练脚本没有保存逐次优化的 loss、每一步 W/b、各超参数候选得分或每轮在线权重快照。

因此本文能说明权重如何更新和最终结果，**不能提供未记录的真实 loss 曲线、逐 epoch 权重曲线或参数变化量**。若后续需要研究权重演化，应在每次在线更新前后记录样本分布、验证指标、`||W_new−W_old||`、阈值与独立 checkpoint；这属于后续新增记录功能。

## 8. 与 HGP 的区别及创新表述边界

对照依据是内嵌 HGP 仓库的提交 `265c7c2` 和 PocketMemory 当前实现。HGP 的 `train/` 在当前工作目录中缺失，但保存在该提交里，可用 `git -C HGP- show 265c7c2:train/bert.py` 查看。HGP 原版以 `uer/chinese_roberta_L-4_H-512` 初始化文本分类模型，训练后由 `train/infer.py` 加载 `memory_bert` checkpoint，使用 sigmoid 为四个标签分别打分。PocketMemory 的 iPhone 收件箱和 HGP 端到端对比实验均使用 Qwen 分类路由器。端到端对比实验通过桥接复用 HGP 的记忆存储。

| 维度 | 本地 HGP 文档/接口 | 当前 PocketMemory 实现 | 可如何描述 |
|---|---|---|---|
| 标签组织 | RoBERTa 输出 `episodic`、`semantic`、`procedural`、`none` 四个独立分数。原版入口可从 episodic 记录提取 working 状态 | 三种主记忆择一或 null，并独立预测 `working_state` | 主分类改为互斥输出，在同一次路由中直接预测 working |
| 无主记忆的消息 | `none` 是已有的多标签输出之一 | `null` 与三类主记忆共同参与 softmax | 差异在互斥约束，原版已有 `none` |
| 分类器训练 | 使用 Hugging Face Trainer 微调中文 RoBERTa 文本分类模型。增量训练可从已有 checkpoint 继续 | 冻结 Qwen3-Embedding-0.6B，拟合 5,125 个分类头参数。实验更新时重新拟合两个头 | 更换编码器和训练方式，只更新分类头 |
| 数据与校准 | 当前可见资料不足以核对 | 面向 MobileMem 的 teacher 标注、working 监督、验证选参和温度校准 | 目标场景的数据与校准方案 |
| 回退和增量更新 | PocketMemory README 将阈值退火归为沿用 HGP 的机制 | LLM 回退 + 去重缓冲 + 基础数据回放重拟合 | 继承机制与工程实现，不宜单独宣称原创 |
| 记忆存储 | 语义图、事件图、程序规则与工作状态 | 实验桥接继续复用 HGP 原版 memory 模块 | 主要继承 HGP，不是重新发明混合图存储 |

比较稳妥的表述是：

> PocketMemory 在 HGP 记忆组织框架上，将主记忆输出改为互斥选择，并直接预测独立的 working 状态。它用冻结 Qwen 编码器上的两个线性分类头完成消息路由，并结合目标场景校准与 LLM 回退。实验模式还实现了带去重和训练数据回放的用户级分类头更新。

这说明已实现的设计差异，不等于已证明论文级原创性或性能优越。特别是“多标签 → 单主标签”的选择也有代价：同一条消息不能同时保留多个长期记忆类别。需要在相同数据和存储条件下，通过标签结构、校准、回放与更新机制的消融实验，才能判断各项设计的贡献。

## 9. 代码与数据索引

- [数据准备与划分](../scripts/prepare_hgp3_2200.py)
- [MobileMem 标注与人工复核合并](../scripts/label_hgp3_2200.py)
- [标签定义 prompt](../scripts/hgp-message-label-prompt.txt)
- [当前离线训练与评估](../scripts/train_hgp3_2200.py)
- [在线路由、阈值退火与重拟合](../scripts/hgp3_router.py)
- [iPhone companion 实际分类入口](../iphone-companion/classifier.py)
- [HGP 原版存储桥接与 observe 调用](../scripts/hgp_original_layer.py)
- [最终标注数据](../outputs/hgp3-train-2200/annotations-unified/dataset.json)
- [已保存训练配置](../outputs/hgp3-train-2200/classifier/config.json)
- [已保存评估指标](../outputs/hgp3-train-2200/classifier/metrics.json)
- [HGP 本地说明](../HGP-/README.md)

当前 checkpoint 配置中的数据 SHA-256：

```text
563fc61028ddfae33f5467084fc94d3bc62bf71bde5bd9b4b99ff41edaebc522
```
