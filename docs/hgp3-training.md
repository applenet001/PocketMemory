# HGP 三类轻量 classifier（3,000 条）

实现：冻结本地 Qwen embedding，仅训练三类 softmax 线性头。working 独立维护状态，
不作为第四个互斥类别。已有四标签训练脚本不变。

## 数据与标签

来源及解析代码在 `scripts/hgp3_data.py`，抽样在 `scripts/prepare_hgp3.py`。

| 来源 | 数量 | 单位与监督 |
|---|---:|---|
| TriMEM | 600 | 完整 Content；合并 personal/general semantic，仅选择合并后单标签行 |
| PerLTQA zh | 900 | 人物描述/关系映射 semantic，事件映射 episodic；不使用整段 dialogues |
| proScript | 600 | 完整场景及流程映射 procedural，属于弱监督，没有原生句子级 procedural 标签 |
| IMCS21 | 300 | 告知用药建议/就医建议/注意事项的句子映射 procedural；属于弱监督，不能证明每条都是可复用规则 |
| MobileMem | 600 | session message 的连续证据片段，DeepSeek 三类预标；前两条消息仅辅助消歧 |

不读取 MobileMem profiles、graphs、QA、答案或生成的摘要作为输入。
MobileMem 长文本按 1,000 字符连续分块；来源及偏移保留，截断边界不明确的内容应排除或人工审核。
外部文本保留原注释单位，超 3,000 字符跳过。编码器发现超 token 上限会报错，不静默截断。
数据没有类别均衡保证；manifest 记录实际比例，训练采用类别权重。

按人物/病例/场景分组，去除规范化后的完全重复文本。不是语义近重复检测。
MobileMem 按 session 分组，不按用户隔离，同一 session 不跨集合；可以评估已有用户的新会话，
不能据此宣称跨用户泛化。每个来源80/10/10，总计 train=2,400、val=300、test=300。
`reserve.jsonl` 保存备用候选；排除样本后必须补足并重新完成相应标注，程序不会伪造到3,000条。

## MobileMem 混合标注

只调用 DeepSeek 标注 MobileMem；外部来源不调用。返回 label、uncertain、reason，
混合/不明确/无价值可以 label=null。保留原始响应与 usage，支持按 ID 断点续跑。
固定模型、prompt 和候选池指纹，避免混用不同版本。

按预测类别分层随机审核20%，向上取整可能略多于120条；不确定样本额外审核。
参考 MemoType/TriMEM 附录 B.3：1,200/6,000条（20%）分层人工核验：
https://arxiv.org/html/2610.11573v1#A2.SS3
这是比例依据，不保证我们的质量等同于论文；原论文至少两人独立核验，我们记录实际 reviewer。

编辑 `review.jsonl`：填写 `final_label`（同意模型也要填）、`reviewed=true`、`reviewer`、
`review_note`；不适合单标签的内容设 `exclude=true`。不能修改 text/context。
默认不增加外部人工工作；如需检查所有 proScript/IMCS21 弱映射，用 `review --review-weak`。
任何一条未完成审核，不能 finalize。未经人工审核的外部映射和 MobileMem teacher 标签
均不作为 human gold。最终同时报告 proxy 指标、来源指标与人工审核子集指标。

## 命令（项目根目录）

```powershell
$py = './pocketmemory/Scripts/python.exe'
& $py scripts/prepare_hgp3.py
# 先20条试标；DEEPSEEK_API_KEY 来自环境或 config/local.env。每次命令都会调用付费API。
& $py scripts/label_hgp3.py label --limit 20
# 检查试标结果后继续；自动跳过已完成ID
& $py scripts/label_hgp3.py label --limit 600
& $py scripts/label_hgp3.py review --fraction 0.2
# 人工编辑 outputs/hgp3-3000/review.jsonl
& $py scripts/label_hgp3.py finalize
& $py scripts/train_hgp3.py
& $py scripts/test_hgp3_pipeline.py
```

输出：`head.pt`、`config.json`、`metrics.json`、`test-predictions.jsonl`。
部署仍依赖同一冻结编码器；小的是分类头，不是整个0.6B编码器。
置信度回退阈值只由验证集选取；外部映射验证分数不能解释为人工gold准确率。

## proScript 中文对话改写

`scripts/synthesize_proscript.py --limit 600` 对已选600条流程逐条改写，仍沿用480/60/60划分。
每条包含中文用户提问 `context` 和完整方法回答 `text`，原始流程保存在 `original_text`。
独立的第二次DeepSeek调用复核遗漏、新增和顺序；自动检查不是人工gold。
`source_issue=true` 或自动复核失败的样本进入 `manual-review.jsonl`。
结果单独保存于 `outputs/proscript-dialogue-600/`，不会自动替换核心候选。
同一parent_id的英文、中文变体不能分到不同集合。脚本支持断点续跑。
