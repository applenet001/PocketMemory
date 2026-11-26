# HGP: 面向个性化交互场景的长文本理解与混合图记忆系统

HGP 是一个面向个性化交互场景的记忆增强与长文本理解系统。项目围绕“个性化交互场景下长文本理解训练方法成果”展开，面向连续对话、长期用户轨迹、偏好建模和上下文约束理解等任务，提供记忆分类路由、异构记忆存储、图结构组织、检索增强和周期性知识抽取能力。

系统将用户交互历史划分为语义记忆、情节记忆、程序性记忆和工作记忆，并通过分类器将长文本交互片段路由到合适的记忆类型中。相比简单向量池或扁平日志检索，HGP 更强调对长程用户轨迹的结构化理解、个性化状态维护和可复用知识沉淀。

## 项目定位

本项目可作为以下方向的基础实现：

- 个性化交互场景下的长文本理解与用户状态建模
- 面向智能体的长期记忆分类、存储和检索
- 用户偏好、历史行为和隐式约束的结构化表示
- 检索增强生成中的个性化上下文管理
- 端侧或轻量化场景下的记忆路由与增量更新

## 核心能力

- **多类型记忆建模**：支持 semantic、episodic、procedural、working 四类记忆。
- **个性化记忆路由**：通过训练得到的分类器识别输入文本应写入的记忆类型。
- **混合图存储**：结合图结构、向量检索和规则存储，组织长期用户轨迹。
- **长程上下文理解**：面向多轮交互和长文本日志，抽取用户偏好、事件、流程和当前状态。
- **知识抽取与巩固**：支持从情节记忆中周期性抽取语义知识和程序性规则。
- **统一调用接口**：提供 `process`、`search`、`extract` 三类命令行模式。

## 目录结构

```text
HGP/
├── data/                 # 示例数据与输入文件
├── experiment/           # 实验脚本与评估相关代码
├── memory/               # 四类记忆的存储、检索和抽取模块
├── pictures/             # 项目框架图
├── train/                # 记忆分类器训练与推理模块
├── main.py               # 统一入口
├── llm_api.py            # LLM API 调用封装
├── requirement.txt       # Python 依赖
└── README.md
```

## 系统架构

<p align="center">
  <img src="pictures/HGP_framework.png" alt="HGP Framework" />
</p>

## 记忆类型

| 记忆类型 | 作用 | 示例 |
|---|---|---|
| Semantic Memory | 存储稳定偏好、属性和概念知识 | 用户偏好低油低盐饮食 |
| Episodic Memory | 存储具体事件、时间和交互片段 | 用户上周购买了旅行指南 |
| Procedural Memory | 存储流程、规则和操作策略 | 若用户预订行程，先确认日期 |
| Working Memory | 存储当前会话状态和短期约束 | 当前任务处于信息收集阶段 |

## 安装

建议使用 Python 3.9 及以上版本。

```bash
git clone https://github.com/Ouan6/HGP.git
cd HGP
pip install -r requirement.txt
```

## 环境配置

如果需要使用 LLM 抽取或评估能力，请配置对应 API Key：

```bash
export OPENAI_API_KEY="your_api_key"
# 或
export DEEPSEEK_API_KEY="your_api_key"
```

如果启用 Neo4j 图存储，请配置：

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your_password"
```

Windows PowerShell 示例：

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USERNAME="neo4j"
$env:NEO4J_PASSWORD="your_password"
```


## 训练与分类器

`train/` 目录包含记忆分类器的训练和推理代码。分类器用于判断输入文本属于哪些记忆类型，是系统实现长文本理解和个性化记忆路由的核心模块之一。

典型流程包括：

1. 构造带有记忆类型标签的交互文本数据；
2. 训练多标签分类器；
3. 在 `main.py` 中通过 `InferencePipeline` 调用分类器完成在线路由。

## 实验与评估

`experiment/` 目录包含与实验评估相关的脚本。项目可用于评估：

- 长程交互场景下的个性化问答能力；
- 用户偏好和约束的检索效果；
- 不同记忆类型对下游任务的贡献；
- 图结构存储与向量检索的互补作用；
- 轻量化路由对在线推理成本的影响。


## License

This project is released under the MIT License. See `LICENSE` for details.
