# PocketMemory — IERG 5250 Course Project

PocketMemory 是一个个人记忆工具。你可以通过 iPhone 的 Safari 或快捷指令，把截图、PDF、链接和选中的文字存进个人记忆库，之后回看原件和提取的文字。

服务运行在 Windows 或 Mac 上。配置 HGP v2 后，系统会把分享内容整理为事件、事实与偏好、可复用的操作规则和当前任务状态。你可以在个人记忆图和「任务与流程」页面查看这些记忆。

## 上游项目与课程贡献

- **HGP** 提供原始的混合图记忆架构与实现，包括情节、语义、程序性和工作记忆。 [HGP 原始仓库](https://github.com/Ouan6/HGP-)
- **MemBase** 提供 MobileMem 的评测框架。本项目沿用其 Construct、Search 和 Eval 三阶段流程。 [MemBase 原始仓库](https://github.com/zjunlp/MemBase)

**Qilin Zhou** 负责本课程项目中 HGP 对 MobileMem 的端到端适配、记忆分类器的增量训练，以及 HGP 在 MobileMem 文本轨道上的测试与评测。 [HGP v2 与 MobileMem 实验说明](docs/hgp-v2-mobilemem.md)

## Quick Start

### 1. 下载

- 准备 Python 3，建议使用 3.11–3.13。 [下载 Python](https://www.python.org/downloads/)
- 下载 `release/v2` 源码 ZIP，解压到「下载」文件夹并将项目目录命名为 `PocketMemory-HGP`。 [下载 v2 源码 ZIP](https://github.com/applenet001/PocketMemory/archive/refs/heads/release/v2.zip)
- **可选** 下载预置数据 ZIP，将里面的 `outputs/` 放到项目根目录，与 `iphone-companion/` 同级。它包含现成的分类权重和历史实验数据。 [下载 outputs ZIP](https://drive.google.com/file/d/1J7TxH57XUXF4oxxzkcYbJW08zs7PL3jp/view?usp=sharing)
- **可选** 复跑 MobileMem 或查看分类数据来源时，下载 `PocketMemory-data.zip`，将包内文件按原有路径放到项目根目录。 [下载 PocketMemory 数据 ZIP](https://drive.google.com/file/d/1rpNIjcEW3qmnFAbtix0H-LizOpjnDrVS/view?usp=sharing) · [文件清单](docs/hgp3-2200-workflow.md#下载源数据)

### 2. 配置本机服务

要使用 DeepSeek 或 Neo4j，在项目根目录创建 `config/local.env`，填入所用服务的配置。使用基础收件箱可以跳过这一步。

```dotenv
DEEPSEEK_API_KEY=填入你的DeepSeek密钥
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=填入你的Neo4j密码
```

DeepSeek 用于近况速览和低置信度复判，会收到相应的记忆片段。使用个人图前，请启动本机 Neo4j。 [申请 DeepSeek 密钥](https://api-docs.deepseek.com/) · [安装 Neo4j Desktop](https://neo4j.com/docs/desktop/current/installation/)

### 3. 启动 Companion

选择对应系统的命令。`pip install` 一行安装 OCR、PDF、DeepSeek 和 Neo4j 依赖。只用基础收件箱时可以跳过该行。

#### Windows PowerShell

```powershell
cd "$HOME\Downloads\PocketMemory-HGP"
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r .\iphone-companion\requirements.txt "openai>=2.54,<3" "neo4j>=6,<7"
& .\.venv\Scripts\python.exe .\iphone-companion\server.py --host 0.0.0.0 --port 8766
```

#### macOS 终端

```bash
cd "$HOME/Downloads/PocketMemory-HGP"
python3 -m venv .venv
./.venv/bin/python -m pip install -r ./iphone-companion/requirements.txt "openai>=2.54,<3" "neo4j>=6,<7"
./.venv/bin/python ./iphone-companion/server.py --host 0.0.0.0 --port 8766
```

项目放在别处时，将 `cd` 后的路径换成实际位置。停服务按 **Ctrl+C**。下次启动只需进入项目目录，运行最后一条命令。

### 4. 连接 iPhone

1. 在电脑打开 `http://127.0.0.1:8766`，用终端显示的 `Administrator key` 登录，再到「用户管理」创建个人账号并保存个人密钥。
2. 保持服务运行。iPhone 和电脑连同一个可信 Wi-Fi，在 Safari 打开终端显示的 `iPhone LAN` 地址，用个人密钥登录。
3. 要从其他 App 分享内容，在网页打开「iPhone 分享」，按说明设置快捷指令。

## 文档

- [iPhone 快捷指令、个人记忆图与管理员](iphone-companion/README.md)
- [HGP3-2200 分类器训练（Windows / macOS）](docs/classifier-training-and-weights.md)
- [HGP3-2200 数据构建与改进](docs/hgp3-2200-workflow.md) · [数据构建](docs/hgp3-2200-workflow.md#1-构建分类数据)
