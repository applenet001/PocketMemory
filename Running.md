# PocketMemory v1 安装和使用教程

PocketMemory 运行在你的 Windows 电脑或 Mac 上。iPhone 用 Safari 访问同一个 Wi-Fi 上的电脑。首次启动只需要本机的 Python 3.13 和源码 ZIP。记忆保存、附件下载和关键词搜索可以先用起来。

## 1. 下载并解压

从 GitHub 下载 v1 源码 ZIP，解压到「下载」文件夹，把解压出来的项目目录改名为 `PocketMemory-v1`。如果你收到的是 `PocketMemory-v1-source.zip`，解压后目录已经是这个名字。 [下载 release/v1 源码](https://github.com/applenet001/PocketMemory/archive/refs/heads/release/v1.zip)

已安装 Git 的用户可以在「下载」文件夹中运行下面的命令来下载同一版本。Windows PowerShell 和 macOS 终端使用同一条命令。

```sh
git clone --branch release/v1 --single-branch https://github.com/applenet001/PocketMemory.git PocketMemory-v1
```

打开项目目录，确认能看到 `README.md` 和 `iphone-companion/`。如果你把项目放到了别处，把下面 `cd` 后面的路径换成实际路径。路径含空格时，保留双引号。

## 2. 用本机 Python 启动

任选与你的电脑对应的一节。`.venv` 是在这台电脑上新建的项目环境，可以直接调用里面的 Python。

### Windows PowerShell

打开 **PowerShell**。运行 `py -3.13 --version`，看到 `Python 3.13.x` 就可以继续。如果 `py` 找不到，但 `python --version` 显示 `3.13.x`，把代码块里的两处 `py -3.13` 换成 `python`。如果两个命令都没有找到，安装 Python 3.13 后重新打开 PowerShell。 [Windows 安装说明](https://docs.python.org/3.13/using/windows.html)

```powershell
cd "$HOME\Downloads\PocketMemory-v1"
py -3.13 --version
py -3.13 -m venv .venv
& .\.venv\Scripts\python.exe .\iphone-companion\server.py --host 0.0.0.0 --port 8766
```

### macOS 终端

打开 **终端**。运行 `python3.13 --version`，看到 `Python 3.13.x` 就可以继续。找不到命令时，安装 Python 3.13，按安装说明运行 `Install Certificates.command`，然后重新打开终端。 [macOS 安装说明](https://docs.python.org/3.13/using/mac.html)

```bash
cd "$HOME/Downloads/PocketMemory-v1"
python3.13 --version
python3.13 -m venv .venv
./.venv/bin/python ./iphone-companion/server.py --host 0.0.0.0 --port 8766
```

## 3. 登录并连接 iPhone

1. 保持终端打开。在电脑浏览器访问 [http://127.0.0.1:8766](http://127.0.0.1:8766)。启动时终端会显示 `Administrator key`。复制这串密钥登录，它也保存在电脑上的 `iphone-companion/runtime/access-key.txt`。
2. 在「用户管理」中创建个人账号，保存当时显示的个人密钥。用个人密钥进入「记忆库」和「上传 / 记录」。
3. iPhone 和电脑连同一个可信 Wi-Fi。在 Safari 打开终端显示的 `iPhone LAN` 地址，例如 `http://192.168.1.20:8766`。如果显示多个地址，选电脑当前 Wi-Fi 的 IPv4 地址。iPhone 要用这个局域网地址。
4. 在 iPhone 用个人密钥登录。可以通过 Safari 分享菜单「添加到主屏幕」。需要从其他 App 分享时，在网页打开「iPhone 分享」，按说明配置快捷指令。

停服务时在终端按 **Ctrl+C**。下次启动只需进入同一个目录，再执行相应平台代码块的最后一条命令。换 Wi-Fi 后，按本次启动显示的地址更新主屏幕入口和快捷指令。电脑端服务在终端关闭或电脑休眠时不可访问。

## 4. 按需安装功能

先按 **Ctrl+C** 停止服务，在同一个项目目录执行所需命令。装完后，重新执行第 2 节的启动命令。

### 图片 OCR 和 PDF 文字提取

**Windows PowerShell**

```powershell
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r .\iphone-companion\requirements.txt
```

**macOS 终端**

```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r ./iphone-companion/requirements.txt
```

第一次使用图片 OCR 时，RapidOCR 可能下载公开模型。识别在本机运行。带文字的 PDF 可以提取，扫描版 PDF 会保存原件。

### 个人「按意思搜」和本地自动分类

个人语义搜索使用本地 Qwen 模型。下载约 1.2 GB。以下模型安装步骤面向 64 位 Windows 和 Apple Silicon Mac。Intel Mac 上的 Python 3.13 若装不到兼容的 PyTorch，可使用基础收件箱和关键词搜索。 [Qwen 模型文件](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/main) · [PyTorch 安装说明](https://pytorch.org/get-started/locally/)

**Windows PowerShell**

```powershell
& .\.venv\Scripts\python.exe -m pip install "sentence-transformers>=5.2.3,<6"
& .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/qwen3-embedding-0.6b')"
```

**macOS 终端**

```bash
./.venv/bin/python -m pip install "sentence-transformers>=5.2.3,<6"
./.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/qwen3-embedding-0.6b')"
```

确认 `models/qwen3-embedding-0.6b/` 里有 `modules.json` 和 `model.safetensors`。重启后，用个人账号在「记忆库」输入一句话，点「按意思搜」。首次加载模型会稍慢。

自动分类需要你另外提供训练好的分类头。将自己的 `head.pt` 和配套的 `config.json` 放入 `outputs/hgp3-train-2200/classifier/`。公开源码 ZIP 中只包含代码和教程。 [分类头准备说明](docs/classifier-training-and-weights.md)

### 近况速览和 DeepSeek 复判

**Windows PowerShell**

```powershell
& .\.venv\Scripts\python.exe -m pip install "openai>=2.54,<3"
```

**macOS 终端**

```bash
./.venv/bin/python -m pip install "openai>=2.54,<3"
```

在项目的 `config/` 内，用纯文本编辑器创建 `local.env`。把占位内容替换成你自己的 DeepSeek 密钥。已有 `local.env` 时，在原文件追加或更新这一行。

```dotenv
DEEPSEEK_API_KEY=填入你自己的密钥
```

点「生成近况速览」时，会将近期记忆的文字片段发送给 DeepSeek。启用本地自动分类后，低置信度复判也会发送对应记录的正文和备注。这个文件只保存在你自己的电脑上。

### 个人记忆关系图

先在电脑上启动可创建多个数据库的 Neo4j。在 `system` 数据库中执行下面的命令，并等 `usermemory` 上线。 [Neo4j 多数据库说明](https://neo4j.com/docs/operations-manual/current/database-administration/standard-databases/create-databases/)

```cypher
CREATE DATABASE usermemory IF NOT EXISTS;
SHOW DATABASE usermemory;
```

**Windows PowerShell**

```powershell
& .\.venv\Scripts\python.exe -m pip install "neo4j>=6,<7"
```

**macOS 终端**

```bash
./.venv/bin/python -m pip install "neo4j>=6,<7"
```

在同一个 `config/local.env` 里添加本机地址和凭据。符合条件的语义记忆和事件会同步到个人图。

```dotenv
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=填入你自己的Neo4j密码
```

### 管理员实验搜索

「实验搜索」读取已有的实验库。源码包已有当前版 MemBase 代码。完成上面的模型、DeepSeek 和 Neo4j 配置后，还需要准备 `semantic` 和 `episodic` 数据库、`outputs/mobilemem-qwen/hgp-history-time-qa/`、对应的 MobileMem 数据和实验用的 512 维模型。用自己的实验备份恢复数据和图库。 [HGP3 实验工作流](docs/hgp3-2200-workflow.md)

**Windows PowerShell**

```powershell
& .\.venv\Scripts\python.exe -m pip install rank-bm25 python-dotenv
& .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/distiluse-base-multilingual-cased-v2', local_dir='models/hgp-distiluse-base-multilingual-cased-v2')"
```

**macOS 终端**

```bash
./.venv/bin/python -m pip install rank-bm25 python-dotenv
./.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/distiluse-base-multilingual-cased-v2', local_dir='models/hgp-distiluse-base-multilingual-cased-v2')"
```

## 5. 常见问题

| 现象 | 检查方法 |
|---|---|
| 找不到 `iphone-companion/server.py` | 确认 `cd` 进入的是含 `iphone-companion/` 的解压目录。 |
| 电脑能打开，iPhone 连不上 | 检查是否同一 Wi-Fi、有无访客网络隔离，以及 `8766` 是否被本机防火墙拦截。只在可信局域网允许访问。 |
| 端口 `8766` 已占用 | 退出已运行的实例，或给启动命令换一个空闲端口，再按终端显示的新地址访问。 |
| 「按意思搜」显示关键词结果 | 确认模型文件和 `sentence-transformers` 都安装在本项目使用的环境里。 |
| 近况速览无法生成 | 检查 `.venv` 中的 SDK、`config/local.env` 的密钥和电脑的网络连接。Windows 上请确认文件名没有自动追加 `.txt`。 |

停服务后，可以备份整个 `iphone-companion/runtime/`。这个目录保存账号、访问密钥、记忆和附件。另有 Neo4j 图数据时，请同时备份对应数据库。
