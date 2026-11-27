# iPhone Companion

启动服务、安装依赖和连接 iPhone 见 [Quick Start](../README.md#quick-start)。

## iPhone 快捷指令

在 iPhone 的 Safari 登录个人账号，进入「iPhone 分享」，复制页面给出的接收地址。

1. 新建「存到 PocketMemory」快捷指令，开启「在共享表单中显示」，接收类型选择图像、PDF 和文件。
2. 添加「获取 URL 内容」，使用接收地址和 `POST` 方法。标头填 `Authorization`，值填 `Bearer 你的个人密钥`。
3. 请求体选「表单」，添加一个名为 `file` 的文件字段，值选「快捷指令输入」。
4. 从照片或文件的分享菜单运行，在「记忆库」确认保存结果。分享链接或选中的文字时，接收类型选 URL 和文本，先用「获取输入中的文本」转换输入，再把结果放进表单的 `text` 文本字段。

可以在快捷指令中用「询问输入」收集来源 App，再把答案作为可选的 `source_app` 字段一起上传。网页手动上传也可以填写来源 App。来源留空时，详情显示「未提供」。

后台用 RapidOCR 运行 PaddleOCR 的 PP-OCR 模型，保存每行识别文字、置信度和位置。PDF 优先读取内嵌文字，最多对 8 个缺少可读文字的页面尝试识别内嵌图片。后台用可靠文字生成草稿。正文无法读取时，详情仍提供原件、文件名和来源 App。

DeepSeek 复判可以为同一条记忆返回多个有明确证据的类别。在「补充正文或更正标签」中可以确认主标签、附加标签和是否需要跟进，也可以补充或替换正文。保存后更新个人分类和相关记忆视图。每条记忆最新确认的主标签和可用正文会用于该账号之后的分类。增量拟合复用 `classifier/embeddings.npz` 和配套的 `annotations-unified/dataset.json`。

## 构建个人记忆图

### 安装分类模型

安装 Quick Start 中的依赖并解压预置 `outputs/`。本地分类读取 `outputs/hgp3-train-2200/classifier/` 下的 `head.pt` 和 `config.json`。在项目根目录安装分类依赖并下载 Qwen。适用于 64 位 Windows 和 Apple Silicon Mac。

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

### 创建并连接 usermemory

启动支持多数据库的本机 Neo4j。在 `system` 库执行 `CREATE DATABASE usermemory IF NOT EXISTS;`。 [安装 Neo4j Desktop](https://neo4j.com/docs/desktop/current/installation/)

完成 Quick Start 的本机服务配置后，重新启动 Companion。 [本机服务配置](../README.md#2-配置本机服务)

分类确定的语义和事件上传会自动同步到 `usermemory`。Companion 按 `namespace` 隔离账号，按 `route` 显示视图。

| `route` | 节点标签 | 页面视图 |
| --- | --- | --- |
| `semantic` | `SemanticCandidate` | 我的偏好与概念 |
| `episodic` | `ImportedEpisode` | 我的事件 |

符合条件的历史记录或同步失败的记录，可以在详情里点「同步到个人图」。

个人图中每条上传对应一个节点。兼具语义和事件标签的记忆会出现在两个视图里，共用一个节点。可在 Companion 点开节点查看上传原件和来源 App。

## 管理员与个人密钥

- 管理员用终端显示的 `Administrator key` 登录。密钥也保存在电脑的 `iphone-companion/runtime/access-key.txt`。
- 管理员在「用户管理」创建、停用账号或重置密钥。每个用户用自己的个人密钥登录和配置快捷指令。重置后须更新快捷指令。
- 管理员的实验图读取实验用的 `semantic` 和 `episodic` 库。 [HGP3-2200 实验指南](../docs/hgp3-2200-workflow.md)
