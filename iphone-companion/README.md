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

如果快捷指令能取得 Todo 的稳定任务 ID，可以补充 `source_app=Todo`、`todo_id` 和 `is_completed`。JSON 使用布尔值，表单使用 `true` / `false`。同账号、同任务的完成记录可以把匹配的 Working 状态提前标为 inactive，并保存触发消息 ID 和接收时间。只输入来源 App 或普通截图也能走 v2 分类。

## HGP v2 个人记忆

新分享由电脑端运行 v2。文字和 OCR 正文交给 `outputs/four-head-500` 的本地 Qwen 分类头，复用 HGP 原版的低置信度 fallback prompt 和 adaptive threshold。DeepSeek 会收到需要复判或提取结构的正文。事件和语义进入本机 `v2-episodic`、`v2-semantic`。Working 从最近事件推断，与 Procedural 一起保存在当前账号的 `hgp/v2` 目录。每个账号有独立的图 namespace 和 adaptive 状态。按来源 App 分配 Working 轨迹，来源未知的分享共用一条轨迹。

启动前按 Quick Start 配置 Neo4j 和 DeepSeek，准备 `outputs/four-head-500`、`models/qwen3-embedding-0.6b` 和 `models/hgp-distiluse-base-multilingual-cased-v2`。本机 Neo4j 需支持多数据库，首次处理时会建立 v2 图库。缺少依赖或处理失败时，收件箱仍保留原件，可从详情重试。

**Windows PowerShell**

```powershell
& .\.venv\Scripts\python.exe -m pip install "sentence-transformers>=5.2.3,<6" "python-dotenv>=1,<2"
```

**macOS 终端**

```bash
./.venv/bin/python -m pip install "sentence-transformers>=5.2.3,<6" "python-dotenv>=1,<2"
```

在个人图中选择带 `HGP v2` 的视图，在「任务与流程」查看 Working 轨迹和 Procedural 规则。早期分享仍对应原来的 `usermemory` 图视图，服务启动时只恢复尚未处理完的 v2 分享。

可以在详情补充无法提取的正文。已经生成 v2 记忆的记录，后续手动更正只更新收件箱。删除分享时会清理它直接生成且可以通过来源 ID 定位的 v2 记录。HGP 之前合并、替换的知识或已完成的 adaptive 训练无法按一条分享回滚。

手机上传的 DeepSeek 用量记在各账号的 `hgp/v2/deepseek-usage.jsonl`。每条日志包含上传 ID、模型和 API 返回的输入、输出及缓存 token 数据。

## 管理员与个人密钥

- 管理员用终端显示的 `Administrator key` 登录。密钥也保存在电脑的 `iphone-companion/runtime/access-key.txt`。
- 管理员在「用户管理」创建、停用账号或重置密钥。每个用户用自己的个人密钥登录和配置快捷指令。重置后须更新快捷指令。
- 管理员的实验图读取实验用的 `semantic` 和 `episodic` 库。 [HGP3-2200 实验指南](../docs/hgp3-2200-workflow.md)
