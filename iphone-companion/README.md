# PocketMemory iPhone Companion v0.2 · 多用户本地版

在 Windows 或 macOS 上运行服务，使用 iPhone 的 Safari 和系统快捷指令分享内容。电脑端提供 SQLite 存储、个人搜索、原件下载、删除和文字索引导出。安装可选依赖后还可以提取图片和 PDF 的文字。

iPhone 和运行服务的电脑需要连同一个可信局域网。手机端快捷指令可按页面内说明手动创建一次。

## 这台电脑启动

按对应系统的教程用本机 Python 启动。终端会显示以下信息。 [Windows 和 macOS 启动命令](../Running.md#2-用本机-python-启动)

- 本机入口 `http://127.0.0.1:8766`
- 一个或多个 `iPhone LAN` 地址，选当前 Wi-Fi/以太网对应的地址
- `Administrator key`，仅供管理员登录；不要交给普通用户

用手机 Safari 打开 LAN 地址，输入访问密钥。在 Safari 分享菜单中选择“添加到主屏幕”。页面里的“iPhone 分享”有完整配置步骤。首次联网请求需要在手机上允许快捷指令连接电脑地址。

## 创建独立用户

1. 使用原 `runtime/access-key.txt` 的密钥登录管理员。升级后旧浏览器 cookie 不再充当登录凭据，需要重新登录。
2. 打开“用户管理”，输入名称，点击“创建用户”。名称可重复，内部用户 ID 唯一。
3. 保存页面仅此时显示的个人密钥，单独交给该用户。密钥遗失时可重置；服务器不保存普通用户密钥明文。
4. 用户在自己的 iPhone 输入个人密钥登录；快捷指令的 Authorization 也使用这把个人密钥，而不是管理员密钥。
5. 页面顶部显示当前用户，支持退出登录。管理员可停用/启用用户、重置个人密钥；这些操作会撤销该用户的浏览器会话，重置还会使旧快捷指令密钥失效。

服务器从凭据确定用户 ID，不接受客户端指定存储身份。列表、检索、上传、去重、下载、导出和删除均使用该用户独立 SQLite 库；不同用户上传同一文件也不会共享记忆记录。

新上传先保存原文和附件并立即返回，后台依次进行 OCR/PDF 提取、本地 HGP3 分类和记忆同步。页面每 3 秒刷新处理阶段；首次加载本地 Qwen 编码器可能较慢。低于训练配置阈值时，正文和备注发送给 DeepSeek 复判，保留本地预测及复判理由；复判失败或仍不确定时不自动同步，失败可在详情重试。只有链接或没有可用文字的附件会跳过分类。服务重启会重新检查历史记录。置信度达标或复判确认的 Semantic/Episodic 同步到个人图，Working/Procedural 的原文记录保存到各用户的 `hgp/uploads/`；Procedural 标为 `source_only`，条件、步骤及跨记录任务关系仍需另行提取。个人账号的「任务与流程」展示本人文件中的任务状态和方法原文，可回看原始记忆；记录当时的 Working 状态不表示任务今天仍未完成。

```text
runtime/
  access-key.txt              原管理员密钥，仅保存在电脑
  accounts.sqlite3           用户、密钥哈希、带期限的会话哈希
  memories.sqlite3           原有收件箱，保留归属管理员
  users/<随机用户ID>/
    memories.sqlite3         该用户的原件与记忆
    hgp/uploads/             该用户的 working / procedural 原文记录
```

管理员可以在实验图视图查看原有实验图，并在“任务与规则”查看 `outputs/` 历史记录。个人图只显示当前登录账号的数据空间，本地历史接口只读取该用户的 `hgp/` 目录。管理员可管理其他用户的账户，个人画像与记忆库仍显示当前账号自己的数据。

管理员登录后进入“实验画像”，可以按 MobileMem 原始姓名选择评测用户。画像参考该用户每类最近 5 条记录，优先按原始记录时间排序。第二个标签“实验搜索”接收整句问题，复用评测检索，每类最多 5 条。Episodic 在电脑后端用 DeepSeek 解析问题中的事件线索，密钥由本地配置读取。Working 使用 BM25。任务与流程有独立标签，展示记录中的状态变化和规则树。能匹配同一条原始消息时，Working 图可展开对应的事件。管理员的记忆关系图也按评测用户选择。

实验搜索显示检索结果后，使用评测的 HGP 上下文和问答提示词请求 DeepSeek 回答。问题和本次检索内容由服务端发送，密钥从 `config/local.env` 或电脑端 `DEEPSEEK_API_KEY` 环境变量读取。浏览器只接收结果和临时检索票据。

近况速览由同一 DeepSeek 模型生成。实验画像使用选中评测用户的记录。个人用户主动点击“生成近况速览”后，服务才发送其近期记忆的文字摘录。页面保留原始片段的语言。

如果 Windows 防火墙询问网络访问，只允许专用网络。服务端没有自动修改防火墙。保持电脑开机且不休眠；终端 Ctrl+C 停止服务。路由器变更电脑 IP 后，要同步更新快捷指令 URL 和主屏幕入口。

## 运行测试

按安装教程创建 `.venv` 并安装图片和 PDF 提取依赖。先停止正在运行的服务，再在项目根目录运行测试。

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s .\iphone-companion -p 'test_*.py' -v
```

```bash
./.venv/bin/python -m unittest discover -s ./iphone-companion -p 'test_*.py' -v
```

首次初始化 RapidOCR 可能下载公开模型，识别在本机进行。提取依赖缺失时，系统会保存原件并显示提取状态。

## 分享接口

`POST /api/memories`，标头 `Authorization: Bearer <该用户的个人密钥>`。`user_id` 请求字段不会改变上传归属。

文字/链接使用 JSON：

```json
{"text":"https://example.com/article","note":"下周再看","title":"文章"}
```

图片/文件使用 multipart 表单，文件字段名 `file`，可附带 `note`、`title`、`text`。快捷指令里使用“获取 URL 内容”→ POST → 请求体“表单”→ `file` 类型“文件”。每次一个文件，多项分享使用“重复每一项”。确认接口返回成功后才算保存；相同内容、备注和文件名的重试会去重。

接口同时支持 JSON `base64`、`name`。单文件限制 20 MiB，请求限制 30 MiB。

## 当前能力与边界

- 管理员的“任务与规则”读取 `outputs/` 下已有文件，普通用户只读取自己的 `runtime/users/<id>/hgp/`。选择 working 或 procedural，再读取来源。Working 展示状态、时间、来源和转移原因，每个任务最多最近 200 个状态；procedural 展示最多 200 条规则及条件、步骤、原文、版本。只读，不运行 HGP 模型、不修改历史记录。
- Working 中的“最新记录状态”指文件最后记录且仍标为 active 的状态，并不表示今天仍在进行；时间不是分类为 working 的唯一依据。
- “上传 / 记录”使用同一个选择框接收图片和文件，可预览浏览器支持的图片。点击“保存记忆”后上传到本机服务。图片文字通过 OCR 提取，上传内容保存在个人记忆库。
- 支持管理员邀请创建多用户，不开放匿名注册。每个试用者使用自己的密钥；不要向读者公开管理员密钥。HTTPS 尚需配置。

- 文本、Markdown、CSV、JSON 提取 UTF-8 正文；PDF 最多前 100 页、20 万字符。扫描 PDF 只保存原件，不做逐页 OCR。
- PNG/JPEG/WebP/BMP 使用本地 OCR。HEIC、Office 等其他格式可以保存和下载，但不提取正文；可在快捷指令中先将 HEIC 转成 JPEG。
- 链接保存来源地址，不抓取网页。需要正文时，另外分享选中文字、文章文本或截图。
- 默认按文字关键词搜索个人记忆，中文支持片段匹配。安装本地 Qwen 模型后，可点「按意思搜」检索同一账号的相关记忆。
- 页面没有离线缓存。电脑离线或手机离开局域网后不能保存或检索。
- 不收集其他 App 的未分享数据，不索取整机读取权限。
- `runtime/` 内保存密钥、SQLite 数据库与原件，已被 Git 忽略。停止服务后备份整个文件夹；导出的 JSON 只有文本与元数据，不含附件。
- 密钥持有者可以访问该密钥所属用户的记忆。浏览器会话最长有效 7 天；退出登录撤销当前会话，管理员停用/重置可撤销该用户所有会话。局域网 HTTP 不加密，不要转发端口到公网。多人真实敏感数据试用前应配置 HTTPS。
- 删除会移除数据库记录和附件引用，但不保证底层磁盘安全擦除，也不会删除导出副本或另行导入的 HGP 数据。

## Neo4j 图与本地 AI 边界

“Neo4j 图”使用 `config/local.env` 连接本机的 Neo4j 实例。个人事件和语义候选存入独立的 `usermemory` 数据库，按用户命名空间隔离。管理员的实验视图读取原有 `semantic` 和 `episodic` 库。在另一台电脑初次部署时，可在 Neo4j 的 `system` 数据库执行 `CREATE DATABASE usermemory IF NOT EXISTS`。

置信度达标的 Semantic 和 Episodic 上传自动同步到 `usermemory`，点节点可查看原始记忆。符合条件的历史记忆或同步失败的记忆可在详情点击“同步到个人图”。Semantic 节点表示候选来源。原版 HGP 的事实、关系与程序规则提取仍需要另行接入本地提取器。

## 接入已有 HGP（验证接口）

先在网页导出 `pocketmemory.json`。适配器把每条记录转为 MemBase `Message`，保留分享时间、来源地址和稳定 ID。默认只检查，不写数据库：

```powershell
& .\.venv\Scripts\python.exe .\iphone-companion\hgp_bridge.py .\pocketmemory.json --user-id my-iphone
```

```bash
./.venv/bin/python ./iphone-companion/hgp_bridge.py ./pocketmemory.json --user-id my-iphone
```

`--apply` 在本地模式下会拒绝执行，避免部分写入后才遇到云端提取依赖。另有必须显式指定的 `--allow-cloud-extraction` 兼容开关；它会将分享正文交给原有云端提取器，不符合本次本地 Edge AI 的选择，不应启用。若未来明确改用该模式，将使用独立 `iphone-shares-v1` 命名空间和 `outputs/iphone-hgp/` 状态目录，不与收件箱同步删除。未进行真实图写入测试。

## HTTPS 支持

服务接受 `--tls-cert <证书 PEM> --tls-key <私钥 PEM>`，启用 TLS 1.2+ 和 Secure cookie。证书必须匹配手机访问的主机名/IP 且被 iPhone 信任；不会自动修改手机信任设置。当前默认启动脚本仍是局域网 HTTP 测试模式，不能将它当作已完成加固的安全版本。使用个人敏感记忆前，应配置有效 HTTPS、电脑磁盘加密和访问权限。本次未部署云端，也未调整系统证书或防火墙。

## 官方参考

- [Apple：从其他 App 运行快捷指令](https://support.apple.com/en-lamr/guide/shortcuts/apd163eb9f95/10.0/ios/27)
- [Apple：快捷指令的 POST / JSON / 表单请求](https://support.apple.com/en-lamr/guide/shortcuts/apd58d46713f/ios)
- [RapidOCR 使用说明](https://github.com/RapidAI/RapidOCRDocs/blob/main/docs/quickstart.en.md)
