# 企业级智能客服与工单协同中枢

基于 **LangGraph 状态机 + FastAPI + SSE** 的企业级智能售后支持系统。
覆盖账号排障、权限申诉、退款政策问答与工单提报，并提供可视化 Web 工作台。

---

## 功能概览

| 能力 | 实现 |
|------|------|
| 知识库问答 | `query_knowledge_base` 工具检索 `data/docs/` 下的 Markdown 政策文档 |
| 工单提报 | `submit_ticket` 工具，生成 `TK-YYYY-XXXX` 流水号并模拟协同系统提交 |
| 意图编排 | LangGraph `StateGraph`：`agent ⇄ tools` 闭环，条件边路由 |
| 槽位填充 | 缺参时自然语言反问，**禁止**用占位符调用工单工具 |
| 流式接口 | `POST /api/chat/stream`，SSE 推送 token / tool_start / tool_end / done |
| Web 工作台 | 单文件原生前端，打字机渲染 + 工具调用胶囊卡片 |

## 快速开始

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥（复制模板后填入真实值）
Copy-Item .env.example .env
#    然后编辑 .env，设置 DASHSCOPE_API_KEY=<你的密钥>

# 3. 启动服务
python -m src.api.main
```

访问地址：

* 客服工作台：<http://127.0.0.1:8000/>
* 接口文档：<http://127.0.0.1:8000/docs>

### 无密钥离线运行

未配置 API Key 时，可启用离线 mock 模型验证状态机结构与路由（不调用 DashScope）：

```powershell
$env:CUSTOMER_AGENT_MOCK="1"
python -m src.agent.customer_agent     # 跑三个内置场景
python -m src.api.main                 # 以 mock 模式启动服务
```

## 项目结构

```
├── data/docs/                  知识库文档（账号排障指南、退款政策）
├── scripts/
│   ├── secret_scan.py          密钥扫描器（钩子与 CI 共用）
│   └── install_hooks.py        安装 pre-commit 钩子
├── src/
│   ├── agent/customer_agent.py LangGraph 状态机 + 业务 Prompt
│   ├── api/main.py             FastAPI 服务 + 静态页挂载
│   ├── api/test_sse.py         SSE 流式终端验证脚本
│   ├── api/static/index.html   Web 工作台（单文件，无需构建）
│   └── tools/                  业务工具函数
└── .github/workflows/          CI：密钥扫描兜底
```

## 密钥泄漏防护

本仓库提供**两层**防护，两者共用同一个扫描器 `scripts/secret_scan.py`，
因此判定规则不会漂移。

### 第一层：本地 pre-commit 钩子

```powershell
python scripts/install_hooks.py              # 安装
python scripts/install_hooks.py --uninstall  # 卸载
```

安装后每次 `git commit` 自动扫描**暂存内容**，发现疑似密钥即阻止提交。

> **注意**：钩子位于 `.git/hooks/`，属于本地配置，**不会被提交**。
> 团队成员需各自执行一次安装；且任何人都可用 `git commit --no-verify` 跳过。

### 第二层：CI 兜底（不可绕过）

`.github/workflows/secret-scan.yml` 在每次 push / PR 时扫描**当前文件与全部历史提交**。
即使本地钩子被跳过或未安装，CI 仍会拦截。

### 手动扫描

```powershell
python scripts/secret_scan.py --staged    # 暂存区（钩子内部调用）
python scripts/secret_scan.py --all       # 全部跟踪文件
python scripts/secret_scan.py --history   # 全部历史提交
```

### 检测能力

* **强特征**：`sk-`（OpenAI/DashScope）、`AIza`（Google）、`AKIA/ASIA`（AWS）、
  `ghp_/github_pat_`（GitHub）、`xox*`（Slack）、PEM 私钥头、数据库 URI 内嵌密码
* **弱特征**：`api_key=` / `token=` / `secret=` 等赋值语句，配合**熵值判定**
* **误报抑制**：占位符关键词、字符多样性、重复段检测；
  文档里的 `your_key_here`、`sk-xxxxxxxx` 不会误报

### 误报处理

若某文件确实含"看起来像密钥的示例"，将其登记到 `.secretsignore`（每行一条路径正则），
而不是放宽全局规则。

### 真的泄漏了怎么办

**顺序很重要**：

1. **先吊销并重建密钥**（在对应平台控制台操作）。密钥一经公开即视为已失效，
   这是唯一有效的止损手段；
2. 从文件中移除，改用环境变量（`.env`，并确认已被 `.gitignore` 忽略）；
3. 仅在需要让对方无法读取历史时才重写 git 历史（`git filter-repo` + 强推），
   **且必须在轮换之后**。

> 只删除文件再提交是**无效**的——密钥仍留在历史对象中，任何人可通过
> `git log` 检出。这正是 `--history` 模式存在的原因。

## 接口示例

```powershell
'{"message":"403 权限不足怎么解决？","thread_id":"demo-1"}' |
  Set-Content "$env:TEMP\b.json" -Encoding utf8
curl.exe -N -s -X POST "http://127.0.0.1:8000/api/chat/stream" `
  -H "Content-Type: application/json" -d "@$env:TEMP\b.json"
```

SSE 事件协议：

```
data: {"type":"token","content":"..."}
data: {"type":"tool_start","tool":"query_knowledge_base","args":{...}}
data: {"type":"tool_end","tool":"query_knowledge_base","output":"..."}
data: {"type":"error","message":"..."}
data: {"type":"done","thread_id":"..."}
```
