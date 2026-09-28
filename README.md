# GPT Researcher · 阿里云百炼 MaaS 适配版

> 本项目基于开源项目 [GPT Researcher](https://github.com/assafelovic/gpt-researcher)（Apache-2.0）二次开发。
> 上游原始 README 保留在 [README.upstream.md](README.upstream.md)。
>
> **请注意**：仓库中的 `deep_agents/` 深度研究基准（RACE / FACT / SimpleQA）及其全部评测数据
> **为上游团队发布的工作，本人未参与**，保留在此仅为说明基线水平。详见文末[声明](#声明哪些不是我做的)。

---

## 背景

原版 GPT Researcher 默认面向公共 OpenAI / DashScope 端点。在本项目使用的
**阿里云百炼 MaaS 专属工作区**上，它无法直接运行；而且即便跑通，产出的报告
**也没有任何机制可以核查引用是否真实**。

因此这里做了两件事：

1. **接入** —— 让整条 LLM + Embedding 链路指向私有 MaaS 工作区
2. **可核查** —— 补一套离线的引用核查工具，量化报告的可信度

---

## 我做了什么

### 1. 接入改造：统一走 `OPENAI_BASE_URL`

原项目内置的 `dashscope` 提供商把端点地址**写死**为公共域名：

```python
# llm_provider/generic/base.py
llm = ChatOpenAI(openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1', ...)
```

专属工作区必须走自己的域名，用这个分支会连到错误的服务。
因此改为让**所有 OpenAI 兼容的调用点统一读取 `OPENAI_BASE_URL`**：

| 文件 | 改动 |
|---|---|
| `gpt_researcher/memory/embeddings.py` | `openai` 分支注入 `openai_api_base`；关闭 `check_embedding_ctx_length` |
| `gpt_researcher/llm_provider/generic/base.py` | `openai` 分支注入 `openai_api_base`；超时与重试预算从环境变量注入 |

### 2. 修复嵌入接口批量上限（核心 Bug）

**症状**：报告会莫名其妙少掉一整节，但任务**不报错**。

**根因**：DashScope MaaS 的 embedding 接口对单次请求的输入条数有硬上限，超出即返回：

```
400 InternalError.Algo.InvalidParameter: Value error,
batch size is invalid, it should not be larger than 20.: input.contents
```

这个异常在 `gpt_researcher/skills/researcher.py` 的**子查询层被 `except Exception` 兜住**，
只写一行日志、不中断整个 run。所以它表现为"报告变薄"，而不是崩溃——
错误不在结果里，只在日志里。另外错误信息结尾的 `input.contents` 具有误导性，
容易让人以为是**某条内容格式有问题**，而不是**条数超限**。

**修复**（`gpt_researcher/memory/embeddings.py`）：新增 `BatchedOpenAIEmbeddings`，
按 `OPENAI_EMBEDDING_BATCH_SIZE` 分片发送。关键点是批量大小**在调用时读取**、
而非类定义时：

```python
def _batch_size(self) -> int:
    """Resolve the batch size at call time.

    Reading the env var here rather than at class-definition time keeps it
    working for entrypoints that import this module before ``load_dotenv()``
    (e.g. cli.py, whose top-level imports run first).
    """
    return int(os.environ.get("OPENAI_EMBEDDING_BATCH_SIZE", self._MAX_BATCH))
```

因为在类定义时读取，`cli.py` 的顶层 import 早于 `load_dotenv()` 执行，
环境变量此时尚未加载，会**静默退回默认值 20**——恰好卡在上限，看起来能用，
但配置实际被忽略。`embed_documents` 与异步的 `aembed_documents` 都做了分片。

**验证**（`logs/app.log`，2026-08-12）：

| 时间 | 嵌入请求 | 结果 |
|---|---|---|
| 22:34 | 2 次 | **2 × HTTP 400，2 个子查询失败** |
| 22:57 起（修复后） | 全部 | **全部 200 OK，零失败** |

### 3. PDF 导出

WeasyPrint 依赖 GTK 原生库，Windows 下缺失时报 `libgobject error 0x7e`。
复用本机 Tesseract 安装目录自带的 GTK DLL，将其加入 DLL 搜索路径后导出正常。

---

## 验证：报告引用核查工具

`eval_fact.py` —— 一个离线的 mini-FACT 实现，用来回答"这份报告里的引用靠不靠得住"。

### 方法

三步，**全程不调用大模型**，纯字符串比对，无需 API Key，结果可复现：

1. 从报告正文提取「断言 + 它引用的链接」
2. 重新抓取被引页面（落本地缓存）
3. 检查页面中是否真的存在该断言里的**数字**与**来源名**

比对时处理了等价形式，避免把正确的引用误判为错误：

- 繁体 / 简体（`環境及生態局` ≡ `环境及生态局`）
- 机构简称（`工信部` ⊂ `工业和信息化部`）
- 跨语言量级（`15 million` ≡ `1500万`）

### 结果

| 指标 | 数值 |
|---|---|
| 报告 / 引用 / 唯一 URL | 6 份 / 121 条 / 48 个 |
| 被引页面抓取成功率 | **48 / 48 = 100%** |
| **中文报告引用可核实率** | **88 / 94 = 93.6%** |
| 整体（含 1 份英文报告） | 94 / 121 = 77.7% |
| 跨语言、无法判定（单列，不计入通过） | 21 条 |
| 页面中查不到对应数字 | 6 条 |

复现：

```bash
python eval_fact.py --judge det
```

### 已知局限（重要）

- 这是**代理指标**：页面可以用不同措辞支持同一断言，字符串匹配会漏判
- 数字"查不到"**只说明被引页面没写**，**不等于**该数字是错的
  （例：报告写"续航门槛从 43 公里提升"，三个被引页面都只写"不低于 100 公里"——
   43 公里是现行标准的真实门槛，但**没有任何被引来源支持它**）
- 英文报告用英文机构名指代中文网页（`Xinhua Net` → `news.cn`），
  这类共 21 条机器判不了，**单列列出，既不算通过也不算失败**
- 判定不使用 LLM，因此不识别同义改写
- 被引页面的缓存不随仓库分发（`eval_fact_cache/` 已 gitignore），脚本会按需重新抓取

---

## 声明：哪些不是我做的

- **框架主体**：本项目是 [GPT Researcher](https://github.com/assafelovic/gpt-researcher)
  的二次开发。planner / executor / publisher 流水线、多智能体图、检索器体系、
  前端等**均为上游实现**。
- **`deep_agents/` 深度研究基准**：RACE、FACT、SimpleQA 等评测方法及其**全部
  数据与结果**为上游团队发布的工作（见 `deep_agents/SHARE_SUMMARY.md`），
  本人未参与。
- **`evals/`** 为上游自带的评测模块。

---

## 快速开始

### 环境

- Python 3.11
- 一个 OpenAI 兼容的 LLM 端点（本项目使用阿里云百炼 MaaS 专属工作区）

### 配置

```bash
cp .env.example .env
```

需要填写的关键项：

```dotenv
# 端点（必须是专属工作区域名，不是公共 dashscope.aliyuncs.com）
OPENAI_BASE_URL=https://<your-workspace>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=<your-key>

# LLM 与 Embedding
SMART_LLM=openai:<your-chat-model>
FAST_LLM=openai:<your-chat-model>
EMBEDDING=openai:<your-embedding-model>

# 嵌入批量上限（见上文 Bug 说明）
OPENAI_EMBEDDING_BATCH_SIZE=10

# 检索器
RETRIEVER=tavily
TAVILY_API_KEY=<your-key>
```

> LLM 必须用 `openai` 提供商：内置的 `dashscope` 分支把地址写死成了公共域名，
> 在专属工作区上会连到错误的服务。

### 启动

```bash
pip install -r requirements.txt
python -m uvicorn backend.server.app:app --host 127.0.0.1 --port 8010
```

浏览器打开 `http://localhost:8010/`。

也可以用上游的 `docker-compose.yml`（默认端口 8000）：

```bash
docker compose up --build
```

### 引用核查

```bash
python eval_fact.py --judge det     # 确定性判定，无需 API Key
```

结果写入 `eval_fact_results.json`。

---

## 目录结构

```
gpt_researcher/          核心库：研究员、检索、上下文压缩、LLM provider、memory
backend/                 后端服务与报告类型（server/、report_type/）
frontend/                前端（Next.js）
multi_agents/            多智能体（LangGraph）路径
deep_agents/             上游的深度研究基准（非本人工作）
evals/                   上游自带的评测模块
outputs/                 生成的报告（gitignore）
eval_fact.py             引用核查工具（本项目新增）
probe_*.py               若干针对性的诊断脚本（本项目新增）
README.upstream.md       上游原始 README
```

---

## 许可证

本项目遵循上游的 **Apache-2.0** 许可证，见 [LICENSE](LICENSE)。
上游项目版权归 [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher) 及其贡献者所有。
