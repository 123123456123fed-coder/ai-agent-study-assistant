# AI Research Assistant Pro

## 项目名称

AI Research Assistant Pro

## 项目介绍

AI Research Assistant Pro 是一个面向论文阅读、科研学习与比赛答辩的多模态科研智能 Agent 系统。

系统使用 Streamlit 构建交互界面，结合 RAG、FAISS、论文结构解析工具和大模型，帮助用户对科研论文进行问答、总结、图表分析、公式分析和结构统计。

## 功能

- 论文问答（RAG）
- Agent任务拆解
- 图表与公式分析
- 作者/元信息提取
- 字数与结构统计
- 多轮对话
- 多工具融合回答（Tool-Augmented LLM）

## 项目结构

```text
app.py
agent/
rag/
tools/
llm/
utils/
data/
requirements.txt
README.md
```

## 本地运行

```bash
streamlit run app.py
```

## GitHub部署

1. push to GitHub
2. open https://streamlit.io/cloud
3. select repo
4. deploy app.py

## Streamlit Cloud 配置

项目默认使用通义千问 DashScope。部署到 Streamlit Cloud 时，请在 App Secrets 中配置：

```toml
DASHSCOPE_API_KEY = "your_dashscope_api_key_here"
```

本地运行时，也可以通过环境变量配置：

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

## 部署兼容性

- 不依赖本地绝对路径
- 不使用 localhost 或本地服务器绑定
- 所有项目路径均基于 `Path(__file__).parent`
- 使用 `faiss-cpu`，不依赖 GPU
- 无 GUI 依赖
- embedding 模型延迟加载，并通过 `st.cache_resource` 缓存
- 上传 PDF 后动态构建 FAISS 索引，`search(query)` 使用真实向量检索
