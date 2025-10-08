# job-rag-demo — LangChain + RAG 求职匹配 (DeepSeek via OpenRouter)

本项目演示如何从零部署一个基于 LangChain + RAG 的求职匹配示例，并使用 DeepSeek (通过 OpenRouter) 作为 LLM。

## 目录结构
```
job-rag-demo/
├── chroma_store/             # 向量数据库存储（运行后生成）
├── data/
│   ├── jobs.json            # 原始岗位数据
│   └── job_docs.json        # prepare_data.py 生成的文档
├── .env                     # 环境变量（包含 API KEY，注意安全）
├── requirements.txt         # Python 依赖
├── prepare_data.py          # 数据准备脚本
├── build_vectorstore.py     # 向量库构建脚本
└── rag_job_matcher.py       # 主程序：RAG 求职匹配
```

## 快速开始（Windows + Chocolatey）
请以管理员身份打开 PowerShell 执行下面步骤。若已安装请跳过对应步骤。

1. **安装 Chocolatey（如果未安装）**
```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

2. **安装 Python 3.10+ 与 Git（可选）**
```powershell
choco install python310 -y
choco install git -y
```

3. **创建项目目录并创建虚拟环境**
```powershell
mkdir job-rag-demo
cd job-rag-demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
如果遇到执行策略错误：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

4. **安装 Python 依赖**
```powershell
pip install -r requirements.txt
```

5. **确认 .env 已正确填写**（本仓库内已包含示例 .env，注意不要在公共仓库泄露 API Key）

6. **依次运行脚本**
```powershell
python prepare_data.py
python build_vectorstore.py
python rag_job_matcher.py
```

## 注意事项
- 本项目示例使用用户提供的 API Key（保存在 .env）。请在生产环境使用更安全的密钥管理方式（Key Vault / 环境变量 / CI secret）。
- 如果遇到 embedding/LLM 调用失败，请检查网络（是否能访问 OpenRouter）、模型名是否支持、以及 .env 中变量是否正确。
- Chroma 会在第一次运行 `build_vectorstore.py` 时创建 `chroma_store/` 目录并持久化数据。

## 扩展建议
- 将 `rag_job_matcher.py` 改写为 Web 服务（FastAPI/Flask）或前端 (Streamlit/Gradio)。
- 添加简历上传与 PDF 解析，批量匹配候选人与岗位。
- 增加缓存、异步调用与更严格的错误处理。

-- 生成于本次会话。
