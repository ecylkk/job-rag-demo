# Windows 快速运行脚本（假设已在项目目录下）
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_data.py
python build_vectorstore.py
python rag_job_matcher.py
