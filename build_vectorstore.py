# build_vectorstore.py (稳健版：显式 openai 配置 + 远程失败回退到本地 embeddings)
import json
import os
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = Path("chroma_store")
DOC_PATH = Path("data/job_docs.json")

# 尝试把 OPENAI key/base 显式设置到 openai 客户端，防止 env 名不一致
def configure_openai_client():
    try:
        import openai
    except Exception:
        # 如果没有 openai 包，这里不终止，langchain_openai 会尝试导入时报错
        return

    key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")

    # 不同 openai 版本 API 名可能不同，尽量兼容设置
    try:
        if key:
            # 旧版 openai 用这个属性
            openai.api_key = key
        if base:
            openai.api_base = base
    except Exception:
        # new-style OpenAI client library may use OpenAI(api_key=..., base_url=...)
        try:
            # 如果 openai.OpenAI 存在，构造客户端实例（部分库版本支持）
            _ = openai.OpenAI(api_key=key, base_url=base)
        except Exception:
            pass

def load_documents(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw

def build_with_remote(texts, metadatas):
    """
    使用远程 OpenAI/OpenRouter embeddings（langchain_openai.OpenAIEmbeddings）
    如果成功返回 True，否则打印异常并返回 False
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma

        print("尝试使用远程 embeddings（OpenAI/OpenRouter）：", EMBEDDING_MODEL)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=str(CHROMA_DIR),
            collection_name="job-postings"
        )
        print(f"✅ 远程向量库已构建，数据量：{len(texts)}，存储目录：{CHROMA_DIR.resolve()}")
        return True
    except Exception as e:
        print("⚠️ 远程 embeddings 调用失败，异常：")
        traceback.print_exc()
        return False

def build_with_local(texts, metadatas):
    """
    回退：使用本地 HuggingFaceEmbeddings (sentence-transformers)
    需要安装：sentence-transformers transformers huggingface-hub
    """
    try:
        # 首选 langchain 的 HuggingFaceEmbeddings（若 langchain 版本支持）
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except Exception:
            # 旧版 langchain 可能放在不同位置
            from langchain.embeddings.huggingface import HuggingFaceEmbeddings

        from langchain_community.vectorstores import Chroma

        print("回退到本地 HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)")
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=emb,
            metadatas=metadatas,
            persist_directory=str(CHROMA_DIR),
            collection_name="job-postings"
        )
        print(f"✅ 本地向量库已构建，数据量：{len(texts)}，存储目录：{CHROMA_DIR.resolve()}")
        return True
    except Exception:
        print("❌ 本地 embeddings 构建失败（请确保已安装 sentence-transformers 等依赖）:")
        traceback.print_exc()
        return False

def main():
    configure_openai_client()

    documents = load_documents(DOC_PATH)
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    # 先尝试远程（OpenRouter / DeepSeek via OpenAI-compatible API）
    ok = build_with_remote(texts, metadatas)

    # 远程失败则回退到本地
    if not ok:
        print("开始回退到本地 embeddings...")
        ok = build_with_local(texts, metadatas)

    if not ok:
        print("最终构建失败。请检查以上错误与依赖，并把错误栈贴给我以便继续排查。")

if __name__ == "__main__":
    main()
