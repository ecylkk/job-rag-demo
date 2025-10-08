import json
from pathlib import Path
from typing import List
from langchain.schema import Document

DATA_PATH = Path("data/jobs.json")
OUTPUT_PATH = Path("data/job_docs.json")

def load_jobs(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def job_to_document(job: dict) -> Document:
    metadata = {
        "job_id": job["id"],
        "title": job["title"],
        "company": job["company"],
        "location": job["location"]
    }
    page_content = (
        f"【岗位】{job['title']}\n"
        f"【公司】{job['company']}\n"
        f"【城市】{job['location']}\n"
        f"【岗位要求】\n- " + "\n- ".join(job["requirements"]) + "\n"
        f"【岗位职责】\n- " + "\n- ".join(job["responsibilities"])
    )
    return Document(page_content=page_content, metadata=metadata)

def save_documents(docs: List[Document], path: Path):
    serialized = [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]
    path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    jobs = load_jobs(DATA_PATH)
    documents = [job_to_document(job) for job in jobs]
    save_documents(documents, OUTPUT_PATH)
    print(f"✅ 已生成 {len(documents)} 条岗位文档 -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
