# debug_embeddings_test.py
import os
import json
import requests

API_KEY = os.getenv("OPENAI_API_KEY")
BASE = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

print("Using BASE:", BASE)
print("Using MODEL:", MODEL)

url = BASE.rstrip("/") + "/embeddings"
payload = {"model": MODEL, "input": "测试一下 embeddings 是否可用"}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

resp = requests.post(url, headers=headers, data=json.dumps(payload))
print("HTTP", resp.status_code)
print("RESPONSE TEXT:")
print(resp.text[:2000])  # 打印前 2000 字符，足够查看错误
