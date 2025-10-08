#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Job Matcher - 完整版本（官方 OpenAI API 调用 + 本地 HuggingFace embeddings + Chroma）
说明:
  - 请在终端设置 OPENAI_API_KEY（推荐）: e.g.
      PowerShell: $env:OPENAI_API_KEY="sk-..."
      mac/linux: export OPENAI_API_KEY="sk-..."
  - 若使用代理或 openrouter，请同时设置 OPENAI_API_BASE。
  - 需要本地 embeddings 相关依赖： sentence-transformers transformers huggingface-hub
  - 先确保已构建好 chroma_store（prepare_data.py / build_vectorstore.py）
用法:
  python rag_job_matcher.py --n 10 --model gpt-3.5-turbo --topk 3
"""

import os
import sys
import json
import random
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any

from rich.console import Console
from rich.table import Table

console = Console()

# ---------------- default config ----------------
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_CHROMA_DIR = Path("chroma_store")
DEFAULT_COLLECTION = "job-postings"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_DATA_DIR.mkdir(exist_ok=True)
# -------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # optional

# ---------------- helpers: candidate gen ----------------
def generate_candidates(n: int = 5) -> List[Dict[str, Any]]:
    """随机/伪造一批候选人（用于测试）"""
    SKILLS = [
        "Python", "Pandas", "SQL", "Tableau", "PowerBI", "A/B Test",
        "机器学习", "NLP", "Spark", "Docker", "Kubernetes", "Java",
        "Golang", "React", "数据可视化", "统计分析", "Excel"
    ]
    CITIES = ["上海", "北京", "深圳", "广州", "杭州", "成都", "南京", "苏州"]

    # 尝试用 Faker 生成中文名字，fallback 随机组合
    names = []
    try:
        from faker import Faker
        fake = Faker("zh_CN")
        for _ in range(n):
            names.append(fake.name())
    except Exception:
        surnames = ["张", "王", "李", "赵", "刘", "陈", "杨", "黄", "周", "吴"]
        given = ["雅", "琳", "强", "明", "婷", "磊", "超", "杰", "华", "斌", "燕", "轩", "浩"]
        for _ in range(n):
            names.append(random.choice(surnames) + random.choice(given))

    objectives = [
        "加入互联网/零售行业数据团队，关注增长分析与数据驱动决策",
        "成为产品数据分析师，提升产品转化率和用户留存",
        "从事数据工程/后端工作，优化数据流水线与性能",
        "在金融数据团队从事风控/特征工程工作",
    ]

    candidates = []
    for i in range(n):
        cand = {
            "name": names[i],
            "city_preference": random.choice(CITIES),
            "years_experience": random.choice([1, 2, 3, 4, 5, 6]),
            "skills": random.sample(SKILLS, k=random.randint(3, 6)),
            "objective": random.choice(objectives),
        }
        candidates.append(cand)
    return candidates

def format_profile(candidate: Dict[str, Any]) -> str:
    skills = "、".join(candidate.get("skills", []))
    profile = f"""姓名：{candidate.get('name')}
目标城市：{candidate.get('city_preference')}优先
工作经验：{candidate.get('years_experience')} 年
技能标签：{skills}
职业目标：{candidate.get('objective')}
"""
    return profile.strip()

# ---------------- embeddings + chroma loader ----------------
def load_hf_embeddings(local_model_name: str = DEFAULT_EMBEDDING_MODEL):
    """兼容不同版本的 langchain 导入 HuggingFaceEmbeddings"""
    console.print(f"[green]使用本地 embeddings（用于查询）：{local_model_name}[/green]")
    try:
        from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings  # type: ignore
    except Exception:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
        except Exception as e:
            console.print("[red]无法导入 HuggingFaceEmbeddings，请安装 sentence-transformers 与 langchain_community 或 langchain。[/red]")
            raise e
    return HuggingFaceEmbeddings(model_name=local_model_name)

def load_chroma_vectorstore(embedding_function, chroma_dir: Path = DEFAULT_CHROMA_DIR, collection_name: str = DEFAULT_COLLECTION):
    """加载已持久化的 Chroma 向量库，并传入 embedding_function（查询时需要）"""
    try:
        # 新版建议使用 langchain_community.vectorstores.Chroma
        from langchain_community.vectorstores import Chroma  # type: ignore
    except Exception:
        try:
            from langchain.vectorstores import Chroma  # fallback
        except Exception as e:
            console.print("[red]无法导入 Chroma，请安装 chromadb / langchain_community。[/red]")
            raise e

    vectorstore = Chroma(
        persist_directory=str(chroma_dir),
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    return vectorstore

# ---------------- retrieve top-k ----------------
def retrieve_topk(query: str, k: int = 3, embedding_model: str = DEFAULT_EMBEDDING_MODEL, chroma_dir: Path = DEFAULT_CHROMA_DIR, collection_name: str = DEFAULT_COLLECTION):
    """使用本地 embeddings 从 chroma_store 检索 top-k 文档"""
    try:
        hf = load_hf_embeddings(local_model_name=embedding_model)
        vectorstore = load_chroma_vectorstore(embedding_function=hf, chroma_dir=chroma_dir, collection_name=collection_name)
    except Exception as e:
        console.print("[red]加载向量库或 embeddings 失败，请检查依赖与 chroma_store 是否存在。[/red]")
        raise

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # 兼容新旧接口
    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        else:
            # fallback to similarity_search on vectorstore (embedding_function must be present)
            docs = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        # 尝试直接 similarity_search（embedding_function 必须在 Chroma 中）
        try:
            docs = vectorstore.similarity_search(query, k=k)
        except Exception as e2:
            console.print("[red]检索失败（已尝试多种方法）：[/red]")
            traceback.print_exc()
            raise RuntimeError(f"检索失败: {e} / {e2}") from e2
    return docs

# ---------------- pretty print sources ----------------
def pretty_print_sources(sources: List[Any]):
    table = Table(title="🔎 命中岗位文档（Top K）", show_lines=True)
    table.add_column("岗位 ID", style="cyan", no_wrap=True)
    table.add_column("岗位名称", style="magenta")
    table.add_column("公司", style="green")
    table.add_column("城市", style="yellow")

    for d in sources:
        meta = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {})
        job_id = meta.get("job_id") or meta.get("id") or "-"
        table.add_row(str(job_id), meta.get("title", "-"), meta.get("company", "-"), meta.get("location", meta.get("city", "-")))
    console.print(table)

# ---------------- robust OpenAI chat call ----------------
def call_openai_chat(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = 800) -> str:
    """
    更鲁棒的 OpenAI chat 调用：先尝试老接口 openai.ChatCompletion.create，再尝试新版 openai.OpenAI 客户端。
    会尽可能解析并返回文本内容；若都失败会抛出异常并尽量打印 raw response 供排查。
    """
    raw_debug = None
    # 1) 旧风格 openai 包的 ChatCompletion API（很多项目里仍可工作）
    try:
        import openai
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
        if OPENAI_API_BASE:
            openai.api_base = OPENAI_API_BASE

        resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        raw_debug = resp
        # 解析最常见结构
        if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
            choice = resp["choices"][0]
            # 现代 ChatCompletion 返回 choices[0]["message"]["content"]
            if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                return choice["message"]["content"]
            # older style might have 'text'
            if isinstance(choice, dict) and "text" in choice:
                return choice["text"]
        # object style
        if hasattr(resp, "choices"):
            choices = getattr(resp, "choices")
            if choices and len(choices) > 0:
                c0 = choices[0]
                # c0 may have .message.content
                msg = getattr(c0, "message", None)
                if msg and getattr(msg, "content", None):
                    return msg.content
                text = getattr(c0, "text", None)
                if text:
                    return text
        # 若未解析则继续到新版 client
    except Exception as e_old:
        console.print(f"[grey]debug: openai.ChatCompletion.create failed or returned odd structure: {e_old}[/grey]")

    # 2) 新 openai client (openai>=1.0) 的用法
    try:
        from openai import OpenAI
        client = None
        try:
            client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
            # 您也可以通过环境变量 OPENAI_API_BASE 控制 base url（OpenRouter 等）
        except Exception:
            client = OpenAI()

        resp2 = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        # 尝试解析 resp2（可能是 OpenAI SDK 的对象）
        try:
            d = resp2 if isinstance(resp2, dict) else (resp2.to_dict() if hasattr(resp2, "to_dict") else dict(resp2.__dict__))
        except Exception:
            d = str(resp2)

        # 尝试打印部分 raw response 便于调试（不会太长）
        try:
            console.print("[grey]DEBUG raw OpenAI.chat.completions response (truncated):[/grey]")
            console.print(json.dumps(d, default=str, ensure_ascii=False)[:2000])
        except Exception:
            console.print("[grey]DEBUG raw OpenAI response (non-json) (truncated).[/grey]")

        # 解析 d
        if isinstance(d, dict) and "choices" in d and d["choices"]:
            ch = d["choices"][0]
            if isinstance(ch, dict):
                if "message" in ch and isinstance(ch["message"], dict) and "content" in ch["message"]:
                    return ch["message"]["content"]
                if "text" in ch:
                    return ch["text"]
        # attributes fallback
        if hasattr(resp2, "choices") and resp2.choices:
            c0 = resp2.choices[0]
            if hasattr(c0, "message") and getattr(c0.message, "content", None):
                return c0.message.content
            if hasattr(c0, "text") and getattr(c0, "text", None):
                return c0.text

        raise RuntimeError("无法解析 OpenAI 客户端返回的响应结构（已打印 raw response）")
    except Exception as e_new:
        # 若两种方法都失败，给出提示信息（通常是 key/endpoint 问题）
        raise RuntimeError(f"OpenAI 调用失败（请确认 OPENAI_API_KEY/OPENAI_API_BASE 是否正确，或 provider 状态）：{e_new}") from e_new

# ---------------- build prompt from docs ----------------
def build_prompt_from_docs(candidate_profile: str, docs: List[Any]) -> List[Dict[str, str]]:
    ctx_parts = []
    for d in docs:
        content = getattr(d, "page_content", "") or (d.get("page_content") if isinstance(d, dict) else "")
        meta = getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {})
        header = f"岗位ID: {meta.get('job_id','-')} | 标题: {meta.get('title','-')} | 公司: {meta.get('company','-')} | 城市: {meta.get('location','-')}"
        ctx_parts.append(header + "\n" + (content.strip() if content else ""))

    context_text = "\n\n---\n\n".join(ctx_parts)

    prompt_template = """你是一位资深 HR 顾问，擅长根据候选人的技能背景推荐匹配岗位。
结合提供的【候选人信息】和检索到的【岗位信息】，请输出：
1. 匹配度评分（百分制，简短理由）
2. 推荐岗位列表（每个岗位包含：岗位名称 / 公司 / 城市 / 推荐理由）
3. 针对候选人的补充建议（如需加强的技能、项目经历等）

输出格式使用 Markdown。

检索到的岗位信息：
{context}

候选人信息：
{candidate}

请给出推荐结果（简洁、条理清晰）。"""

    user_prompt = prompt_template.format(context=context_text, candidate=candidate_profile.strip())

    messages = [
        {"role": "system", "content": "你是一个友好的助理，回答时要简洁并给出清晰建议。"},
        {"role": "user", "content": user_prompt},
    ]
    return messages

# ---------------- main batch runner ----------------
def run_batch(n_candidates: int = 5, model: str = DEFAULT_MODEL, topk: int = 3, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    # 生成候选人并保存
    candidates = generate_candidates(n_candidates)
    cand_file = DEFAULT_DATA_DIR / "candidates.json"
    with open(cand_file, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    console.print(f"[green]已生成候选人并保存：{cand_file}[/green]")

    results = []

    for idx, cand in enumerate(candidates, start=1):
        console.rule(f"[bold blue]候选人 {idx}/{len(candidates)}：{cand['name']}")
        profile_text = format_profile(cand)
        console.print(profile_text)

        console.rule("[bold cyan]检索 Top-K 岗位")
        try:
            docs = retrieve_topk(profile_text, k=topk, embedding_model=embedding_model)
            pretty_print_sources(docs)
        except Exception as e:
            console.print("[red]检索失败：跳过该候选人并记录错误。[/red]")
            results.append({"candidate": cand, "error": str(e)})
            continue

        console.rule("[bold magenta]调用模型生成推荐（OpenAI Chat）")
        try:
            messages = build_prompt_from_docs(profile_text, docs)
            reply = call_openai_chat(messages=messages, model=model, temperature=DEFAULT_TEMPERATURE, max_tokens=800)
            console.print("\n[bold green]模型回复：[/bold green]\n")
            console.print(reply)
            results.append({
                "candidate": cand,
                "profile_text": profile_text,
                "rag_answer": reply,
                "sources_meta": [ (getattr(d, "metadata", None) or (d.get("metadata") if isinstance(d, dict) else {})) for d in docs ]
            })
        except Exception as e:
            console.print("[red]调用模型失败：记录错误并继续。[/red]")
            traceback.print_exc()
            results.append({"candidate": cand, "error": str(e)})

    out_file = DEFAULT_DATA_DIR / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    console.print(f"[green]全部完成。结果已保存：{out_file.resolve()}[/green]")

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="生成候选人人数")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI 模型名（gpt-3.5-turbo/gpt-4 等）")
    p.add_argument("--topk", type=int, default=3, help="检索 Top-K 文档")
    p.add_argument("--embedding", type=str, default=DEFAULT_EMBEDDING_MODEL, help="本地 embedding 模型")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not OPENAI_API_KEY:
        console.print("[yellow]警告：当前未检测到 OPENAI_API_KEY 环境变量。若要使用官方 OpenAI，请先在终端导出或在 .env 中设置。[/yellow]")
    run_batch(n_candidates=args.n, model=args.model, topk=args.topk, embedding_model=args.embedding)
