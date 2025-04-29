# conversational_product_search_agent.py
"""
Interactive Conversational Product‑Search Agent (BM25s + GPT‑4.1‑mini)
====================================================================
Search Amazon *Sports & Outdoors* metadata in a 5‑round dialogue that
narrows 64 → 4 products. Built on **bm25s** for fast lexical retrieval
and LangChain for LLM prompting.

Pipeline
--------
1. User gives an **initial query**.
2. **BM25s** retrieves Top‑64 documents.
3. **GPT‑4.1‑mini** summarises them & asks one discriminating question.
4. User answers → refine query → shrink pool: 64→32→16→8→4.
5. Show final 4 products; user selects one.

Quick start
-----------
```bash
pip install datasets bm25s[core] langchain openai tiktoken tqdm PyStemmer
export OPENAI_API_KEY="sk‑…"
python conversational_product_search_agent.py  # first run builds index
```

Index facts
-----------
* **Dataset** : Amazon‑Reviews‑2023 → *raw_meta_Sports_and_Outdoors*
* **Doc id**  : `parent_asin`
* **Doc body**: `title + feature[] + description`
* **Cache dir**: `sports_bm25s_index/` (retriever + tokenizer + corpus)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv


import bm25s
from datasets import load_dataset
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from Stemmer import Stemmer
from tqdm import tqdm
import re

load_dotenv()
# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────
MODEL_NAME = "gpt-4.1-mini"
TEMPERATURE = 0.2
INDEX_DIR = Path("toys_bm25s_index")
TOP_KS = [64, 32, 16, 8, 4]        # pool sizes per round
MAX_PRODUCTS = None                # None → full split; set small for demo

FORBIDDEN_PATTERNS = [
    r"\b\d{3}-\d{3,4}-\d{4}\b",      # 전화번호 패턴
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # 이메일
    r"(욕설1|욕설2|비속어)",          # 욕설/비속어 (실제 리스트로 대체)
]

def is_safe_user_input(text):
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return False
    return True

def is_safe_llm_output(text):
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return False
    return True

# ──────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────

def _iter_products(limit: int | None = None):
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Toys_and_Games",
        split="full",
        trust_remote_code=True,
    )
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        pid = row["parent_asin"]
        title = row.get("title") or ""
        features = " ".join(row.get("features", [])) if row.get("features") else ""
        desc = row.get("description") or ""
        text = " ".join(filter(None, [str(title), str(features), str(desc)]))
        if text:
            yield {"id": pid, "text": text}

# ──────────────────────────────────────────────────
# Index build / load
# ──────────────────────────────────────────────────

def _build_or_load_index(limit: int | None = None):
    """Load cached BM25s index or build it if absent."""
    stemmer = Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer, stopwords='en')
    if INDEX_DIR.exists():
        print("[+] Loading cached BM25s index…")
        retriever = bm25s.BM25.load(INDEX_DIR, mmap=True, load_corpus=True)
        tokenizer.load_vocab(INDEX_DIR)
        tokenizer.load_stopwords(INDEX_DIR)
        return retriever.corpus, tokenizer, retriever

    print("[+] Building BM25s index (first run — please wait)…")
    corpus = list(_iter_products(limit))
    texts = [d["text"] for d in corpus]

    tokens = tokenizer.tokenize(texts)
    retriever = bm25s.BM25(corpus=corpus, backend="numba")
    retriever.index(tokens)
    retriever.vocab_dict = {str(k): v for k, v in retriever.vocab_dict.items()}

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    retriever.save(INDEX_DIR, corpus=corpus)
    tokenizer.save_vocab(INDEX_DIR)
    tokenizer.save_stopwords(INDEX_DIR)
    print(f"[✓] Saved index ({len(corpus):,} docs) → {INDEX_DIR}")
    return corpus, tokenizer, retriever

# ──────────────────────────────────────────────────
# Retrieval helper
# ──────────────────────────────────────────────────

def bm25_search(query: str, idx_tuple: Tuple[List[Dict[str, Any]], Any, Any], k: int) -> List[Tuple[str, str, float]]:
    corpus, tok, ret = idx_tuple  # Ensure idx_tuple is unpacked correctly
    q_tokens = tok.tokenize([query], update_vocab=False)
    docs_mat, scores_mat = ret.retrieve(q_tokens, k=k)
    docs, scores = docs_mat[0], scores_mat[0]
    return [(d["id"], d["text"], float(s)) for d, s in zip(docs, scores)]

# ──────────────────────────────────────────────────
# LLM prompt helper
# ──────────────────────────────────────────────────
QUESTION_PROMPT = PromptTemplate(
    input_variables=["items"],
    template=(
        "You are a helpful product‑search assistant. Without recommending any specific item, "
        "ask **one** short question that best distinguishes among the products below.\n\n"
        "Do NOT ask about sensitive topics (e.g., age, gender, race, religion, health, or personal information). "
        "Ensure your question is neutral, non-offensive, and respects user privacy. "
        "Return **only** the question.\n\n"
        "Products (id · snippet):\n{items}\n\n"
    ),
)

def ask_disambiguation(llm: ChatOpenAI, docs):
    snippets = []
    for pid, text, _ in docs:
        head = " ".join(text.split()[:20]) + (" …" if len(text.split()) > 20 else "")
        snippets.append(f"{pid} · {head}")
    while True:
        prompt = QUESTION_PROMPT.format(items="\n".join(snippets))
        q = llm.invoke(prompt).content.strip()
        if is_safe_llm_output(q):
            return q
        print("[!] Unsafe question generated. Retrying...")

# ──────────────────────────────────────────────────
# Main chat loop
# ──────────────────────────────────────────────────

def interactive_loop():
    idx_tuple = _build_or_load_index(MAX_PRODUCTS)
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE, streaming=True)

    print("안내: 개인정보(이름, 연락처, 이메일 등)는 입력하지 마세요. 본 대화는 제품 추천 목적에만 사용됩니다.")
    print("=========== Conversational Product Search (BM25s) ===========")
    user_query = input("You: ").strip()
    if not is_safe_user_input(user_query):
        print("[!] 부적절한 입력입니다. 다시 시도해주세요.")
        return

    if not user_query or user_query.startswith("/exit"):
        return

    for k in TOP_KS:
        hits = bm25_search(user_query, idx_tuple, k)
        if k == 4:
            break  # final pool ready
        q = ask_disambiguation(llm, hits)
        print(f"Agent: {q}")
        ans = input("You: ").strip()
        if not is_safe_user_input(ans):
            print("[!] 부적절한 입력입니다. 다시 시도해주세요.")
            return
        user_query = f"{user_query} {ans}".strip()

    print("\nHere are the final 4 products — pick 1‑4 (or /exit):")
    for i, (pid, text, _) in enumerate(hits, 1):
        prev = " ".join(text.split()[:40]) + (" …" if len(text.split()) > 40 else "")
        print(f" {i}. [{pid}] {prev}")

    choice = input("Your choice: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= 4:
        sel = hits[int(choice) - 1]
        print("\nYou selected:\nID: {}\nDescription: {}".format(sel[0], sel[1]))
    else:
        print("No valid selection. Bye!")

# ──────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        interactive_loop()
    except KeyboardInterrupt:
        print("\n[Session terminated]")
