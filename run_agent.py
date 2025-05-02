# -*- coding: utf-8 -*-

from __future__ import annotations


import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv


import numpy as np
import faiss
import bm25s
from datasets import load_dataset
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from Stemmer import Stemmer
from tqdm import tqdm

from user_simulator import user_simulator, accumulate_retrieval_result

load_dotenv()
# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────
MODEL_NAME = "gpt-4.1-mini"
TEMPERATURE = 0.2
INDEX_DIR = Path("cellphones_bm25s_index")
# INDEX_DIR = Path("toys_bm25s_index")
# INDEX_DIR = Path("magazines_bm25s_index")
# VEC_DIR = Path("toys_faiss")
VEC_DIR = Path("cellphones_faiss")
TOP_KS = [10, 10, 10, 10, 10]        # pool sizes per round
MAX_PRODUCTS = None                # None → full split; set small for demo
SEM_K_FACTOR = 2                  # retrieve k*factor from each modality
HYBRID_WEIGHT = 0.5               # 0.5 lexical + 0.5 semantic
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
DEVICE = "cuda"

# ──────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────

def _iter_products(limit):
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        # "raw_meta_Toys_and_Games",
        "raw_meta_Cell_Phones_and_Accessories",
        # 'raw_meta_Magazine_Subscriptions',
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

def _build_or_load_bm25_index(limit):
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
    # print(f"[✓] Saved index ({len(corpus):,} docs) → {INDEX_DIR}")
    return corpus, tokenizer, retriever


def _build_or_load_vector_index(corpus: List[Dict[str, str]]):
    """Load or build FAISS index with BGE embeddings."""
    if (VEC_DIR / "index.faiss").exists():
        print("[+] Loading cached FAISS vector index…")
        index = faiss.read_index(str(VEC_DIR / "index.faiss"))
        id_map = json.loads((VEC_DIR / "id_map.json").read_text())
        model = SentenceTransformer(EMBED_MODEL_NAME)
        return index, id_map, model

    print("[+] Building FAISS vector index (first run — please wait)…")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    model.max_seq_length = 512
    texts = [d["text"] for d in corpus]

    # Embed in batches to avoid OOM
    embeddings = []
    for i in tqdm(range(0, len(texts), 256), desc="Embedding"):
        batch_emb = model.encode(texts[i:i+256], show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings).astype('float32')

    # Build FAISS index (inner product on unit vectors == cosine sim)
    # use mps for FAISS
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    # Persist
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VEC_DIR / "index.faiss"))
    (VEC_DIR / "id_map.json").write_text(json.dumps([d["id"] for d in corpus]))
    print(f"[✓] Saved FAISS index ({len(corpus):,} vectors) → {VEC_DIR}")
    return index, [d["id"] for d in corpus], model


# ──────────────────────────────────────────────────
# Retrieval helper
# ──────────────────────────────────────────────────

def bm25_search(query: str, idx_tuple, k: int) -> List[Tuple[str, str, float]]:
    _, tok, ret = idx_tuple
    q_tokens = tok.tokenize([query], update_vocab=False)
    docs_mat, scores_mat = ret.retrieve(q_tokens, k=k)
    docs, scores = docs_mat[0], scores_mat[0]
    return [(d["id"], d["text"], float(s)) for d, s in zip(docs, scores)]


def semantic_search(query: str, vec_tuple, k: int) -> List[Tuple[str, float]]:
    index, id_map, model = vec_tuple
    q_emb = model.encode([query], normalize_embeddings=True)[0].astype('float32')
    scores, idxs = index.search(q_emb[None, :], k)
    return [(id_map[int(i)], float(s)) for i, s in zip(idxs[0], scores[0])]


def hybrid_search(query: str, idx_tuple, vec_tuple, k: int, w: float = HYBRID_WEIGHT):
    # Retrieve from each modality
    bm25_hits = bm25_search(query, idx_tuple, k=k*SEM_K_FACTOR)
    sem_hits = semantic_search(query, vec_tuple, k=k*SEM_K_FACTOR)

    # Build score dicts
    bm25_dict = {pid: s for pid, _, s in bm25_hits}
    sem_dict = {pid: s for pid, s in sem_hits}

    # Normalise scores [0,1] within each modality
    if bm25_dict:
        bm_min, bm_max = min(bm25_dict.values()), max(bm25_dict.values())
    else:
        bm_min = bm_max = 0
    if sem_dict:
        sm_min, sm_max = min(sem_dict.values()), max(sem_dict.values())
    else:
        sm_min = sm_max = 0

    def norm(val, vmin, vmax):
        return 0.0 if vmax == vmin else (val - vmin) / (vmax - vmin)

    # Union of document ids
    docs_all = set(bm25_dict) | set(sem_dict)

    # Compute hybrid score
    scored_docs = []
    for pid in docs_all:
        bm = norm(bm25_dict.get(pid, bm_min), bm_min, bm_max)
        sm = norm(sem_dict.get(pid, sm_min), sm_min, sm_max)
        hybrid = w * bm + (1 - w) * sm
        scored_docs.append((pid, hybrid))

    # Sort by hybrid score desc
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Retrieve full text for top‑k
    corpus, _, _ = idx_tuple
    id_to_text = {d["id"]: d["text"] for d in corpus}
    topk = [(
        pid,
        id_to_text.get(pid, ""),
        score,
    ) for pid, score in scored_docs[:k]]
    return topk



# ──────────────────────────────────────────────────
# LLM prompt helper
# ──────────────────────────────────────────────────
QUESTION_PROMPT = PromptTemplate(
    input_variables=["items", "history"],
    template=(
        "You are a helpful product-search assistant. Without recommending any specific item, "
        "ask **one** concise question that best distinguishes among the products below.\n\n"
        "Products (id · snippet):\n{items}\n\nReturn **only** the consie question."

        "Previously asked questions:\n{history}\n\n"
        "Never repeat similar question asked before."
        "If you think there's no need to ask a new question, return [END]"
    ),
)

def ask_disambiguation(llm: ChatOpenAI, docs, prev_qs: List[str]):
    # build product snippets
    snippets = []
    for pid, text, _ in docs:
        head = " ".join(text.split()[:20]) + (" …" if len(text.split()) > 20 else "")
        snippets.append(f"{pid} · {head}")

    history = "None so far." if not prev_qs else "\n".join(f"- {q}" for q in prev_qs)
    prompt = QUESTION_PROMPT.format(items="\n".join(snippets), history=history)
    print(history)
    return llm.invoke(prompt).content.strip()
# ──────────────────────────────────────────────────
# Main chat loop
# ──────────────────────────────────────────────────

def interactive_loop():
    bm25_idx = _build_or_load_bm25_index(MAX_PRODUCTS)
    vec_idx = _build_or_load_vector_index(bm25_idx[0])
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE, streaming=True)

    print("=========== Conversational Product Search (BM25s) ===========")
    user_query = input("You: ").strip()
    if not user_query or user_query.startswith("/exit"):
        return
    
    prev_questions: List[str] = []


    for k in TOP_KS:
        hits = hybrid_search(user_query, bm25_idx, vec_idx, k)
        # if k == 4:
        #     break  # final pool ready
        q = ask_disambiguation(llm, hits, prev_questions)
        prev_questions.append(q)


        print(f"Agent: {q}")

        if "END" in q:
            hits = hybrid_search(user_query, bm25_idx, vec_idx, 4)
            break
        
        ans = input("You: ").strip()
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
# Evaluation chat loop
# ──────────────────────────────────────────────────

def eval_loop():
    bm25_idx = _build_or_load_bm25_index(MAX_PRODUCTS)
    vec_idx = _build_or_load_vector_index(bm25_idx[0])
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE, streaming=True)

    # Open the test set (jsonl file)
    eval_data_paths = [
        # Path("./sample_data/toy_sample.jsonl"),
        Path("./PSA/sample_data/cellphone_sample.jsonl"),
    ]
    toy_meta = []
    cellphone_meta = []

    for path in eval_data_paths:
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                if path.name == "toy_sample.jsonl":
                    toy_meta.append(data)
                elif path.name == "cellphone_sample.jsonl":
                    cellphone_meta.append(data)

    retrieval_results = []
    reciprocal_ranks = []

    for meta in tqdm(cellphone_meta, desc="Evaluating CellPhone Samples"):
        user_sim = user_simulator(
            meta=meta,
            llm=llm
        )

        user_query = user_sim.initial_ambiguous_query()
        # print(f"You: {user_query}")

        prev_questions = []

        for k in TOP_KS:
            hits = hybrid_search(user_query, bm25_idx, vec_idx, k)
            user_sim.eval_retrieval(hits, k)
            question = ask_disambiguation(llm, hits, prev_questions)
            # print(f"Agent: {question}")
            prev_questions.append(question)
            if question.strip() == "[END]":
                break

            answer = user_sim.answer_clarification_question(question)
            # print(f"You: {answer}")

            user_query = f"{user_query} {answer}".strip()

        final_hits = hybrid_search(user_query, bm25_idx, vec_idx, 10)
        user_sim.eval_retrieval(final_hits, 10)

        r, rr = user_sim.get_result()
        retrieval_results.append(r)
        reciprocal_ranks.append(rr)
    
    lengths, hit_at_k_per_turn, mrr_per_turn = accumulate_retrieval_result(retrieval_results, reciprocal_ranks)

    print("\n\n==================== Evaluation Results ====================")
    for turn_idx, (hit, mrr) in enumerate(zip(hit_at_k_per_turn, mrr_per_turn)):
        print(f"Turn {turn_idx + 1}:")
        print(f"Hit@{10}: {hit:.4f}")
        print(f"MRR@{10}: {mrr:.4f}")


# ──────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        # interactive_loop()
        eval_loop()
    except KeyboardInterrupt:
        print("\n[Session terminated]")
