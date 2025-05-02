from __future__ import annotations


import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
import requests
import base64
import time
import os
from PIL import Image


import numpy np
import faiss
import bm25s
from datasets import load_dataset
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from Stemmer import Stemmer
from tqdm import tqdm
import warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
from datasets import load_dataset
from collections import defaultdict

load_dotenv()

MODEL_NAME = "gpt-4.1-mini"
TEMPERATURE = 0.2
# INDEX_DIR = Path("toys_bm25s_index")
INDEX_DIR = Path("toys_bm25s_index")
VEC_DIR = Path("toys_faiss")
TOP_KS = [20, 20, 20, 4]        # pool sizes per round
MAX_PRODUCTS = None                # None → full split; set small for demo
SEM_K_FACTOR = 2                  # retrieve k*factor from each modality
HYBRID_WEIGHT = 0.5               # 0.5 lexical + 0.5 semantic
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
NOVITA_API_URL = "https://api.novita.ai/v3/async/img2video"
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY", "")  # .env 파일에서 API 키 가져오기

# 이미지-비디오 변환 모델 이름 및 가격 정보
MODEL_NAMES = {
    "svd": "SVD",
    "svd_xt": "SVD-XT"
}

MODEL_PRICING = {
    "svd": 0.0134,
    "svd_xt": 0.024,
}

# 이미지 최대 해상도 설정
MAX_IMAGE_WIDTH = 576
MAX_IMAGE_HEIGHT = 1024


def _iter_products(limit: int | None = None):
    # 1) 메타 정보
    meta_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Toys_and_Games",
        split="full",
        trust_remote_code=True,
    )

    # 2) 리뷰 정보 →  parent_asin ➜ [review strings]
    review_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Toys_and_Games",
        split="full",
        trust_remote_code=True,
    )

    reviews_by_pid: dict[str, list[str]] = defaultdict(list)
    for row in review_ds:
        pid = row["parent_asin"]
        rv_title = row.get("title") or ""
        rv_text  = row.get("text")  or ""
        if rv_title or rv_text:
            reviews_by_pid[pid].append(f"{rv_title} {rv_text}".strip())

    # 3) 메타 + 리뷰를 합쳐서 반환
    for i, row in enumerate(meta_ds):
        if limit and i >= limit:
            break

        pid      = row["parent_asin"]
        title    = row.get("title") or ""
        features = " ".join(row.get("features", [])) if row.get("features") else ""
        desc     = row.get("description") or ""
        rv_blob  = " ".join(reviews_by_pid.get(pid, []))

        text = " ".join(filter(None, [str(title), str(features), str(desc), str(rv_blob)]))
        if text:
            yield {"id": pid, "text": text}


# ──────────────────────────────────────────────────
# Index build / load
# ──────────────────────────────────────────────────

def _build_or_load_bm25_index(limit: int | None = None):
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


def _build_or_load_vector_index(corpus: List[Dict[str, str]]):
    """Load or build FAISS index with BGE embeddings."""
    if (VEC_DIR / "index.faiss").exists():
        print("[+] Loading cached FAISS vector index…")
        index = faiss.read_index(str(VEC_DIR / "index.faiss"))
        id_map = json.loads((VEC_DIR / "id_map.json").read_text())
        model = SentenceTransformer(EMBED_MODEL_NAME)
        return index, id_map, model

    print("[+] Building FAISS vector index (first run — please wait)…")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    model.max_seq_length = 512
    texts = [d["text"] for d in corpus]

    # Embed in batches to avoid OOM
    embeddings = []
    for i in tqdm(range(0, len(texts), 256), desc="Embedding"):
        batch_emb = model.encode(texts[i:i+256], show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings).astype('float32')

    # Build FAISS index (inner product on unit vectors == cosine sim)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    # Persist
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VEC_DIR / "index.faiss"))
    (VEC_DIR / "id_map.json").write_text(json.dumps([d["id"] for d in corpus]))
    print(f"[✓] Saved FAISS index ({len(corpus):,} vectors) → {VEC_DIR}")
    return index, [d["id"] for d in corpus], model

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
# ❶ 초기 질문 ‑> 검색용 쿼리로 ‘재작성’
# ──────────────────────────────────────────────────
REWRITE_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "You are an expert e‑commerce search assistant.\n\n"
        "Rewrite the user's input as a short, precise search query.\n"
        "If the input is already an optimal search query, return it unchanged.\n\n"
        "User: {user_input}\n"
        "Search‑query:"
    )
)
def rewrite_query(llm: ChatOpenAI, user_input: str) -> str:
    return llm.invoke(REWRITE_PROMPT.format(user_input=user_input)).content.strip()


# ──────────────────────────────────────────────────
# ❷ 대화 이력 + 새 답변 -> ‘재구성된 쿼리’ 생성
# ──────────────────────────────────────────────────
REFORM_PROMPT = PromptTemplate(
    input_variables=["history"],
    template=(
        "You are refining a product‑search query.\n\n"
        "Conversation so far:\n{history}\n\n"
        "Compose ONE refined search query that captures all constraints implicit "
        "or explicit in the conversation. Return ONLY the query."
    )
)
def reformulate_query(llm: ChatOpenAI, turns: list[tuple[str, str]]) -> str:
    """turns = [(question, answer), ...]"""
    history_txt = "\n".join(f"Q: {q}\nA: {a}" for q, a in turns)
    return llm.invoke(REFORM_PROMPT.format(history=history_txt)).content.strip()


# ──────────────────────────────────────────────────
# ❸ 마지막 iteration: 문서 4개 요약
# ──────────────────────────────────────────────────
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["docs"],
    template=(
        "Summarise the following 4 product descriptions in bullet points (≤ 40 words each).\n\n"
        "{docs}\n\n"
        "Return exactly 4 bullet points."
    )
)
def summarise_docs(llm: ChatOpenAI, docs: list[tuple[str, str]]) -> str:
    flat = "\n\n".join(f"[{pid}]\n{text}" for pid, text in docs)
    return llm.invoke(SUMMARY_PROMPT.format(docs=flat)).content.strip()


QUESTION_PROMPT = PromptTemplate(
    input_variables=["items", "context"],
    template=(
        # 역할
        "You are a helpful product‑search assistant.\n\n"
        # 컨텍스트 설명
        "The products listed below were retrieved after considering the entire prior conversation with the user.\n\n"
        # 상품 목록
        "Products (id · snippet):\n{items}\n\n"
        # 대화 맥락
        "Conversation context:\n{context}\n\n"
        # 요청
        "Using BOTH the conversation context and the product list, ask **one** concise follow‑up question "
        "that will help the user further specify what they want.\n"
        "• Do **not** recommend any specific item.\n"
        "• Return **only** the question text.\n"
        "• Do **not** ask something already answered or obvious from the context.\n\n"
    ),
)

def ask_disambiguation(llm: ChatOpenAI, docs, qa_turns):
    # build product snippets
    snippets = []
    for pid, text, _ in docs:
        # head = " ".join(text.split()[:20]) + (" …" if len(text.split()) > 20 else "")
        head = text
        snippets.append(f"{pid} · {head}")

    context =  "None so far." if not qa_turns else "\n".join(f"Q: {turn[0]} A: {turn[1]}" for turn in qa_turns)
    # history = "None so far." if not prev_qs else "\n".join(f"- {q}" for q in prev_qs)
    prompt = QUESTION_PROMPT.format(items="\n".join(snippets), context=context)
    return llm.invoke(prompt).content.strip()


#### main loop



def conversational_search():
    # ㊀ 인덱스 / LLM 초기화
    bm25_idx = _build_or_load_bm25_index(MAX_PRODUCTS)
    vec_idx  = _build_or_load_vector_index(bm25_idx[0])
    llm      = ChatOpenAI(model_name=MODEL_NAME,
                          temperature=TEMPERATURE,
                          streaming=True)

    print("=== Hybrid Conversational Product‑Search ===")
    raw_input = input("You: ").strip()
    if not raw_input or raw_input == "/exit":
        return

    # ㊁ Generation‑1: 초기 쿼리 재작성
    search_query = rewrite_query(llm, raw_input)
    print(f"[ rewritten‑query ] → {search_query}")

    # 대화 이력
    qa_turns: list[tuple[str, str]] = []
    # prev_questions: list[str] = []

    # ㊂‑㊇ 반복
    for round_idx, k in enumerate(TOP_KS, start=1):

        # Retrieval
        docs_k = hybrid_search(search_query, bm25_idx, vec_idx, k)
        # Generation: Clarifying question
        question = ask_disambiguation(llm, docs_k, qa_turns)
        # prev_questions.append(question)

        if question == "[END]" or round_idx == len(TOP_KS):
            #   ↳ 마지막 iteration: 문서 4개 요약 후 종료
            final_hits = hybrid_search(search_query, bm25_idx, vec_idx, 4)


            pids = [pid for pid, _, _ in final_hits]   # keep order if you like
            images_by_pid = fetch_images_for_pids(pids)

            summary = summarise_docs(llm, [(pid, txt) for pid, txt, _ in final_hits])
            print("\n🔎  Top‑4 summary\n" + summary)

            print("\n🖼  Top‑4 제품 이미지 URL")
            for pid in pids:
                image_url = get_best_image_url(images_by_pid.get(pid, []))
                if image_url:
                    print(f"{pid}: {image_url}")
            
            # 사용자에게 비디오 변환 여부 묻기
            should_convert = input("\n이미지를 비디오로 변환하시겠습니까? (y/n): ").strip().lower()
            
            if should_convert == 'y':
                model_type = input("사용할 모델을 선택하세요 (svd/svd_xt, 기본값: svd): ").strip().lower()
                if model_type not in MODEL_NAMES:
                    model_type = "svd"
                    
                print(f"\n[i] 모델 정보: {MODEL_NAMES[model_type]}")
                print(f"[i] 예상 비용: ${MODEL_PRICING[model_type]}/비디오")
                
                if not NOVITA_API_KEY:
                    print("[!] NOVITA_API_KEY가 설정되지 않았습니다. .env 파일에 추가해주세요.")
                else:
                    # 사용자가 변환할 제품 선택
                    print("\n변환할 제품의 번호를 선택하세요 (쉼표로 구분하여 여러 제품 선택 가능, 예: 1,3):")
                    for i, pid in enumerate(pids, 1):
                        print(f"{i}. {pid}")
                    
                    selection = input("선택: ").strip()
                    selected_indices = []
                    
                    try:
                        # 선택한 번호를 인덱스로 변환
                        if selection == "all":
                            selected_indices = list(range(len(pids)))
                        else:
                            selected_indices = [int(idx.strip())-1 for idx in selection.split(",")]
                        
                        # 유효한 인덱스만 필터링
                        selected_indices = [idx for idx in selected_indices if 0 <= idx < len(pids)]
                        
                        if selected_indices:
                            results = []
                            
                            for idx in selected_indices:
                                pid = pids[idx]
                                image_url = get_best_image_url(images_by_pid.get(pid, []))
                                
                                if image_url:
                                    print(f"\n[+] 제품 {pid} 이미지를 비디오로 변환 중...")
                                    result = process_product_to_video(pid, image_url, model=model_type)
                                    results.append((pid, result))
                                else:
                                    print(f"[!] 제품 {pid}의 이미지를 찾을 수 없습니다.")
                            
                            # 결과 요약
                            print("\n=== 변환 결과 요약 ===")
                            success_count = sum(1 for _, r in results if r.get("success"))
                            print(f"성공: {success_count}/{len(results)}")
                            
                            for pid, result in results:
                                if result.get("success"):
                                    status = "기존 비디오 사용" if result.get("already_exists") else "새로 생성됨"
                                    video_path = result.get("video_path", "")
                                    print(f"[✓] 제품 {pid} 비디오 {status}: {video_path}")
                                else:
                                    error = result.get("error", "알 수 없는 오류")
                                    print(f"[✗] 제품 {pid} 비디오 변환 실패: {error}")
                            
                            # 비용 계산
                            new_videos = sum(1 for _, r in results if r.get("success") and not r.get("already_exists"))
                            if new_videos > 0:
                                total_cost = MODEL_PRICING.get(model_type, 0) * new_videos
                                print(f"\n[i] 총 비용: ${total_cost:.4f} ({new_videos}개 비디오 생성)")
                                
                    except ValueError:
                        print("[!] 잘못된 입력입니다.")

            return

        # User prompt ↔ answer 수집
        print(f"Agent: {question}")
        answer = input("You: ").strip()

        qa_turns.append((question, answer))
        # Generation: 대화 이력 기반 쿼리 재구성
        search_query = reformulate_query(llm, qa_turns)
        print(f"[ refined‑query ] → {search_query}\n")


def fetch_images_for_pids(pids: list[str]) -> dict[str, list[str]]:
    """
    Return {pid: [image_url, …]} for every pid in `pids`
    by looking them up in the Amazon‑Reviews‑2023 metadata split.
    """
    pid_set = set(pids)

    meta_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Toys_and_Games",   # ← category split that matches your index
        split="full",
        trust_remote_code=True,
    )

    # HuggingFace Datasets supports vectorised filtering, which is faster
    sub_ds = meta_ds.filter(lambda r: r["parent_asin"] in pid_set)

    # zip → dict: parent_asin ➔ images (list of URLs)
    return dict(zip(sub_ds["parent_asin"], sub_ds["images"]))


def encode_image_to_base64(image_path):
    """이미지 파일을 base64로 인코딩합니다."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"[!] 이미지 인코딩 중 오류 발생: {e}")
        return ""


def resize_image(image_path, output_path=None):
    """이미지 크기를 API 제한에 맞게 조정합니다."""
    if output_path is None:
        output_path = image_path
        
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
            if width / MAX_IMAGE_WIDTH > height / MAX_IMAGE_HEIGHT:
                new_width = MAX_IMAGE_WIDTH
                new_height = int(height * (MAX_IMAGE_WIDTH / width))
            else:
                new_height = MAX_IMAGE_HEIGHT
                new_width = int(width * (MAX_IMAGE_HEIGHT / height))
            
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"[+] 이미지 크기 조정: {width}x{height} → {new_width}x{new_height}")
            resized_img.save(output_path)
            return True
        else:
            print(f"[+] 이미지 크기 적합: {width}x{height}, 조정 불필요")
            return True
    except Exception as e:
        print(f"[!] 이미지 크기 조정 실패: {e}")
        return False


def download_image(url, save_path):
    """URL에서 이미지를 다운로드합니다."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            print(f"[!] 이미지 다운로드 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"[!] 이미지 다운로드 중 오류 발생: {e}")
        return False


def get_best_image_url(images):
    """이미지 목록에서 가장 좋은 품질의 이미지 URL을 반환합니다."""
    if not images or not isinstance(images, list):
        return None
    
    for img in images:
        if isinstance(img, dict):
            if img.get("hi_res"):
                return img["hi_res"]
            elif img.get("large"):
                return img["large"]
            elif img.get("thumb"):
                return img["thumb"]
    
    return None


def convert_image_to_video(image_path, output_path, model="svd"):
    """Novita API를 사용하여 이미지를 비디오로 변환합니다."""
    if not NOVITA_API_KEY:
        print("[!] NOVITA_API_KEY가 설정되지 않았습니다. .env 파일에 추가해주세요.")
        return {"success": False, "error": "API 키가 없습니다"}
    
    try:
        # 모델 선택 및 예상 비용 계산
        model_name = MODEL_NAMES.get(model.lower())
        estimated_cost = MODEL_PRICING.get(model.lower(), "알 수 없음")
        print(f"[i] 선택한 모델: {model_name}, 예상 비용: ${estimated_cost}/비디오")
        
        # 이미지 base64로 인코딩
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return {"success": False, "error": "이미지 인코딩 실패"}
        
        # API 파라미터 설정
        frames_num = 14 if model.lower() == "svd" else 25
        
        payload = {
            "model_name": model_name,
            "image_file": image_base64,
            "frames_num": frames_num,
            "frames_per_second": 6,
            "image_file_resize_mode": "ORIGINAL_RESOLUTION",
            "steps": 20,
            "motion_bucket_id": 40,
            "cond_aug": 0.02
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {NOVITA_API_KEY}"
        }
        
        print(f"[+] Novita API 호출 시작: 이미지 {image_path} → 비디오 변환 중...")
        
        # API 호출
        response = requests.post(NOVITA_API_URL, json=payload, headers=headers)
        print(f"[+] API 응답 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            
            if not task_id:
                return {"success": False, "error": "작업 ID를 받지 못했습니다"}
            
            print(f"[+] 작업 ID: {task_id}. 변환 시작...")
            return check_video_conversion_status(task_id, output_path, model=model)
        else:
            print(f"[!] API 호출 실패: {response.status_code}")
            return {"success": False, "error": f"API 호출 실패: {response.status_code}"}
    except Exception as e:
        print(f"[!] 비디오 변환 중 오류 발생: {e}")
        return {"success": False, "error": str(e)}


def check_video_conversion_status(task_id, output_path, model="svd"):
    """비디오 변환 작업 상태를 확인하고 완료되면 다운로드합니다."""
    status_url = f"https://api.novita.ai/v3/async/task-result?task_id={task_id}"
    
    headers = {
        "Authorization": f"Bearer {NOVITA_API_KEY}"
    }
    
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            response = requests.get(status_url, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                task_info = result.get("task", {})
                status = task_info.get("status")
                
                if status == "TASK_STATUS_SUCCEED":
                    videos = result.get("videos", [])
                    if videos and len(videos) > 0:
                        video_url = videos[0].get("video_url")
                        if video_url:
                            print(f"[+] 비디오 생성 완료. 다운로드 중...")
                            video_response = requests.get(video_url)
                            if video_response.status_code == 200:
                                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                                with open(output_path, 'wb') as f:
                                    f.write(video_response.content)
                                print(f"[✓] 비디오 저장 완료: {output_path}")
                                return {"success": True, "video_path": output_path}
                            else:
                                return {"success": False, "error": "비디오 다운로드 실패"}
                        else:
                            return {"success": False, "error": "비디오 URL을 찾을 수 없습니다"}
                    else:
                        return {"success": False, "error": "비디오 정보를 찾을 수 없습니다"}
                elif status == "TASK_STATUS_FAILED":
                    reason = task_info.get("reason", "알 수 없는 이유")
                    return {"success": False, "error": f"작업 실패: {reason}"}
                elif status == "TASK_STATUS_PENDING" or status == "TASK_STATUS_RUNNING":
                    progress = task_info.get("progress_percent", 0)
                    print(f"[+] 변환 중... ({attempt+1}/{max_attempts}) - 진행률: {progress}%")
                    time.sleep(10)
                else:
                    time.sleep(10)
            else:
                time.sleep(10)
        except Exception as e:
            print(f"[!] 상태 확인 중 오류: {e}")
            time.sleep(10)
    
    return {"success": False, "error": "시간 초과"}


def process_product_to_video(product_id, image_url, output_dir="output_videos", model="svd"):
    """제품 이미지를 비디오로 변환합니다."""    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 파일 경로 설정
    image_path = os.path.join(temp_dir, f"{product_id}.jpg")
    video_path = os.path.join(output_dir, f"{product_id}.mp4")
    
    # 이미지가 이미 존재하는지 확인
    if not os.path.exists(image_path):
        if not download_image(image_url, image_path):
            return {"success": False, "error": "이미지 다운로드 실패"}
    
    # 이미지 크기 조정
    if not resize_image(image_path):
        return {"success": False, "error": "이미지 크기 조정 실패"}
    
    # 비디오가 이미 존재하는지 확인
    if os.path.exists(video_path):
        print(f"[!] 비디오가 이미 존재합니다: {video_path}")
        return {"success": True, "video_path": video_path, "already_exists": True}
    
    # 이미지를 비디오로 변환
    result = convert_image_to_video(image_path, video_path, model=model)
    
    if result.get("success"):
        result["model"] = model
    
    return result


if __name__ == "__main__":
    try:
        conversational_search()
    except KeyboardInterrupt:
        print("\n[Session terminated]")
