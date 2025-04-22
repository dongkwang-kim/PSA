# Conversational Product Search Agent

원하는 제품을 효율적으로 찾을 수 있도록 지원하는 대화형 제품 검색 에이전트. 이 시스템은 BM25s 검색 모델과 GPT 기반 언어 모델(gpt-4.1-mini)을 활용하여 검색 쿼리를 점진적으로 구체화하고 제품 선택지를 점차 줄여나가는 방식으로 작동합니다.

## Overview

### Workflow

1. **초기 검색**: 사용자가 텍스트 쿼리로 검색을 시작합니다.
2. **초기 결과 추출 (Top-64)**: 시스템이 초기 64개의 제품 후보군을 추출합니다.
3. **요약 및 질문 생성**: LLM이 추출된 제품 정보를 요약하고 사용자 의도를 더 명확히 파악할 수 있는 질문을 생성합니다.
4. **사용자 응답**: 사용자가 질문에 답변하여 보다 구체적인 정보를 제공합니다.
5. **반복적 세부화**: 이 과정을 반복하여 검색 쿼리를 세부화하며 제품 후보군을 점차 줄여나갑니다:
    - 64 → 32 → 16 → 8 → 4
6. **최종 선택**: 마지막으로 남은 4개의 제품 중 사용자가 가장 적합한 제품을 선택합니다.


## Dataset

The agent utilizes the **Amazon Reviews 2023** dataset from Hugging Face, specifically the Sports & Outdoors category.

- **Dataset**: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Subset**: `raw_meta_Toys_and_Games`

### Data Loading

```python
from datasets import load_dataset

dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Toys_and_Games",
    split="full",
    trust_remote_code=True
)
```

### Data Structure

- **ID**: `parent_asin`
- **Content**: Concatenation of `title`, `features`, and `description`

## Retrieval Model

The retrieval component uses **BM25s**, a sparse retrieval model optimized for text search.

- **Index Fields**: Product titles, features, descriptions
- **Search Strategy**: Iteratively refines results based on user feedback and LLM-generated queries

## Usage

### Dependencies Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### Running the Agent

```bash
python run_agent.py
```