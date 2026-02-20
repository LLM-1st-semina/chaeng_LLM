import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. 모델 설정 (BPE 기반 구형 vs 문맥 기반 한국어 특화)
MODELS = {
    "GPT-2 (BPE/Global)": "gpt2",
    "KLUE-RoBERTa (Contextual/KR)": "klue/roberta-small" # DeBERTa와 유사한 로직의 한국어 모델
}

# 2. 분석할 문장 (다의어 '사과'를 활용해 문맥 파악 능력 테스트)
sentences = [
    "시장에서 맛있는 사과를 샀다",          # 과일 (A)
    "아삭한 사과가 먹고 싶다",            # 과일 (B) - A와 가까워야 함
    "친구에게 진심으로 사과를 했다",        # 사죄 (C) - A, B와 멀어야 함
    "미안한 마음을 담아 사과를 건넸다"      # 사죄 (D) - C와 가까워야 함
]

def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] 토큰 혹은 평균 풀링을 사용하여 문장 전체의 수치화된 의미(Vector) 추출
    return outputs.last_hidden_state[:, 0, :].numpy()

print(f"--- [데이터와 알고리즘: 왜 모델은 내 말을 오해할까?] ---\n")

for name, model_path in MODELS.items():
    print(f"### 분석 모델: {name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # GPT-2는 padding 토큰이 없으므로 설정
    if "gpt2" in model_path:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModel.from_pretrained(model_path)

    # A. 토큰화 결과 비교 (데이터 왜곡 확인)
    sample_text = sentences[0]
    tokens = tokenizer.tokenize(sample_text)
    print(f"1. 토큰화 예시 ({sample_text}):")
    print(f"   결과: {tokens}")
    print(f"   개수: {len(tokens)}개")

    # B. 문맥 기반 유사도 분석 (벡터 공간 보정 확인)
    vecs = [get_embeddings(s, model, tokenizer) for s in sentences]
    
    # '과일(A)'과 '과일(B)'의 유사도 vs '과일(A)'과 '사죄(C)'의 유사도
    sim_fruit = cosine_similarity(vecs[0], vecs[1])[0][0]
    sim_conflict = cosine_similarity(vecs[0], vecs[2])[0][0]

    print(f"2. 문맥 분석 결과:")
    print(f"   - [과일 vs 과일] 유사도: {sim_fruit:.4f}")
    print(f"   - [과일 vs 사죄] 유사도: {sim_conflict:.4f}")
    
    if sim_fruit > sim_conflict:
        print(f"   => 결과: 문맥을 올바르게 구분하고 있습니다. (보정 완료)")
    else:
        print(f"   => 결과: 단어 모양에 집착하여 문맥을 오해하고 있습니다. (왜곡 발생)")
    print("-" * 50)
