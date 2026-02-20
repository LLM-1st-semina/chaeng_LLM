from sentence_transformers import SentenceTransformer, util

# 한국어 문맥 파악에 최적화된 S-BERT(DeBERTa/RoBERTa 기반) 모델
model = SentenceTransformer('jhgan/ko-sbert-nli')

sentences = [
    "어제 먹은 사과가 정말 달고 맛있었다.",         # 기준 문장 (과일)
    "과일 가게에 가서 빨간 사과 한 봉지를 샀다.",  # 의미 유사
    "그는 자신의 잘못을 진심으로 사과했다."         # 단어 중복, 의미 다름
]

embeddings = model.encode(sentences)

# 코사인 유사도 계산
sim_pos = util.cos_sim(embeddings[0], embeddings[1])
sim_neg = util.cos_sim(embeddings[0], embeddings[2])

print(f"\n[실습 : 임베딩 분석] 기준: {sentences[0]}")
print("="*60)
print(f"1. 의미 유사 문장(과일) 유사도: {sim_pos.item():.4f}")
print(f"2. 단어 중복 문장(사죄) 유사도: {sim_neg.item():.4f}")
print("-" * 60)
print("결과: 모델은 '사과'라는 글자가 아닌 문맥 전체의 의미 벡터를 비교함.")
