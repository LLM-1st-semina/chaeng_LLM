from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. QA 모델 로드
qa_nlp = pipeline("question-answering", model="timpal0l/mdeberta-v3-base-squad2")

# 2. 시나리오 설정
# [실험 환경] 지문에는 '1608년'만 있고 '광해군'이라는 정보는 의도적으로 뺌
passage = """
대동법은 조선 시대에 공물을 쌀로 통일하여 바치게 한 제도로, 1608년 경기도에서 처음 실시되었다. 
이후 숙종 대에 이르러 전국적으로 확대되었으며, 농민의 조세 부담을 줄여주었다.
"""
question = "대동법은 어느 왕 때 처음 실시되었나?"

# [실험 1] No-RAG: 지식의 한계로 인한 환각 유도
# 모델은 지문에 답이 없으면 지문 속의 다른 단어(예: 숙종)를 가져와 잘못된 답을 내놓음
res_no_rag = qa_nlp(question=question, context="조선 시대에는 다양한 조세 제도가 존재했다.")

# [실험 2] RAG: 정확한 지식 베이스 제공
# 실제 상용 RAG라면 검색 엔진이 '광해군'이 포함된 더 정확한 문서를 찾아왔을 상황을 가정
fact_passage = passage + " 당시 왕은 광해군이었다."
res_rag = qa_nlp(question=question, context=fact_passage)

# 데이터 정리
results = [
    {"Label": "No-RAG\n(정보 부족/환각)", "Answer": res_no_rag['answer'], "Score": res_no_rag['score']},
    {"Label": "RAG\n(데이터 보강/팩트)", "Answer": res_rag['answer'], "Score": res_rag['score']}
]
df = pd.DataFrame(results)

plt.style.use('dark_background')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 7))

# 이미지와 유사한 그라데이션 컬러 (Magma 팔레트 사용)
colors = sns.color_palette("magma", len(df))
bars = ax.bar(df['Label'], df['Score'], color=colors, edgecolor='white', linewidth=1.5, alpha=0.8)

# 이미지 스타일의 상단 텍스트 추가 (Cyan 컬러 강조)
for bar, res in zip(bars, results):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f"Score: {res['Score']:.4f}\n(Ans: {res['Answer']})",
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='cyan')

# 그래프 디테일 설정 (이미지 레이아웃 최적화)
ax.set_title("🔍 RAG 시스템 성능 분석: 대동법 환각(Hallucination) 방지 테스트", fontsize=20, pad=40, fontweight='bold')
ax.set_ylabel("Confidence Score (0~1)", fontsize=12, color='gray')
ax.set_ylim(0, 1.2) # 텍스트 공간 확보
ax.grid(axis='y', linestyle='--', alpha=0.3)
sns.despine(left=True, bottom=True)

plt.tight_layout()
print("\n[세미나용 시각화 창 확인 중...]")
plt.show()
