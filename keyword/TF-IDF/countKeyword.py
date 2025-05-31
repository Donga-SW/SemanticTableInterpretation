import pandas as pd
from collections import Counter
from ast import literal_eval
import os

# 파일 경로 설정
base_path = r"C:\Users\lenovo\Desktop"
input_path = os.path.join(base_path, "TFIDF_keyword.csv")
output_path = os.path.join(base_path, "countKeyword.csv")

# CSV 파일 읽기
df = pd.read_csv(input_path, encoding="utf-8")

# 사건_키워드 컬럼을 리스트로 변환
df["사건_키워드"] = df["사건_키워드"].apply(literal_eval)

# 키워드 평탄화 및 빈도 계산
all_keywords = [kw for keywords in df["사건_키워드"] for kw in keywords]
keyword_counts = Counter(all_keywords)

# 데이터프레임 생성 및 정렬
df_keywords = pd.DataFrame(keyword_counts.items(), columns=["키워드", "키워드_빈도"])
df_keywords = df_keywords.sort_values(by="키워드_빈도", ascending=False)

# CSV로 저장
df_keywords.to_csv(output_path, index=False, encoding="utf-8-sig")
