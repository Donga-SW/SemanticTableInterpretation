import pandas as pd
import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. CSV 로드
df = pd.read_csv("C:/Users/lenovo/Desktop/KEYWORD/최종성명ID완료유나.csv")

# 2. '사건' 컬럼이 존재하고 비어있지 않은 행만 유지
if "사건" not in df.columns:
    raise ValueError("'사건' 컬럼이 CSV 파일에 없습니다.")

df = df[df["사건"].notnull() & df["사건"].astype(str).str.strip().astype(bool)]

# 3. 명사 추출 함수 정의
okt = Okt()
def extract_nouns(text):
    text = re.sub(r"[^가-힣\s]", " ", str(text))
    nouns = okt.nouns(text)
    return " ".join(n for n in nouns if len(n) > 1)

# 4. 명사 추출 처리 + 진행상황 출력
nouns_result = []
for i, text in enumerate(df["사건"]):
    if i % 50 == 0:
        print(f"{i}번째 사건 처리 중...")
    try:
        nouns = extract_nouns(text)
    except Exception as e:
        nouns = ""
    nouns_result.append(nouns)

df["사건_명사"] = nouns_result
df = df[df["사건_명사"].str.strip().astype(bool)]

# 5. TF-IDF 적용
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["사건_명사"])
feature_names = vectorizer.get_feature_names_out()

# 6. 사건별 상위 키워드 추출
top_n = 3
keywords = []
for row in X.toarray():
    top_indices = row.argsort()[::-1][:top_n]
    top_keywords = [feature_names[i] for i in top_indices if row[i] > 0]
    keywords.append(top_keywords)

df["사건_키워드"] = keywords

# 7. 결과 저장
df.to_csv("C:/Users/lenovo/Desktop/TFIDF_keyword.csv", index=False, encoding='utf-8-sig')

# 8. 확인 출력
print("\n처리 완료. 결과 샘플:")
print(df[["성명", "사건", "사건_키워드"]].head())
