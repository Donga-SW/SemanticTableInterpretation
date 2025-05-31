import pandas as pd
from collections import defaultdict, Counter
from ast import literal_eval

# 파일 경로 설정
input_path = r"C:\Users\lenovo\Desktop\KEYWORD\TF-IDF\csv\TFIDF_KEYWORD.csv"
output_path = r"C:\Users\lenovo\Desktop\keyword\TF-IDF\csv\countPersonKeyword.csv"

# CSV 파일 읽기
df = pd.read_csv(input_path, encoding="utf-8")
df["사건_키워드"] = df["사건_키워드"].apply(literal_eval)

# 키워드 빈도 계산
all_keywords = [kw for keywords in df["사건_키워드"] for kw in keywords]
keyword_freq = Counter(all_keywords)

# 키워드별 인물 수 및 인물 이름 수집
keyword_to_people = defaultdict(set)
for _, row in df.iterrows():
    name = row["성명"]
    for kw in row["사건_키워드"]:
        keyword_to_people[kw].add(name)

# 최종 데이터 구성
rows = []
for idx, (kw, freq) in enumerate(keyword_freq.most_common()):
    people = list(keyword_to_people[kw])
    people_count = len(people)
    people_display = ", ".join(people[:10])  # 최대 10명
    rows.append({
        "index": idx,
        "키워드": kw,
        "키워드_빈도": freq,
        "포함 인물 수": people_count,
        "포함 인물": people_display
    })

# 데이터프레임 생성 및 저장
df_result = pd.DataFrame(rows)
df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
