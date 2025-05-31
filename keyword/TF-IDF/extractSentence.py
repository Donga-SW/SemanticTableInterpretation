import pandas as pd
from ast import literal_eval

# 파일 경로 설정
keyword_path = r"C:\Users\lenovo\Desktop\KEYWORD\TF-IDF\csv\countPersonKeyword.csv"
data_path = r"C:\Users\lenovo\Desktop\KEYWORD\TF-IDF\csv\TFIDF_keyword.csv"
output_path = r"C:\Users\lenovo\Desktop\KEYWORD\TF-IDF\csv\extractSentence_keyword.csv"

# CSV 파일 읽기
df_keywords = pd.read_csv(keyword_path)
df_data = pd.read_csv(data_path)

# 사건_키워드 컬럼을 리스트로 변환
df_data["사건_키워드"] = df_data["사건_키워드"].apply(literal_eval)

# 상위 30개 키워드 추출
top30 = df_keywords.head(30)[["키워드", "키워드_빈도", "포함 인물 수"]]

# 결과 저장 리스트
results = []

# 각 키워드별 문장 수집
for idx, row in top30.iterrows():
    keyword = row["키워드"]
    freq = row["키워드_빈도"]
    people_count = row["포함 인물 수"]

    # 해당 키워드가 포함된 사건 문장들 찾기
    matched_sentences = df_data[df_data["사건_키워드"].apply(lambda kws: keyword in kws)]["사건"]
    unique_sentences = matched_sentences.dropna().unique()
    sentence_count = len(unique_sentences)
    vertical_sentences = "\n".join(unique_sentences)

    results.append({
        "index": idx,
        "키워드": keyword,
        "키워드_빈도": freq,
        "포함_인물_수": people_count,
        "포함_문장_수": sentence_count,
        "포함_문장": vertical_sentences
    })

# DataFrame 생성 및 저장
df_result = pd.DataFrame(results)
df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
