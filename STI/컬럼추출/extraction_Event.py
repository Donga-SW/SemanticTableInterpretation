from openai import OpenAI
import pandas as pd
import time
'''
사건 추출에 관한 코드입니다.
1. csv파일에서 사건개요만 뽑아서 시리즈로 변환
2. GPT LLM으로 1차 독립 사건인지 아닌지 이진함수 분류
3. 키워드 값으로 2차 분류
3. LLM + 키워드 바탕  최종 분류 
4. 추출 후 저장
'''
client = OpenAI(api_key="")
df = pd.read_csv("incident_all.csv")

# 독립 관련 키워드 정리 > 추후에 추가 될 수 있음.
keywords = [
    "독립", "의열단", "대한민국임시정부", "상해임시정부", "조선총독부", "항일", "일제", "무장투쟁", "무장", "저항",
    "폭탄", "의거", "거사", "투탄", "폭발", "사보타주", "태극기", "독립선언서", "3.1운동", "3·1운동", "만세운동",
    "한인애국단", "애국단", "이봉창", "윤봉길", "김좌진", "안중근", "신채호", "백범", "김구", "독립신문", "광복군",
    "항쟁", "독립투사", "애국지사", "애국열사", "조선청년독립단", "학생운동", "민족해방", "독립만세", "자결", "공화제",
    "독립운동", "독립의사", "자유", "민족자결", "민족자주", "자주", "해방"
]

# 이진함수 분류
def classify_with_gpt(text):
    prompt = f"다음 사건 설명이 독립운동과 관련이 있으면 1, 관련이 없으면 0이라고만 답하세요:\n\"{text}\""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1
        )
        reply = response.choices[0].message.content.strip()
        return int(reply) if reply in ['0', '1'] else 0
    except Exception as e:
        print("GPT 오류:", e)
        return 0

# 키워드 라벨링 함수
def classify_with_keyword(text):
    return int(any(k in str(text) for k in keywords))

# 라벨링 루프
label_gpt = []
label_keyword = []

for i, row in df.iterrows():
    text = str(row["사건"])
    gpt_label = classify_with_gpt(text)
    keyword_label = classify_with_keyword(text)
    label_gpt.append(gpt_label)
    label_keyword.append(keyword_label)
    time.sleep(1.2) #슬립의 딜레마..


df["label_gpt"] = label_gpt
df["label_keyword"] = label_keyword
df["label_final"] = df[["label_gpt", "label_keyword"]].max(axis=1)

df.to_csv("incident_labeled_final.csv", index=False, encoding="utf-8-sig")

print("최종 라벨링 완료 → incident_labeled_final.csv 저장됨")
