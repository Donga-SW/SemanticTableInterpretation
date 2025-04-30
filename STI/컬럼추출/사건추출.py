import pandas as pd

# 두 파일 불러오기
df1 = pd.read_csv("judgement1.csv", encoding="utf-8")
df2 = pd.read_csv("judgement2.csv", encoding="utf-8")

# '사건' 컬럼만 추출하고 NaN 제거
incident1 = df1["사건"].dropna()
incident2 = df2["사건"].dropna()

# 두 시리즈 합치기
all_incidents = pd.concat([incident1, incident2]).reset_index(drop=True)

# CSV로 저장
all_incidents.to_csv("incident_all.csv", index=False, header=["사건"])

