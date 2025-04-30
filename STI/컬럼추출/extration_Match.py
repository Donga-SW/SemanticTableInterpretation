import pandas as pd


labeled_df = pd.read_csv("incident_labeled_final.csv", encoding="utf-8")
judgement_df = pd.read_csv("judgement2.csv", encoding="utf-8")

#  독립운동 관련 사건 개요만 필터링
related_incidents = labeled_df[labeled_df["label_final"] == 1]["사건"].dropna().unique()

# judgement2에서 사건이 해당하는 행만 추출
filtered_df = judgement_df[judgement_df["사건"].isin(related_incidents)]


filtered_df.to_csv("judgement2_final.csv", index=False, encoding="utf-8")

