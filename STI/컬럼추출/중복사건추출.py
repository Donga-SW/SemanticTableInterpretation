import pandas as pd

df = pd.read_csv("judgement2_final.csv", encoding="utf-8")

counts = df["사건"].value_counts()
duplicated_incidents = counts[counts >= 2].index
filtered_df = df[df["사건"].isin(duplicated_incidents)].copy()

filtered_df["사건_내_순번"] = filtered_df.groupby("사건").cumcount() + 1

filtered_df = filtered_df.sort_values(by=["사건", "사건_내_순번"])

filtered_df.to_csv("judgement2_multi.csv", index=False, encoding="utf-8-sig")
