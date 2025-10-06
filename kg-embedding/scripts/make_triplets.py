import pandas as pd

# 1. CSV 불러오기
df = pd.read_csv("../data/independence-cases.csv", encoding="utf-8")

triplets = []

# 2. 각 행(row)에서 관계 추출
for _, row in df.iterrows():
    head = row["성명ID"]
    # 인물 ↔ 속성 관계들
    if pd.notna(row["나이"]):
        triplets.append((head, "나이", str(row["나이"])))
    if pd.notna(row["죄명"]):
        triplets.append((head, "죄명", str(row["죄명"])))
    if pd.notna(row["주문"]):
        triplets.append((head, "주문", str(row["주문"])))
    if pd.notna(row["판결날짜"]):
        year = str(row["판결날짜"]).split("-")[0]
        triplets.append((head, "판결연도", f"year_{year}"))
    if pd.notna(row["사건_ID"]):
        triplets.append((head, "연루사건", str(row["사건_ID"])))
    if pd.notna(row["주소"]):
        triplets.append((head, "본적주소", f"addr_{row['주소']}"))
    if pd.notna(row["죄명_URI"]) and row["죄명_URI"] != "없음":
        triplets.append((row["죄명"], "URI", row["죄명_URI"]))
    if pd.notna(row["주소_URI"]) and row["주소_URI"] != "없음":
        triplets.append((row["주소"], "URI", row["주소_URI"]))

# 3. CSV로 저장
triplet_df = pd.DataFrame(triplets, columns=["head", "relation", "tail"])
triplet_df.to_csv("../data/triplets_independent.csv", index=False, encoding="utf-8-sig")

print(f"Triplet 변환 완료: {len(triplets)}개 관계 생성됨")
