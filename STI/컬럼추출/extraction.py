import pandas as pd
import numpy as np

df = pd.read_csv("judge1.csv",encoding="cp949")

columns = ['이름_별명','당시나이','본적주소','죄명','주문','판결날짜','사건개요']


new_df = df[columns]
new_columns = ['성명','나이','주소','죄명','주문','판결날짜','사건']
new_df.columns = new_columns
# 저장
new_df.to_csv("judgement1.csv", index=False, encoding="utf-8")
