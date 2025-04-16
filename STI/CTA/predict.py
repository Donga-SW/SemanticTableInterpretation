import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import torch.nn.functional as F

# 훈련 시 사용한 라벨 맵 고정
label_map = {"성명": 0, "출생": 1, "성별": 2, "지역": 3}

# 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=4):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

# 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bertmodel = BertModel.from_pretrained("monologg/kobert")
model = BERTClassifier(bertmodel, num_classes=4)
model.load_state_dict(torch.load("kobert_name_finetuned.pt", map_location="cpu"))
model.eval()

# 성명 예측 함수
def predict_is_name(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.softmax(logits, dim=1)
        return probs[0][label_map["성명"]].item()

# 판결문 CSV 불러오기
df = pd.read_csv("judgement.csv", encoding='cp949')

# 컬럼별 평균 성명확률 계산
results = []
for col in df.columns:
    values = df[col].dropna().astype(str).tolist()[:100]
    probs = [predict_is_name(val) for val in values]
    avg_prob = sum(probs) / len(probs) if probs else 0
    results.append((col, avg_prob))

# 출력
results.sort(key=lambda x: x[1], reverse=True)
print("CSV에서 '성명' 컬럼으로 가장 유력한 후보:")
for col, score in results:
    print(f" {col}: 성명일 확률 평균 {score:.4f}")
