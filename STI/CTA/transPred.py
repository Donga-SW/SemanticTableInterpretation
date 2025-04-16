import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch import nn

# 학습 시 사용한 라벨맵 (예: 나이, 주소, 죄명 등)
label_map = {"죄명": 0, "판결": 1, "등등": 2, "등등": 3}
num_classes = len(label_map)

# 모델 정의 (학습과 동일하게)
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=4):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

# 토크나이저 및 모델 불러오기
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
bertmodel = BertModel.from_pretrained("monologg/kobert")
model = BERTClassifier(bertmodel, num_classes=num_classes)
model.load_state_dict(torch.load("kobert_newlabels_finetuned.pt", map_location="cpu"))
model.eval()

# 예측 함수
def predict_label(text, target_label="주소"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.softmax(logits, dim=1)
        return probs[0][label_map[target_label]].item()

# CSV 불러오기
df = pd.read_csv("judgement.csv", encoding='cp949')

# 컬럼별 평균 확률 계산
target = "주소"  # 여기 바꾸면 됨 ("성명", "나이", "죄명" 등)
results = []
for col in df.columns:
    values = df[col].dropna().astype(str).tolist()[:100]
    probs = [predict_label(val, target_label=target) for val in values]
    avg_prob = sum(probs) / len(probs) if probs else 0
    results.append((col, avg_prob))

# 출력
results.sort(key=lambda x: x[1], reverse=True)
print(f"\n🔍 CSV에서 '{target}' 컬럼으로 가장 유력한 후보:")
for col, score in results:
    print(f" - {col}: {target}일 확률 평균 {score:.4f}")
