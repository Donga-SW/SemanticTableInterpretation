import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BertModel, get_scheduler

# 1. 데이터 불러오기
df = pd.read_csv("train_new_labels.csv", encoding='utf-8-sig')
new_labels = df["label"].unique().tolist()
label_map = {label: i for i, label in enumerate(new_labels)}
df["label_id"] = df["label"].map(label_map)

# 2. KoBERT 로드
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
bertmodel = BertModel.from_pretrained('monologg/kobert')  # 기존 KoBERT 가중치 그대로

# 3. Dataset 정의
class NewLabelDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

dataset = NewLabelDataset(df["text"].tolist(), df["label_id"].tolist())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. 새로운 classifier만 정의
class NewClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=len(label_map)):
        super().__init__()
        self.bert = bert  # 기존 KoBERT
        self.classifier = nn.Linear(hidden_size, num_classes)  # 새 라벨용 분류기

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(pooled)

# 5. 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NewClassifier(bertmodel).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)*5)

# 6. 학습 루프
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} 완료 - 평균 Loss: {avg_loss:.4f}")

# 7. 새 모델 저장
torch.save(model.state_dict(), "kobert_newlabels_finetuned.pt")
