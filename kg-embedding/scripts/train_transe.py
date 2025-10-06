import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os

# 데이터 로드
df = pd.read_csv("../data/triplets_independent.csv")
triples = list(df.itertuples(index=False, name=None))

# 엔티티 / 관계 ID 매핑
entities = sorted(set([h for h, r, t in triples] + [t for h, r, t in triples]))
relations = sorted(set([r for h, r, t in triples]))
entity2id = {e: i for i, e in enumerate(entities)}
relation2id = {r: i for i, r in enumerate(relations)}

# 파라미터
embedding_dim = 128
margin = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 임베딩 초기화
entity_emb = nn.Embedding(len(entities), embedding_dim).to(device)
relation_emb = nn.Embedding(len(relations), embedding_dim).to(device)
nn.init.xavier_uniform_(entity_emb.weight.data)
nn.init.xavier_uniform_(relation_emb.weight.data)

# 거리 함수
def distance(h, r, t):
    return torch.norm(h + r - t, p=1, dim=1)

# 손실 함수
def transe_loss(pos, neg, margin):
    pos_dist = distance(*pos)
    neg_dist = distance(*neg)
    return torch.relu(margin + pos_dist - neg_dist).mean()

# Optimizer
optimizer = optim.Adam(list(entity_emb.parameters()) + list(relation_emb.parameters()), lr=1e-3)

# 학습 루프
EPOCHS = 10
BATCH_SIZE = 512
triples_id = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]

for epoch in range(EPOCHS):
    random.shuffle(triples_id)
    total_loss = 0
    for i in range(0, len(triples_id), BATCH_SIZE):
        batch = triples_id[i:i+BATCH_SIZE]
        h, r, t = zip(*batch)
        h = torch.tensor(h).to(device)
        r = torch.tensor(r).to(device)
        t = torch.tensor(t).to(device)

        # negative sampling (tail corruption)
        t_neg = torch.randint(0, len(entities), t.shape).to(device)

        pos = (entity_emb(h), relation_emb(r), entity_emb(t))
        neg = (entity_emb(h), relation_emb(r), entity_emb(t_neg))
        loss = transe_loss(pos, neg, margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# 임베딩 저장
os.makedirs("../outputs/manual_transe", exist_ok=True)
torch.save(entity_emb.state_dict(), "../outputs/manual_transe/entity_emb.pt")
torch.save(relation_emb.state_dict(), "../outputs/manual_transe/relation_emb.pt")

print("TransE manual training complete.")
print(f"Entities: {len(entities)} | Relations: {len(relations)}")
