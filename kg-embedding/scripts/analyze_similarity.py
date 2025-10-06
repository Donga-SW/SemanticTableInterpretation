import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# 경로 설정
data_path = "../data/triplets_independent.csv"
entity_emb_path = "../outputs/manual_transe/entity_emb.pt"
relation_emb_path = "../outputs/manual_transe/relation_emb.pt"

# CSV 로드 및 매핑 복원
df = pd.read_csv(data_path)
triples = list(df.itertuples(index=False, name=None))

entities = sorted(set([h for h, r, t in triples] + [t for h, r, t in triples]))
relations = sorted(set([r for h, r, t in triples]))
entity2id = {e: i for i, e in enumerate(entities)}
id2entity = {i: e for e, i in entity2id.items()}

#  임베딩 로드
embedding_dim = 128
entity_emb = nn.Embedding(len(entities), embedding_dim)
entity_emb.load_state_dict(torch.load(entity_emb_path, map_location="cpu"))
entity_emb.eval()

print(f"✅ Loaded embeddings: {len(entities)} entities")

# 유사도 함수 (코사인 유사도)
def cosine_similarity(vec1, vec2):
    return torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm() + 1e-9)

def find_similar(entity_name, top_k=5):
    if entity_name not in entity2id:
        print(f"'{entity_name}' not found in entities.")
        return []

    idx = entity2id[entity_name]
    target_vec = entity_emb(torch.tensor(idx))
    sims = []

    for i, name in id2entity.items():
        if name == entity_name:
            continue
        vec = entity_emb(torch.tensor(i))
        sim = cosine_similarity(target_vec, vec).item()
        sims.append((name, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

# 테스트 실행
query = "N0001"  # 임의의 인물ID
top_sim = find_similar(query, top_k=5)

print(f"\n🔍 '{query}'와 가장 유사한 인물 TOP 5:")
for name, score in top_sim:
    print(f"  - {name}: {score:.4f}")
