import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re

# 1. Load data
data_path = "../data/triplets_independent.csv"
entity_emb_path = "../outputs/manual_transe/entity_emb.pt"

df = pd.read_csv(data_path)
triples = list(df.itertuples(index=False, name=None))

entities = sorted(set([h for h, r, t in triples] + [t for h, r, t in triples]))
entity2id = {e: i for i, e in enumerate(entities)}
id2entity = {i: e for e, i in entity2id.items()}

# 2. Load embedding
embedding_dim = 128
entity_emb = nn.Embedding(len(entities), embedding_dim)
entity_emb.load_state_dict(torch.load(entity_emb_path, map_location="cpu"))
entity_emb.eval()

emb_matrix = entity_emb.weight.detach().numpy()
print(f"Embedding loaded: {emb_matrix.shape}")

# 3. Sample subset for visualization
np.random.seed(42)
sample_size = min(1500, len(entities))
sample_idx = np.random.choice(len(entities), size=sample_size, replace=False)
sample_emb = emb_matrix[sample_idx]
sample_labels = [id2entity[i] for i in sample_idx]

# 4. Assign color by entity type
def get_type(label):
    if re.match(r'^N\d+', label):
        return "Person"
    elif re.match(r'^e\d+', label):
        return "Event"
    elif re.match(r'^crime_', label):
        return "Crime"
    elif re.match(r'^addr_', label):
        return "Address"
    elif re.match(r'^year_', label):
        return "Year"
    else:
        return "Other"

types = [get_type(lbl) for lbl in sample_labels]
colors = {
    "Person": "purple",
    "Event": "green",
    "Crime": "skyblue",
    "Address": "orange",
    "Year": "gold",
    "Other": "gray"
}

# 5. t-SNE dimensionality reduction (128D → 2D)
print("Running t-SNE (this may take a few minutes)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=50, init="pca")
emb_2d = tsne.fit_transform(sample_emb)

# 6. Plot 2D visualization
plt.figure(figsize=(10, 8))
for t in set(types):
    idx = [i for i, typ in enumerate(types) if typ == t]
    plt.scatter(
        emb_2d[idx, 0],
        emb_2d[idx, 1],
        c=colors[t],
        label=t,
        alpha=0.7,
        s=25
    )

plt.title("TransE Embedding Visualization (t-SNE 2D)", fontsize=14)
plt.legend()
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.show()
