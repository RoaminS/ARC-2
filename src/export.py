import faiss
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from src.dsl import dsl
import os
import random

model = load_model("models/cnn.h5")

def compute_embedding(grid, model, size=30):
    padded = np.zeros((size, size))
    padded[:grid.shape[0], :grid.shape[1]] = grid
    inp = padded[np.newaxis, ..., np.newaxis] / 10.0
    return model.predict(inp, verbose=0)[0]

embeddings = []
programs = []
for _ in range(5000):
    grid = np.random.randint(0, 10, (random.randint(5, 30), random.randint(5, 30)))
    prim = random.choice(dsl.primitives)
    out = dsl.apply(grid, prim)
    e1 = compute_embedding(grid, model)
    e2 = compute_embedding(out, model)
    diff = e2 - e1
    embeddings.append(diff)
    programs.append(prim)

embeddings = np.array(embeddings).astype('float32')
index = faiss.IndexHNSWFlat(128, 32)
index.add(embeddings)

faiss.write_index(index, "models/faiss.index")
with open("models/programs.pkl", "wb") as f:
    pickle.dump(programs, f)

print("[✅] Export FAISS + programmes terminés")
