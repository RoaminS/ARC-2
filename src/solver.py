import numpy as np
import pickle
import faiss
from tensorflow.keras.models import load_model
from src.dsl import dsl
import random
import os

MODEL_PATH = "models/cnn.h5"
INDEX_PATH = "models/faiss.index"
PROGRAMS_PATH = "models/programs.pkl"
RL_POLICY_PATH = "models/rl_policy_model.h5"

cnn = load_model(MODEL_PATH)
index = faiss.read_index(INDEX_PATH)
with open(PROGRAMS_PATH, "rb") as f:
    program_map = pickle.load(f)

# === Chargement agent RL
try:
    rl_model = load_model(RL_POLICY_PATH)
    print("[🎯] RL agent chargé avec succès")
except:
    rl_model = None
    print("[⚠️] Pas d’agent RL détecté")

class MCTSNode:
    def __init__(self, grid, program=None, parent=None):
        self.grid = grid
        self.program = program or []
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.reward = 0.0

def compute_embedding(grid, cnn_model, size=30):
    padded = np.zeros((size, size))
    padded[:grid.shape[0], :grid.shape[1]] = grid
    inp = padded[np.newaxis, ..., np.newaxis] / 10.0
    return cnn_model.predict(inp, verbose=0)[0]

# === RL suggestion: proposer top N actions les plus probables
def rl_suggest(state_seq, top_n=3):
    if rl_model is None:
        return []
    preds = rl_model.predict(state_seq[np.newaxis, :], verbose=0)[0]
    top_indices = preds.argsort()[-top_n:][::-1]
    return [dsl.primitives[i] for i in top_indices]

# === MCTS avec FAISS + RL suggestions
def mcts_search(input_grid, output_grid, iterations=300):
    root = MCTSNode(input_grid)
    for _ in range(iterations):
        node = root
        while node.children:
            scores = {
                p: c.reward / (c.visits + 1e-6) + 0.7 * np.sqrt(np.log(node.visits + 1) / (c.visits + 1e-6))
                for p, c in node.children.items()
            }
            node = node.children[max(scores, key=scores.get)]

        emb_in = compute_embedding(node.grid, cnn)
        emb_out = compute_embedding(output_grid, cnn)
        target = emb_out - emb_in
        _, indices = index.search(np.array([target]).astype('float32'), k=3)
        faiss_suggestions = [program_map[i] for i in indices[0]]

        # RL suggestions
        state_seq = np.zeros((6,))
        for i, p in enumerate(node.program[-6:]):
            if p in dsl.primitives:
                state_seq[i] = dsl.primitives.index(p)

        rl_suggestions = rl_suggest(state_seq.astype(int), top_n=3)
        all_candidates = list(set(faiss_suggestions + rl_suggestions + random.choices(dsl.primitives, k=3)))
        primitive = random.choice(all_candidates)

        if isinstance(primitive, str):
            new_grid = dsl.apply(node.grid, primitive)
            new_program = node.program + [primitive]
        else:
            new_grid = node.grid.copy()
            for p in primitive:
                new_grid = dsl.apply(new_grid, p)
            new_program = node.program + primitive

        node.children[str(primitive)] = MCTSNode(new_grid, new_program, node)
        sim_grid = new_grid
        sim_program = new_program.copy()

        for _ in range(6 - len(sim_program)):
            p = random.choice(dsl.primitives)
            sim_grid = dsl.apply(sim_grid, p)
            sim_program.append(p)

        error = np.mean((sim_grid - output_grid)**2)
        reward = 1 - error / (np.max(output_grid)**2 + 1e-6)

        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent

    best = max(root.children.values(), key=lambda n: n.visits, default=root)
    return best.program

# === Test d'intégration RL+MCTS
def generate_test_task():
    grid = np.random.randint(0, 10, (10, 10))
    p = random.choice(dsl.primitives)
    out = dsl.apply(grid, p)
    return {"input": [grid], "output": [out], "expected": p}

def main():
    task = generate_test_task()
    inp, out = task["input"][0], task["output"][0]
    print("⏳ Résolution de la tâche avec MCTS+RL...")
    program = mcts_search(inp, out, iterations=300)
    print(f"[✅] Programme trouvé : {program}")
    print(f"[🔍] Primitive attendue : {task['expected']}")

if __name__ == "__main__":
    main()
