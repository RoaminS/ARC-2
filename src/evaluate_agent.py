import numpy as np
from src.solver import mcts_search
from src.dsl import dsl
import time

def generate_task():
    grid = np.random.randint(0, 10, (10, 10))
    p = np.random.choice(dsl.primitives)
    out = dsl.apply(grid, p)
    return {"input": [grid], "output": [out], "primitive": p}

def compare_grids(pred, target):
    return np.array_equal(pred, target)

def run_evaluation(n_tasks=100, iterations=300):
    success = 0
    total_time = 0
    all_logs = []

    for i in range(n_tasks):
        task = generate_task()
        inp, expected = task["input"][0], task["output"][0]

        print(f"⚙️  Tâche {i+1}/{n_tasks} — Primitive cible : {task['primitive']}")

        start = time.time()
        program = mcts_search(inp, expected, iterations=iterations)
        end = time.time()
        elapsed = end - start
        total_time += elapsed

        grid = inp.copy()
        for p in program:
            grid = dsl.apply(grid, p)

        ok = compare_grids(grid, expected)
        result = "✅" if ok else "❌"
        print(f"{result} | Programme généré : {program} | Temps : {elapsed:.2f}s")

        all_logs.append({
            "success": ok,
            "program": program,
            "expected": task["primitive"],
            "time": elapsed
        })

        if ok:
            success += 1

    precision = success / n_tasks * 100
    avg_time = total_time / n_tasks

    print("\n=== 🔎 Résultats finaux ===")
    print(f"🎯 Précision : {precision:.2f}%")
    print(f"🕒 Temps moyen / tâche : {avg_time:.2f}s")

    return precision, all_logs

if __name__ == "__main__":
    run_evaluation(n_tasks=200, iterations=300)
