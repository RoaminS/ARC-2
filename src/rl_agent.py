import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from src.dsl import dsl
import random

PRIM_COUNT = len(dsl.primitives)
SEQ_LEN = 6

# === RL Policy Model (LSTM Based) ===
def build_policy_model():
    inp = Input(shape=(SEQ_LEN,))
    x = Embedding(PRIM_COUNT, 64)(inp)
    x = LSTM(128, return_sequences=False)(x)
    out = Dense(PRIM_COUNT, activation='softmax')(x)
    return Model(inp, out)

policy_model = build_policy_model()
optimizer = tf.keras.optimizers.Adam(1e-3)

# === Grille padding ===
def resize(grid, size=30):
    padded = np.zeros((size, size))
    padded[:grid.shape[0], :grid.shape[1]] = grid
    return padded

# === Reward inversé à l'erreur MSE ===
def compute_reward(grid_pred, grid_target):
    error = np.mean((grid_pred - grid_target)**2)
    max_val = np.max(grid_target) + 1e-6
    reward = 1 - error / (max_val ** 2 + 1e-6)
    return reward

# === Entraînement RL sur une génération de tâche ===
def train_agent(epochs=1000):
    for episode in range(epochs):
        # Génération tâche aléatoire
        grid = np.random.randint(0, 10, (10, 10))
        target_prim = random.choice(dsl.primitives)
        target_grid = dsl.apply(grid, target_prim)

        state = np.zeros((1, SEQ_LEN))
        actions = []
        probs = []

        current_grid = grid.copy()

        for t in range(SEQ_LEN):
            logits = policy_model.predict(state, verbose=0)[0]
            action = np.random.choice(PRIM_COUNT, p=logits)
            primitive = dsl.primitives[action]
            current_grid = dsl.apply(current_grid, primitive)
            state[0, t] = action
            actions.append(action)
            probs.append(logits[action])

        reward = compute_reward(current_grid, target_grid)

        # === Policy gradient update ===
        with tf.GradientTape() as tape:
            logits = policy_model(state, training=True)
            action_masks = tf.one_hot(actions, PRIM_COUNT)
            log_probs = tf.math.log(tf.reduce_sum(logits * action_masks, axis=1) + 1e-6)
            loss = -tf.reduce_mean(log_probs * reward)

        grads = tape.gradient(loss, policy_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))

        if episode % 100 == 0:
            print(f"[🎯] Épisode {episode} — Reward: {reward:.4f} — Loss: {loss:.4f}")

    # Sauvegarde du modèle
    policy_model.save("models/rl_policy_model.h5")
    print("[✅] Agent RL sauvegardé dans models/rl_policy_model.h5")

if __name__ == "__main__":
    train_agent()
