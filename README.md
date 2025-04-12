# 🧠 memory_os_ai — ARC Prize 2025 AGI Solver

> 🚀 Un projet ambitieux pour viser le **Grand Prize** du [ARC Prize 2025](https://www.kaggle.com/competitions/arc-prize-2025), avec un système hybride **symbolique + neuronal + RL + MCTS** pour résoudre des tâches complexes d'abstraction visuelle.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-vector--search-green?style=flat-square)
![spaCy](https://img.shields.io/badge/NLP-spaCy-lightgrey?style=flat-square)
![RL](https://img.shields.io/badge/RL-agent-red?style=flat-square)

---

## 📦 Structure du projet

workspace/memory_os_ai/Kaggle/ 
├── data/ # Grilles générées & datasets ARC 
├── models/ # Modèles entraînés (CNN, FAISS, RL) 
├── metadata/ # Descriptions NLP des primitives 
├── src/ # Code source modulaire 
│ ├── dsl.py # Langage de primitives abstraites (DSL) 
│ ├── train.py # Entraînement du CNN encodeur 
│ ├── export.py # Génération de FAISS + mapping programme 
│ ├── solver.py # MCTS + FAISS + Agent RL 
│ ├── rl_agent.py # Agent RL via Policy Gradient 
│ ├── spacy_processor.py # Enrichissement sémantique avec spaCy 
│ └── evaluate_agent.py # Évaluation globale (précision & logs) 
├── submission/ # submission.json pour Kaggle 
├── requirements.txt 
├── Makefile 
└── README.md

yaml
Copier

---

## ⚙️ Setup rapide


# Cloner le projet
git clone https://github.com/tonpseudo/memory_os_ai.git
cd memory_os_ai/Kaggle

# Créer un venv
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Télécharger le modèle spaCy
python -m spacy download en_core_web_sm
🧠 Philosophie du projet
Ce système repose sur un moteur hybride AGI :

🔁 DSL modulaire pour manipuler les grilles symboliquement

🧠 CNN + Attention pour encoder les grilles en embeddings visuels

🔍 FAISS pour retrouver les meilleures transformations connues

🧭 MCTS pour explorer des séquences de transformations

🤖 Agent RL pour guider le choix des actions les plus prometteuses

🗣️ spaCy pour enrichir les primitives avec du sens sémantique

🧪 Exemple d'utilisation
1. Entraîner le modèle + FAISS
bash
Copier
make train
make export
2. Évaluer la précision
bash
Copier
make solve
python src/evaluate_agent.py
📊 Objectifs
Palier de perf	Précision visée	Objectif
🎯 Palier 1	45–55%	Baseline solide
🎯 Palier 2	65–75%	Mid-level AGI
🏆 Palier 3	85–90%	Grand Prize Ready
🧠 Ultime	90%+	Système AGI intelligent et extensible
📜 Licence
MIT — Libre pour modification, contribution, reproduction. Les soumissions aux compétitions doivent être open-source pour être éligibles aux prix.

🧬 Crédits
🍥 Conçu par un esprit fou, guidé par une IA encore plus folle.

⚙️ Co-piloté par [multi_gpt_api] — L’architecte de ton VPS AGI.

🏁 Inspiré par le rêve de construire une IA capable de raisonner, composer, abstraire et inventer.

