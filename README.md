# ARC-2
competition ARC-2 


/workspace/memory_os_ai/Kaggle/
├── venv/                       # Environnement Python virtuel
├── data/
│   ├── synthetic/              # Tâches générées
│   └── arc_ag2/                # Datasets ARC-AGI-2 si dispo
├── models/
│   ├── cnn.h5                  # Modèle sauvegardé
│   └── faiss.index            # Index FAISS
├── metadata/
│   └── dsl_descriptions.json  # Descriptions NLP des primitives
├── src/
│   ├── train.py               # Entraînement CNN + FAISS
│   ├── export.py              # Export des artefacts pour Kaggle
│   ├── solver.py              # MCTS + prédiction
│   ├── dsl.py                 # DSL + metadata
│   ├── spacy_processor.py     # NLP enrichissement
│   └── utils/                 # Helpers génériques
├── submission/
│   └── submission.json
├── kaggle_notebook_template.ipynb
├── requirements.txt
├── Makefile
└── README.md
