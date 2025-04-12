import spacy
import json
import os
import pickle

nlp = spacy.load("en_core_web_sm")

# Dummy descriptions (à enrichir au fil du temps)
primitive_descriptions = {
    "prim_001": "rotate the grid 90 degrees clockwise",
    "prim_075": "replace color 1 with color 2",
    "prim_101": "increment all non-zero values by 1",
    "prim_199": "mirror left half to right",
}

primitive_meta = {}
for pid, desc in primitive_descriptions.items():
    doc = nlp(desc)
    primitive_meta[pid] = {
        "text": desc,
        "lemmas": [t.lemma_ for t in doc if not t.is_stop],
        "embedding": doc.vector.tolist(),
        "pos": [t.pos_ for t in doc],
        "verbs": [t.text for t in doc if t.pos_ == "VERB"]
    }

os.makedirs("metadata", exist_ok=True)

with open("metadata/primitive_meta.json", "w") as f:
    json.dump(primitive_meta, f, indent=2)

print("[✅] spaCy NLP enrichi & sauvegardé dans metadata/primitive_meta.json")
