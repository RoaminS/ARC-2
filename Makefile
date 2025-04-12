train:
	python src/train.py

export:
	python src/export.py

solve:
	python src/solver.py

spacy:
	python -m spacy download en_core_web_sm
	python src/spacy_processor.py

all: train export solve
