VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: default setup train predict clean fclean help
default: setup

setup: $(VENV)/bin/python
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

$(VENV)/bin/python:
	python3 -m venv $(VENV)

train: 
	@$(PY) src/train.py $(filter-out $@,$(MAKECMDGOALS))

predict: 
	$(PY) src/predict.py

evaluate: 
	@$(PY) src/evaluate.py $(filter-out $@,$(MAKECMDGOALS))

clean:
	rm -rf __pycache__ src/__pycache__ model/*.json plots/*.png plot.png epochs.gif

fclean: clean
	rm -rf $(VENV)

help:
	@echo "Targets: setup, train, predict, evaluate, clean, fclean"
	@echo "Usage: make train data/data.csv"
	@echo "Usage: make evaluate data/data.csv"

%:
	@:
