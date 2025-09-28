VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: default setup train predict clean help
default: setup

setup: $(VENV)/bin/python
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

$(VENV)/bin/python:
	python3 -m venv $(VENV)

train: 
	$(PY) src/train.py data/data.csv

predict: 
	$(PY) src/predict.py

evaluate: 
	$(PY) src/evaluate.py

clean:
	rm -rf $(VENV) __pycache__ src/__pycache__ model/*.json plots/*.png plot.png epochs.gif

help:
	@echo "Targets: setup, train, predict, clean"
