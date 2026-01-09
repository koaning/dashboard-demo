.PHONY: docs

docs:
	uvx marimo -y -q export html-wasm --mode edit dashboard.py -o docs
