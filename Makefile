.PHONY: docs 

docs: 
	uvx marimo -y -q export html-wasm dashboard.py -o docs

