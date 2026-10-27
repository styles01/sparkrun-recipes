# The TandemLLM manual.
#   make            regenerate the reference tables and build tandemllm-manual.pdf
#   make tables     only the tables (manual/generated, manual/data)
#   make check      exit 1 when a generated table is stale against the code
#   make clean      remove LaTeX's auxiliary files
PYTHON ?= python3

all: tables pdf

tables:
	$(PYTHON) tools/gen_reference.py

check:
	$(PYTHON) tools/gen_reference.py --check

pdf:
	latexmk -lualatex tandemllm-manual.tex

clean:
	latexmk -C tandemllm-manual.tex

.PHONY: all tables check pdf clean
