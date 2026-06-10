.PHONY: build style-check style quality test

check_dirs := src tests

# uvx with fallback, e.g.: 1. `uvx ruff check` 2. `ruff check`
TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")
RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run --env-file .env --no-sync" || echo "")

all: style quality

style-check:
	$(TOOL) ruff check $(check_dirs)
	$(TOOL) ruff format --check $(check_dirs)

style:
	$(TOOL) ruff check $(check_dirs) --fix
	$(TOOL) ruff format $(check_dirs)

quality:
	$(TOOL) ty check $(check_dirs)

test:
	$(RUN) pytest tests -rA

appcode:
	$(RUN) uvicorn postprocess.verifier.code.api:app --host 0.0.0.0 --port 1234 --reload