.PHONY: test lint typecheck format all

test:
	PYTHONPATH=src python -m pytest tests/ -v

lint:
	ruff check src/ tests/

typecheck:
	PYTHONPATH=src mypy src/

format:
	ruff format src/ tests/
	isort src/ tests/

all: lint typecheck test
