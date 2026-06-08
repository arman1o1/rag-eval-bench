.PHONY: setup ingest evaluate ablation dashboard test lint clean

setup:
	uv sync
	docker compose up -d

ingest:
	uv run python -m src.ingestion.pipeline --corpus-dir data/corpus --strategy all

ingest-one:
	uv run python -m src.ingestion.pipeline --corpus-dir data/corpus --strategy $(STRATEGY)

evaluate:
	uv run python -m src.evaluation.benchmark --configs all --output results/

ablation:
	uv run python -m src.evaluation.ablation --output results/ablations/

dashboard:
	uv run streamlit run src/dashboard/app.py

test:
	uv run pytest tests/ -v

test-integration:
	uv run pytest tests/ -v -m integration

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/

clean:
	docker compose down -v
	rm -rf results/ .pytest_cache/ __pycache__/
