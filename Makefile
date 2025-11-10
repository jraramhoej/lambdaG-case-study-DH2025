.PHONY: lock sync install test clean

# Generate uv.lock from pyproject.toml
lock:
	uv lock

# Sync venv with lockfile (create or update)
sync:
	uv sync --all-extras

# Install the package in editable mode
install: sync
	uv pip install -e .

# Run tests
test:
	uv run pytest

# Clean generated files
clean:
	rm -rf .venv __pycache__ **/__pycache__ *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
