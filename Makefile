init:
	@command -v rye >/dev/null 2>&1 || (echo "rye not installed. Installing..." && curl -sSf https://rye.astral.sh/get | bash && echo "rye is installed!")
	rye sync
	rye run pre-commit install

format:
	@echo "Formatting code..."
	rye fmt

format-check:
	rye fmt --diff

lint:
	@echo "Linting code..."
	rye run pyright --project src
	rye check

test:
	@echo "Running tests..."
	rye run pytest tests/${target} -vv -cov=config_manager --cov-config=.coveragerc --cov-report=html --cov-report=term-missing --cov-fail-under=100
	coverage-badge -fo coverage.svg
