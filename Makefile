.PHONY: help check test
.DEFAULT_GOAL := help

# Read here, before any include, so it names this file even when MAKEFILES loads others.
THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))

# The Check workflow runs make check and make test on every pull request.
# Locally, point PYTHON at the dev environment: make check PYTHON=.venv/bin/python
PYTHON ?= python3

help: ## List the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf " %-6s %s\n", $$1, $$2}' $(THIS_MAKEFILE)

check: ## Lint the code
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	docker compose -f compose.yaml config -q

test: ## Run the tests
	$(PYTHON) -m pytest
