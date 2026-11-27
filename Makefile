.PHONY: help check test
.DEFAULT_GOAL := help

# Read here, before any include, so it names this file even when MAKEFILES loads others.
THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))

# The Check workflow runs make check and make test on every pull request.
# Uses the dev environment in .venv when there is one (README, Development), else python3,
# where CI installs the tools. Override with PYTHON=...
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

help: ## List the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf " %-6s %s\n", $$1, $$2}' $(THIS_MAKEFILE)

check: ## Lint the code
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

test: ## Run the tests
	$(PYTHON) -m pytest
