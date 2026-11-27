.PHONY: help check test
.DEFAULT_GOAL := help

# Read here, before any include, so it names this file even when MAKEFILES loads others.
THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))

# The Check workflow runs make check and make test on every pull request.
# Replace each recipe below once the project has something to lint or test.

help: ## List the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf " %-6s %s\n", $$1, $$2}' $(THIS_MAKEFILE)

check: ## Lint the code
	@echo ' Nothing to check yet.'

test: ## Run the tests
	@echo ' Nothing to test yet.'
