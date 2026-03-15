#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ai-detector-model
PYTHON_VERSION = 3.12.12
PYTHON_INTERPRETER = python

#################################################################################
# ENV                                                                           #
#################################################################################

include .env

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## create python environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)

## sync dependencies
.PHONY: requirements
requirements:
	uv sync --all-groups

## configure pre-commit
.PHONY: precommit
precommit:
	uv run pre-commit

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: clean_experiments
clean_experiments:
	find ./reports/experiments/ -mindepth 1 ! -name ".gitkeep" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format

## Run tests
.PHONY: test
test:
	uv run pytest tests

## Build project documents locally
.PHONY: build_docs
build_docs:
	(cd ./docs && uv run mkdocs build)

## Builds and serves documents on DOCS_LOCAL_ADDRESS
.PHONY: serve_docs
serve_docs:
	(cd ./docs && uv run mkdocs serve -a $(DOCS_LOCAL_ADDRESS))

## Builds and deploys documents to project github-pages
.PHONY: deploy_docs
deploy_docs:
	(cd ./docs && uv run mkdocs gh-deploy)

## Starts serving model, use HOST= and/or PORT= parameters to specify, e.g. make server HOST=127.0.0.5 PORT=2005; default 127.0.0.1:8000
.PHONY: server
HOST ?= 127.0.0.1
PORT ?= 8000
server:
	uv run uvicorn ai_detector_model.api.api:app --reload --host $(HOST) --port $(PORT)

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	uv run ai_detector_model/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
