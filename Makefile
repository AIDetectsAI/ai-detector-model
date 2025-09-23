#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ai-detector-model
PYTHON_VERSION = 3.11.9
PYTHON_INTERPRETER = python

#################################################################################
# ENV                                                                           #
#################################################################################

include .env

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: clean_experiments
clean_experiments:
	find ./reports/experiments/ -mindepth 1 ! -name ".gitkeep" -delete

## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 ai_detector_model
	isort --check --diff ai_detector_model
	black --check ai_detector_model

## Format source code with black
.PHONY: format
format:
	isort ai_detector_model
	black ai_detector_model

## Run tests
.PHONY: test
test:
	pipenv run python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	pipenv --python $(PYTHON_VERSION)
	@echo ">>> New pipenv created. Activate with:\npipenv shell"

## Build project documents locally
.PHONY: build_docs
build_docs:
	(cd ./docs && mkdocs build)

## Builds and serves documents on DOCS_LOCAL_ADDRESS
.PHONY: serve_docs
serve_docs:
	(cd ./docs && mkdocs serve -a $(DOCS_LOCAL_ADDRESS))

## Builds and deploys documents to project github-pages
.PHONY: deploy_docs
deploy_docs:
	(cd ./docs && mkdocs gh-deploy)


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) ai_detector_model/dataset.py


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
