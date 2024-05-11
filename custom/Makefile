SHELL := /usr/bin/env bash

PROJECT_ROOT = $(shell git rev-parse --show-toplevel)
# FILE_PATH=$(shell git ls-files --full-name Makefile | sed -e "s/Makefile//g")
# multi-repo structure
DOCKER_IMAGE_NAME=$(shell basename $(PROJECT_ROOT))
# mono-repo structure
# DOCKER_IMAGE_NAME=$(shell echo $(FILE_PATH) | sed -e "s/\//-/g" | rev | cut -c2- | rev )

# git tag | tail -n 1
GIT_TAG=$(shell git describe --abbrev=0 --tags)
GIT_SHA=$(shell git rev-parse HEAD)
DOCKER_IMAGE_VERSION=$(GIT_SHA)

PYTHON_VERSION=3.12

ifeq ($(BUILD_ENV), "RELEASE")
	DOCKER_IMAGE_VERSION=$(GIT_TAG)
endif

.PHONY: build_release_docker_image
build_release_docker_image: build_ci_docker_image
	@echo "Building Docker image..."
	docker build --no-cache -f ./Dockerfile --build-arg PYTHON_VERSION=$(PYTHON_VERSION) --target release -t $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION) .

.PHONY: run_docker_image
run_docker_image: build_release_docker_image
	@echo "Building Docker image..."
	docker run -it --rm --name "$(DOCKER_IMAGE_NAME)" $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)

.PHONY: build_ci_docker_image
build_ci_docker_image:
	@echo "Building Docker image..."
	docker build --no-cache -f ./Dockerfile --build-arg PYTHON_VERSION=$(PYTHON_VERSION) --target ci -t $(DOCKER_IMAGE_NAME)-ci:$(DOCKER_IMAGE_VERSION) .

.PHONY: run_docker_bash
run_docker_bash:
	@echo "Running python:$(PYTHON_VERSION) Docker"
	docker run -it --rm --name "$(DOCKER_IMAGE_NAME)"-ci -v "$(PROJECT_ROOT)"/:/app python:$(PYTHON_VERSION) bash
	# cd /app
	# make dev_dependencies
	# make activate_pipenv
	# make format

.PHONY: clean
clean:
	docker rmi -f $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)
	docker rmi -f $(DOCKER_IMAGE_NAME)-ci:$(DOCKER_IMAGE_VERSION)

.PHONY: upgrade_pipenv
upgrade_pipenv:
	pip install pipenv --upgrade
	pipenv lock

# install Python Dev dependencies
.PHONY: dev_dependencies
dev_dependencies:
	pip install pipenv
	PIPENV_VENV_IN_PROJECT=1 pipenv install --dev

# activate virtual environment
.PHONY: activate_pipenv
activate_pipenv:
	# . $(pipenv --venv)/bin/activate
	# install the packages specified in the Pipfile:
	# pipenv install
	# to activate this project's virtualenv, run pipenv shell
	pipenv shell
	# alternatively, run a command inside the virtualenv with pipenv run.
	# verify that the packages are installed correctly:
	# pipenv --venv

## -- CI LOCAL TARGETS --

.PHONY: format
format:
	@echo "Running fmt..."
	echo "Removing unused imports"
	autoflake --in-place --recursive --remove-unused-variables --exclude .venv .
	echo "Running isort to sort imports"
	isort .
	echo "Running black formatter"
	black --line-length 140 .

.PHONY: ci
ci: lint test coverage

.PHONY: lint
lint:
	@echo "Running flake8 linter..."
	if [[ "${shell flake8 --config=.flake8 ./src 2> /dev/null; echo $$?}" == "1" ]]; then echo "Fix linting" && exit 1 ; fi
	flake8 --config=.flake8 ./src
	@echo "Running black formatter"
	if [[ "${shell black --line-length 140 . --check --diff; echo $$?}" == "1" ]]; then echo "Fix formatting" && exit 1 ; fi
	@echo "Running isort sort imports"
	if [[ "${shell isort . --check-only 2> /dev/null; echo $$?}" == "1" ]]; then echo "Remove unused imports" && exit 1 ; fi
	@echo "Running autoflake unused imports"
	if [[ "${shell autoflake --check --recursive --exclude .venv . 2> /dev/null; echo $$?}" == "1" ]]; then echo "Remove unused imports" && exit 1 ; fi

.PHONY: test
test:
	@echo "Running tests..."
	pytest tests -v --log-cli-level=DEBUG --durations=10 --cov-report term --cov-report html:htmlcov --cov-report xml --cov-fail-under=90

.PHONY: coverage
coverage:
	@echo "Running code coverage..."
	coverage run --omit="*/test*" -m pytest tests/
	coverage report -m

