SHELL := /usr/bin/env bash

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
