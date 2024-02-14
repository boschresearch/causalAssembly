#Makefile for causalAssembly

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(notdir $(patsubst %/,%,$(dir $(MAKEFILE_PATH))))

VENV_NAME := venv_${CURRENT_DIR}
PYTHON=${VENV_NAME}/bin/python

sync-venv:
	: # Create or update default virtual environment to latest pinned
	: # dependencies
	test -d $(VENV_NAME) || \
		python3.10 -m virtualenv $(VENV_NAME); \
		${PYTHON} -m pip install -U pip; \
		${PYTHON} -m pip install pip-tools
	. $(VENV_NAME)/bin/activate && pip-sync requirements_dev.txt
	#. $(VENV_NAME)/bin/activate && pip install --no-deps -e .

requirements:
	: # Update requirements_dev.txt if only new library is added
	: # Assumes virtual environment with pip-tools installed is activated
	pip-compile --extra dev -o requirements_dev.txt pyproject.toml --annotation-style line --no-emit-index-url --no-emit-trusted-host --allow-unsafe --resolver=backtracking


update-requirements:
	: # Update requirements_dev.txt if dependencies should be updated
	: # Assumes virtual environment with pip-tools installed is activated
	pip-compile --extra dev -o requirements_dev.txt pyproject.toml --annotation-style line --no-emit-index-url --no-emit-trusted-host --allow-unsafe --resolver=backtracking --upgrade
