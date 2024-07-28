SHELL := /bin/bash

.PHONY: help

help:
	@echo "----------------------------------------------------"
	@echo "test-all          : run all test cases"
	@echo "----------------------------------------------------"

t1:
	TIGTEST=1 python test.py

t2:
	TIGTEST=2 python test.py

test-all: t1 t2
	@echo "Done!"
