#!/bin/bash

export PYTHONPATH="."

python3 -m pytest \
        --cov-config tests/.coveragerc \
        --cov-report html \
        --cov bandstructure \
        --cov tests \
        tests

rm -f .coverage .coverage.*
