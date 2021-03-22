#!/usr/bin/env bash

VENVNAME=Assignment4_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

test -f requirements.txt && pip install requirements.txt

deactivate
echo "build $VENVNAME"