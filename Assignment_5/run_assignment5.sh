#!/usr/bin/env bash

VENVNAME=VA_Assignment5_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

test -f requirements.txt && pip install requirements.txt

echo "build $VENVNAME"

#Run scripts
python3 cnn-artists.py

#deactivate environment
deactivate