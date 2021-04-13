#!/usr/bin/env bash

VENVNAME=Assignment4_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

test -f requirements.txt && pip install requirements.txt

echo "build $VENVNAME"

#run the program
python3 cnn-artists.py $@

deactivate
