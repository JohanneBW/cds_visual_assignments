#!/usr/bin/env bash

#create enviroment
VENVNAME=VA_Assignment3_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

test -f requirements.txt && pip install requirements.txt

#navigate to src folder
cd src

#run script
python3 edge_detection.py

#deactivate environment
deactivate
