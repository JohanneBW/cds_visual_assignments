#!/usr/bin/env bash

VENVNAME=VA_Project_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

#Install the libraries in the requirements file
test -f requirements.txt && pip install requirements.txt

echo "build $VENVNAME"

#Navigate to src folder
cd src

#Run script
python3 CNN_Genre.py

#deactivate environment
deactivate