#!/usr/bin/env bash

#Enviroment name
VENVNAME=Assignment4_venv

python3 -m venv $VENVNAME

#Activate enviroment
source $VENVNAME/bin/activate

#Upgrade pip
pip install --upgrade pip

#Install from requirements.txt
test -f requirements.txt && pip install -r requirements.txt


#Navigate to src folder
cd src

#Run scripts
python3 lr-mnist.py

python3 nn-mnist.py

#deactivate environment
deactivate
