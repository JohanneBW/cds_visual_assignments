#!/usr/bin/env bash

#Enviroment name
VENVNAME=Assignment4_venv

#Activate enviroment
source $VENVNAME/bin/activate

#Upgrade pip
pip install --upgrade pip

#Problems when installing from requirements.txt
test -f requirements.txt && pip install -r requirements.txt


#Navigate to src folder
cd src

#Run scripts
python3 lr-mnist.py $@

python3 nn-mnist.py $@

#deactivate environment
deactivate
