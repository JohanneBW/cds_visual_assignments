#!/usr/bin/env bash

VENVNAME=VA_Project_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

#Install the libraries in the requirements file
test -f requirements.txt && pip install requirements.txt

#Install the libraries directly from bash script because of version error
pip3 install pandas
pip3 install wget
pip3 install numpy
pip3 install opencv-python
pip3 install Path
pip3 install matplotlib
pip3 install sklearn
pip3 install tensorflow
pip3 install pydot


#Navigate to src folder
cd src

#Run scripts
python3 data_cleaning.py
python3 CNN_Genre_test.py

#deactivate environment
deactivate
