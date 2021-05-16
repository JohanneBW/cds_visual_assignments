#!/usr/bin/env bash

VENVNAME=VA_Assignment5_venv

python3 -m venv $VENVNAME

source $VENVNAME/bin/activate
pip install --upgrade pip

#Install packages, error when installing from requirements.txt
pip3 install numpy
pip3 install pandas
pip3 install matplotlib
pip3 install opencv-python
pip3 install sklearn
pip3 install tensorflow
pip3 install pydot
pip3 install google

#Run script
cd src
python3 cnn-artists.py

#deactivate environment
deactivate
