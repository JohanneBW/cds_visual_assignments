#!/usr/bin/env python

"""
---------- Import libraries ----------
"""
import os
import sys
sys.path.append(os.path.join(".."))

import argparse

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util
from utils.neuralnetwork import NeuralNetwork

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn import datasets


"""
---------- Main function ----------
"""

def main():
    """
    ---------- Parameters ----------
    """
    # Create an argument parser from argparse
    ap = argparse.ArgumentParser()       
    # add argument about size of training data with 80% as default
    ap.add_argument("-trs", "--train_size", 
                    required=False, default = 0.8, 
                    type = float, 
                    help="The size of the train data as percent, the default is 0.8")
    # add argument about size of test data with 20 % as default
    ap.add_argument("-tes", "--test_size", 
                    required=False, 
                    default = 0.2, 
                    type = float, 
                    help="The size of the test data as percent, the default is 0.2")
    args = vars(ap.parse_args())
    
    trs_size = args["train_size"]
    tes_size = args["test_size"]
    
    """
    ---------- Logistic Regression model ----------
    """
    # Fetch data. When fetching the data like this, the X and y is already defined as the data and the labels. 
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Create training and test dataset. X contains the data and will be split into training and test data. y contains the labels and will split into train and test as well.
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    train_size=trs_size, 
                                                    test_size=tes_size)
    # Scaling the features to a value between 0 and 1
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0
    # Train a logistic regression model.
    clf = LogisticRegression(penalty='none', 
                         tol=0.1, 
                         solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, y_train)
    # Check the accurcy
    y_pred = clf.predict(X_test_scaled)
    cm = metrics.classification_report(y_test, y_pred)
    print(cm)

#Define behaviour when called from command line
if __name__ == "__main__":
    main()   
        
