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
    # add argument about number of epochs with 20 epochs as default
    ap.add_argument("-epo", "--epochs_number", 
                    required=False, 
                    default = 20, 
                    type = int, 
                    help="The number of epochs, the default is 100")
    
    
    args = vars(ap.parse_args())
    
    trs_size = args["train_size"]
    tes_size = args["test_size"]
    epochs_number = args["epochs_number"]
    
    """
    ---------- Neural network model ----------
    """
    # Fetch data
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
     # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # MinMax regularization
    X = ( X - X.min())/(X.max() - X.min()) 
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                         y, 
                                                        train_size = trs_size,
                                                        test_size=tes_size)
    # Convert labels from integers to vectors
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    
    # Train the network
    print("[INFO] training network...")
    # The layers are 32 and 16 and the output is 10
    nn = NeuralNetwork([X_train.shape[1], 32, 16, 10])
    print("[INFO] {}".format(nn))
    nn.fit(X_train, y_train, epochs=epochs_number)
    
    # Evaluate network
    print(["[INFO] evaluating network..."])
    predictions = nn.predict(X_test)
    predictions = predictions.argmax(axis=1)
    print(classification_report(y_test.argmax(axis=1), predictions))

#Define behaviour when called from command line
if __name__ == "__main__":
    main() 
    
