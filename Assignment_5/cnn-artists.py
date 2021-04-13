#!/usr/bin/env python

"""
---------- Import libraries ----------
"""

# data tools
import os
import sys
#sys.path.append(os.path.join("..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import cv2
import re
import shutil
from shutil import copyfile
import random

import argparse

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K   


"""
---------- Main function ----------
"""

def main():

    """
    ---------- Parameters ----------
    """
    # Create an argument parser from argparse
    ap = argparse.ArgumentParser()       
    # add argument about the number of epochs when training the model. The default value is 20 epochs.
    ap.add_argument("-epo", "--epochs_number", 
                    required=False, 
                    default = 20, 
                    type = int, 
                    help="The number of epochs, the default is 20")
    
    args = vars(ap.parse_args())
    
    epochs_number = args["epochs_number"]
    
    """
    ---------- Sample data ----------
    """
    # This section creates a smaller sample data for the training data. I have uploaded the samples so the code for this will not be needed. 
    # The folder which contains the sub directories 
    #source_dir = 'data/archive/training/training/'

    # List sub directories 
    #for root, dirs, files in os.walk(source_dir):
        # Iterate through them
        #for i in dirs: 
            # Create a new folder with the name of the iterated sub dir
            #path = 'data/small_training/' + "%s/" % i
            #os.makedirs(path)
            # Take random sample, here 3 files per sub dir
            #filenames = random.sample(os.listdir('data/archive/training/training/' + "%s/" % i ), 40)
            #Copy the files to the new destination
            #for j in filenames:
                #shutil.copy2('data/archive/training/training/' + "%s/" % i  + j, path)
    
    # Create a smaller sample data for the test/validation data 
    # Folder which contains the sub directories we want to copy
    #val_source_dir = 'data/archive/validation/validation/'
    
    # List sub directories 
    #for root, dirs, files in os.walk(val_source_dir):
    # Iterate through them
        #for i in dirs: 
            # Create a new folder with the name of the iterated sub dir (small_training)
            #path = 'data/small_validation/' + "%s/" % i
            #os.makedirs(path)
            # Take random sample, here 10 files per sub dir
            #filenames = random.sample(os.listdir('data/archive/validation/validation/' + "%s/" % i ), 10)
            # Copy the files to the new destination
            #for j in filenames:
                #shutil.copy2('data/archive/validation/validation/' + "%s/" % i  + j, path)
    
    """
    ---------- Find and create labels ----------
    """
    # Path to training folder with painters
    training_dir = os.path.join("data", "small_training")

    # Names as a string
    label_names = []
    # Training labels
    y_train = []

    # We find the painters names using regex. These will be our labels 
    i = 0
    for folder in Path(training_dir).glob("*"):
        # Findall returns a list
        painter = re.findall(r"(?!.*/).+", str(folder))
        # To acces the values in the list we use the index[0] 
        label_names.append(painter[0])

        # Append the indexes to the y_train list
        for img in folder.glob("*"):
            y_train.append(i)

        i +=1
    
    # Labels for validation
    # Path to training folder with painters
    validation_dir = os.path.join("data", "small_validation")

    # Test labels
    y_test = []

    i = 0
    # We append the indexes to the y_test list
    for folder in Path(validation_dir).glob("*"):
        for img in folder.glob("*"):
            y_test.append(i)

        i +=1
    
    # Integers to one-hot vectors. Transform the labels into a binary.
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
   
    """
    ---------- Resize the images ----------
    """
    # Resize the training data
    # The path were the images are located
    filepath = os.path.join("data", "small_training")

    X_train=[]

    # for each image in the folders, resize it
    for folder in Path(filepath).glob("*"):
        for file in Path(folder).glob("*"):
            image_open = cv2.imread(str(file))
            # The new dimensions 
            dim = (120, 120)
            # resize the image
            resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
            # append the resized images to the X_train list
            X_train.append(resize_image.astype("float")/255.)
    
    # Resize the test data
    # The path were the images are located
    filepath = os.path.join("data", "small_validation")

    X_test=[]

    # For each image in the folders, resize it
    for folder in Path(filepath).glob("*"):
        for file in Path(folder).glob("*"):
            image_open = cv2.imread(str(file))
            # The new dimensions 
            dim = (120, 120)
            # Resize the image 
            resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
            # Append the resized images to the X_train list
            X_test.append(resize_image.astype("float")/255.)

    # Convert to numpy array
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    """
    ---------- LeNet model ----------
    """
    # Plot history
    def plot_history(H, epochs):
        # Visualize performance
        plt.style.use("fivethirtyeight")
        fig = plt.figure()
        plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig("model_performance.png")
     
    # Define model
    model = Sequential()
    # First set of CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), 
                     padding="same", 
                     input_shape=(120, 120, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))
    # Second set of CONV => RELU => POOL
    model.add(Conv2D(50, (5, 5), 
                     padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))
    # FC => RELU
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    # Softmax classifier
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    # Compile model
    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    
    # Model summary
    plot_model(model, to_file = "model_architecture.png", show_shapes=True, show_layer_names=True)
    
    # Train model
    H = model.fit(X_train, y_train, 
                  validation_data=(X_test, y_test), 
                  batch_size=32,
                  epochs=epochs_number,
                  verbose=1)
    
    # Plot history
    plot_history(H, epochs_number)
    
    # Print the classification report
    predictions = model.predict(X_test, batch_size=32)
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_names))
    

#Define behaviour when called from command line
if __name__ == "__main__":
    main()  

