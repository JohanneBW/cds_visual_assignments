'''
-------------- Import libraries:---------------
'''
# Libraries
import pandas as pd
import os
import wget
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path

# sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     AveragePooling2D, 
                                     Activation,
                                     Dropout,
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

'''
---------------------- Function for plotting and save history:------------------- 
'''
#Define function for plotting and save history 
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
    #Save the history as model_performance.png
    fig.savefig("../output/model_performance.png")

'''
-------------- Main function:---------------
'''

def main():
    
    '''
    -------------- Import data:---------------
    '''
    df_path = os.path.join("..", "data", "sorted_df.csv")
    
    sorted_df = pd.read_csv(df_path)
    
    '''
    ---------------------- Append categories to list:------------------- 
    '''
    #Iterating through the unique cathegories.
    cathegories = []
    for cat in sorted_df["Genre"].unique():
    #We find the Genres, that is only described by cathegory
        # "|" indicate that the movie has more than one genre.
        if "|" not in str(cat):
            cathegories.append(cat)

    print(f"The cathegories in the data set: {cathegories}")
    
    '''
    ---------------------- Convert the images into arrays and create data frame for image data:------------------- 
    '''
    #Define the image path where the posters are located
    image_path = os.path.join("..", "data", "Poster_data")

    #Create empty list, where the arrays will be storred
    np_images = pd.DataFrame(columns=["Title", "np_array"])

    #Convert every image in the image_path to numpy arrays
    for image in Path(image_path).glob("*jpg"):
        image_array = cv2.imread(str(image)) 

        #Append the image data to the cloumns in the data frame
        #We split the title string so only the actual name will be in the data frame
        np_images = np_images.append({'Title' : str(image)[20:len(str(image))-4],
                                      'np_array' : image_array}, 
                                       ignore_index=True)
    
    
    '''
    ---------------------- Merge data frames:------------------- 
    '''
    #Sorting the dataframe with image date in alphabetical order
    sorted_np_images_df = np_images.sort_values(by=['Title']).reset_index(drop=True)
    
    
    #Sorting the dataframe with the genres in alphabetical order
    alpha_df = sorted_df.sort_values(by=['Title']).reset_index(drop=True)
    
    
    #Merge the two dataframes by title. The titles are the same in both data frames.
    #We do this to make sure the posters and the genres are in the same order
    merged_df = pd.merge(alpha_df, sorted_np_images_df, on="Title")
    
    
    #Dropping duplicates in the merged dataset.
    merged_df = merged_df.drop_duplicates(subset=['Title']).reset_index(drop=True)
    
    
    #Make the arrays in the data frame into a list again.
    #The model does not work, when np_arrays come from a dataframe
    np_list = merged_df["np_array"].to_list()
    
    
    
    '''
    ---------------------- Define traning and test data and labels:------------------- 
    '''
    # X represent the data and y the labels. The posters will be the data and the genres will be the labels   
    #Define the test and traing data and size
    X_train, X_test, y_train, y_test = train_test_split(np_list, 
                                                        merged_df["Genre"],
                                                        test_size= 0.25,
                                                        random_state= 9)
    
    #Convert the data into arrays. We have to convert the lists into arrays again to avoid an error 
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    #Flatten the data
    X_train = X_train/255.0
    X_test = X_test/255.0
    
    
    #Binarize the labels.
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
      
      
    '''
    ---------------------- Define and train CNN model:------------------- 
    '''   
    '''We used the GridSearchCV.py script which is a script we made to estimate the best parameters for our model.
    We found out that the optimizer SGD, a batch size of 10 and 10 epochs would give the best results.'''
    
    #Define model as being Sequential and add layers
    #The Sequential model allow us to group a linear stack of layers in the model.
    model = Sequential()
    
    #First set of CONV => RELU => POOL
    #The filter is a mesure of how much the image is split up, and how many times the kernal runs through each image.
    model.add(Conv2D(32, (3, 3),  #The filter is set to the input 50 and the kernel to 3x3
                    padding="same", 
                    input_shape=(268, 182, 3))) #The shape of all the posters with height, width and dimensions
    model.add(Activation("relu"))
    model.add(AveragePooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    #Second set of CONV => RELU => POOL
    model.add(Conv2D(16, (5, 5), #The filter is set to 16 and the kernel to 5x5
                     padding="same"))
    model.add(Activation("relu"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Add dropout layer to prevent overfitting
    model.add(Dropout(0.15))
    
    # FC => RELU
    model.add(Flatten())
    model.add(Dense(12)) #The filter is set to 12
    model.add(Activation("relu"))

    # Softmax classifier. 
    # We use softmax as the activation for the last layer because we are interested in the categorical probabilities.
    model.add(Dense(3))  #The filter is set to 3 which is the number of unique labels
    model.add(Activation("softmax"))

    # Compile model
    # We use the stochastic gradient descent optimizer
    opt = SGD(lr=0.01)
    # We use categorical crossentropy as loss function because we are dealing with more than two labels. 
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Save model summary as model_architecture.png
    plot_model(model, to_file = "../output/model_architecture.png", show_shapes=True, show_layer_names=True)

    # Train the model
    H = model.fit(X_train, y_train, 
                  validation_data=(X_test, y_test), 
                  batch_size=10,
                  epochs=10,
                  verbose=1)

    #View the summary
    model.summary()
    
    
    # Plot and save history via the earlier defined function
    plot_history(H, 10)
    

    # Print the classification report
    predictions = model.predict(X_test, batch_size=10)
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=cathegories))

#Define behaviour when called from command line
    
if __name__ == "__main__":
    main() 