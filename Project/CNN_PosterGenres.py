
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
import matplotlib as plt

# sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K  

'''
-------------- Main function:---------------
'''
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
    ---------- Data wrangling: ----------
    """
    #Path to data
    input_file = os.path.join("data", "MovieGenre.csv")

    #Reading data
    data = pd.read_csv(input_file, encoding = "ISO-8859-1")

    #Remove the NANs
    data = data.dropna()

    #Reset the index
    data = data.reset_index(drop=True)

    # Replace the whitespaces in the titles with a underscore
    data["Title"] = data["Title"].str.replace(pat=" ", repl="_")

    """
    ---------- Genre manipulation: ----------
    """
    # Replace genre cathegories containing the word Animation to only containing Animation.
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Animation(.*)', value='Animation', regex=True)
    # Replace genre cathegories containing the genres Drama|Romance to only containing Romance.
    data["Genre"] = data["Genre"].replace(to_replace='Drama|Romance', value='Romance', regex=False)
    # Replace genre cathegories containing the word Western to only containing Western.
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Western(.*)', value='Western', regex=True)
    # Replace genre cathegories containing the word Sci-Fi to only containing Sci-Fi.
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Sci-Fi(.*)', value='Sci-Fi', regex=True)
    # Replace genre cathegories containing the word Horror to only containing Horror. If there are any sci-fi horrors, they will be in the sci-fi cathegory.
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Horror(.*)', value='Horror', regex=True)

    #Getting all unique cathegories from the dataset:
    unique_data = data.Genre.unique()
    
    #Iterating through the unique cathegories.
    unique_cathegories = []
    for cat in unique_data:
        #We find the Genres, that is only described by cathegory
        # "|" indicate that the movie has more than one genre.
        if "|" not in str(cat):
            unique_cathegories.append(cat)

    #Create data frame with the posters with only one genre
    one_Genre_df = data[data.Genre.isin(unique_cathegories)]

    #Create data frame based on the genres we want to be included in the CNN
    final_df = one_Genre_df[data.Genre.isin(["Drama", "Comedy", "Documentary", "Horror", "Thriller", "Western", "Romance", "Animation"])]
    #Reset the index
    final_df = final_df.reset_index(drop=True)
    #Print the uniqe genres in the data frame, these will be our labels later on
    final_df["Genre"].unique()

    '''
    ----------------------Creating data folder:-------------------
    '''
    #Create a data set based on the url in the Poster column in the data frame and find and save the errors in a list
    #Error handling: create folder if it does not allready exist
    try:
        os.mkdir("Posters")
    except FileExistsError:
        print(f"Posters already exists.")


    '''
    -------------- Download the images to a folder:---------------
    '''
    #Create empty list, where the error indexes will be storred
    errors = []


    #For the length of the data set
    for i in range(len(final_df)):
        #Define the index to be the index of the data frame
        index = str(i)

        #Create name of poster files based on the index in the data frame and the movie title
        filename = "Posters/"+ index + "_" + str(final_df["Title"][i]) + ".jpg"
        print(filename)

        #Accessing the links for the posters
        image_url = final_df["Poster"][i]

        #Error handling: Download the images to the folder, if there are any issues, add the index with the issue to the errors list
        try:
            #Download the image to the folder
            image_filename = wget.download(image_url, filename)
        #If the poster has an error: append the index to the list, pass, and move on to the next file.
        except: 
            errors.append(int(index))
            pass

   '''
   -------------- Save data frame and remove errors:---------------
   '''
    #Save the data frame as a csv
    final_df.to_csv("genre_df.csv") 

    #Remove the errors from the data frame based on the index
    final_df = final_df.drop(labels=errors, axis=0).reset_index()


     #We now want to create a new dataset where the errors are removed
     '''
     ----------------------Create data folder for the data without the errors:------------------- 
     '''
     #Error handling: create folder if it does not allready exist
      
     try:
         os.mkdir("Sorted_Posters")
     except FileExistsError:
         print(f"Sorted_Posters already exists.")

     '''
     -------------- Download the images to the new folder:---------------
     '''
     #For the length of the data set
     for i in range(len(final_df)):
         #Define the index to be the index of the data frame
         index = str(i)

         ##Create name of poster files based on the index in the data frame and the movie title
         filename = "Sorted_Posters/"+ index + "_" + str(final_df["Title"][i]) + ".jpg"
         print(filename)

         #Accessing the links for the posters
         image_url = final_df["Poster"][i]

         #Error handling: Download the images to the folder
         try:
             #Download the image to the folder
             image_filename = wget.download(image_url, filename)          
             except: 
                 pass
    '''
    -------------- Save data frame:---------------
    '''     
     #Save the data frame without the errors as a csv
     final_df.to_csv("sorted_genre_df.csv")  
      
    '''
    ---------------------- Convert the images into arrays and append to list:------------------- 
    '''
     #Define the image path
     image_path = os.path.join("Sorted_Posters")
      
     #Create empty list, where the arrays will be storred
     np_images = []

     #Convert every image in the image_path to numpy arrays
     for image in Path(image_path).glob("*"):
         image_open = cv2.imread(str(image))
         # The new dimensions 
         #dim = (32, 32)
         # resize the image
         #resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
         # append the resized images to the np_images list
         #np_images.append(resize_image.astype("float")/255.)
         np_images.append(image_open)

    '''
    ---------------------- Define traning and test data and labels:------------------- 
    '''
     # X represent the data and y the labels. The posters will be the data and the genres will be the labels   
     #Define the test and traing data and size
     X_train, X_test, y_train, y_test = train_test_split(np_images, 
                                                         final_df["Genre"], 
                                                         train_size = 1920, 
                                                         test_size = 480)    

     #Convert the data into arrays 
     X_train = np.array(X_train)
     X_test = np.array(X_test)

     #Flatten the data
     X_train = X_train/255.0
     X_test = X_test/255.0

     #Convert labels to one-hot encoding
     lb = LabelBinarizer()
     y_train = lb.fit_transform(y_train)
     y_test = lb.fit_transform(y_test)
      
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
        fig.savefig("model_performance.png")
      
    '''
    ---------------------- Define and train LeNet CNN model:------------------- 
    '''   
     #Define model as being Sequential and add layers
     model = Sequential()
     # First set of CONV => RELU => POOL
     model.add(Conv2D(268, (3, 3),  #NB: the filter is set to the input height (268) and the kernel to 3x3
                      padding="same", 
                      input_shape=(268, 182, 3))) #The shape of all the posters with height, width and dimensions
     model.add(Activation("relu"))
     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

     #View the summary
     model.summary()

     #Second set of CONV => RELU => POOL
     model.add(Conv2D(400, (5, 5), #NB: the filter is set to 400 and the kernel to 3x3
                      padding="same"))
     model.add(Activation("relu"))
     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

     # FC => RELU
     model.add(Flatten())
     model.add(Dense(500)) #NB: the filter is set to 500
     model.add(Activation("relu"))

     # Softmax classifier
     model.add(Dense(8))  #NB: the filter is set to 8 which is the number of unique labels
     model.add(Activation("softmax"))

     # Compile model
     opt = SGD(lr=0.01)
     model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

     # Save model summary as model_architecture.png
     plot_model(model, to_file = "model_architecture.png", show_shapes=True, show_layer_names=True)

     # Train the model
     H = model.fit(X_train, y_train, 
                   validation_data=(X_test, y_test), 
                   batch_size=10,
                   epochs=10, #NB: can be set as a paramenter
                   verbose=1)

     # Plot and save history via the earlier defined function
     plot_history(H, 10) #NB: epochs(10) can be set as a paramenter

     # Print the classification report
     predictions = model.predict(X_test, batch_size=10)
     print(classification_report(y_test.argmax(axis=1),
                                 predictions.argmax(axis=1),
                                 target_names=label_names))

#Define behaviour when called from command line
if __name__ == "__main__":
    main()
