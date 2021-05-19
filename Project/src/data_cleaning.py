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
 

'''
-------------- Main function:---------------
'''
def main():
    
    '''
    ---------------------Import data:------------------------------
    '''
    #Path to data
    input_file = os.path.join("..", "data", "MovieGenre.csv")

    #Reading data
    data = pd.read_csv(input_file, encoding = "ISO-8859-1")
    
    """
    ---------- Genre manipulation: -------------------------
    """

    '''Some of the genres do not have many posters associated with them. 
    This includes: Animation, Horror and Western. 
    These are some of the genres that we would like to include in our model, 
    as we have an expectation that their genre will be clearly expressed in the posters.
    Therefore, we compromise and accept these genres as the primary ones. That is, if a film is both animation and western, 
    it will be considered solely as animation. 
    We have done this in order so that the most dominant genres become the primary genres.
    '''

    # Replace genre cathegories containing the word Animation to only containing Animation.
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Animation(.*)', value='Animation', regex=True)
    # Replace genre cathegories containing the word Western to only containing Western.
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Western(.*)', value='Western', regex=True)
    # Replace genre cathegories containing the word Horror to only containing Horror.
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


    # Replace the whitespaces in the titles with a underscore
    data["Title"] = data["Title"].str.replace(pat=" ", repl="_")
    
    #Make a data frame
    one_Genre_df = data[data.Genre.isin(unique_cathegories)]
    
    
    #Create data frame based on the genres we want to be included in the CNN
    one_Genre_df = one_Genre_df[data.Genre.isin(["Horror", "Western", "Animation"])]
    #Reset the index
    one_Genre_df = one_Genre_df.reset_index()
    
    
    #Iterating through the unique cathegories.
    cathegories = []
    for cat in one_Genre_df["Genre"].unique():
    #We find the Genres, that is only described by cathegory
        # "|" indicate that the movie has more than one genre.
        if "|" not in str(cat):
            cathegories.append(cat)

    print(f"The cathegories in the data set: {cathegories}")
    
    
    
    '''
    ----------------------Creating data folder:-------------------
    '''
    #Create a data set based on the url in the Poster column in the data frame and find and save the errors in a list
    
    #Error handling: create folder if it does not allready exist
    try:
        os.mkdir(os.path.join("..", "data", "Poster_data"))
        print("Poster_data was created!")
    except FileExistsError:
        print("Poster_data already exists!")
        
    '''
    -------------- Download the images to a folder:---------------
    '''
    #Create empty list, where the error indexes will be storred
    errors = []
    
    #For the length of the data set
    for i in range(len(one_Genre_df)):
        #Define the index to be the index of the data frame
        index = str(i)
        #Create name of poster files based on the index in the data frame and the movie title
        filename = "../data/Poster_data/"+ str(one_Genre_df["Title"][i]) + ".jpg"
        print(filename)
    
        #Accessing the links for the posters
        image_url = one_Genre_df["Poster"][i]
    
        #Error handling: Download the images to the folder, if there are any issues, add the index with the issue to the errors list
        try:
            image_filename = wget.download(image_url, filename)
            #If the poster has an error: append the index to the list, pass, and move on to the next file.
        except:
            print("There was an error")
            errors.append(int(index))
            pass
    
    '''
    -------------- Save data frame and remove errors:---------------
    '''
    #Remove the errors from the data frame based on the index
    sorted_df = one_Genre_df.drop(labels=errors, axis=0).reset_index(drop=True)
    
    #Save the data frame as a csv
    sorted_df.to_csv(os.path.join("..", "data", "sorted_df.csv"))
    
#Define behaviour when called from command line

if __name__ == "__main__":
    main() 