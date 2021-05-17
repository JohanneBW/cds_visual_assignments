## Project – Genres by movie posters ##
**Johanne Brandhøj Würtz**

__Assignment description__

This is the self-assigned project. For our project, we have chosen to make a CNN model to recognize movie genre based on movie posters. We have found a csv file based on IMDB's genre categorizations of movies as well as URL for the movie posters. We have chosen to only work with movies that have one genre associated with it and we have therefore combined some genres to get more data for the individual genre categories. We ended up picking out some genres we would focus on. In my case, I have chosen the six genres: Animation, Western, Sci-fi, Horror, Documentary and Comedy. These genres have been chosen as I have a presumption that the genres are clearly expressed in their posters. In other words, I believe that these are genres that are basically easy to categorize, as they have some 'clear' characteristics. In the Animation genre the style is recognizable, in Westerns there are special objects such as horses and cowboy hats etc. The question I want to answer through the project is: can the model successfully categorize the six genre categories based on the posters?

__Methods__

There has been a lot of preparation in relation to getting the desired genre categories and the associated posters. Here we have used simple Panda features as well as string manipulation using regex. In addition, we have used error handling to find and subsequently remove errors from our data. Once the data was in place, we especially used features from Sklearn to split, train and evaluate our model. We have created a supplementary script (GridSearchCV.py) that can also be found in the src folder, where we used Sklearn's GridSerchCV function to estimate the best parameters for our model in relation to optimizers, batch size and epochs.

__Usage__

_Structure:_
The repository contains the folders data, src and output. The data folder contains the csv file MovieG-enre.csv, which contains information from IMDB regarding genre, URL etc. The csv file can be found here: https://www.kaggle.com/neha1703/movie-genre-from-its-poster?select=MovieGenre.csv. When you run the script, two folders with movie posters are created. One for all the posters (Posters), where we do error handling and one with the posters we then use as data in our model (Sort-ed_posters). These two folders are going to be in the data folder. 
The src folder contains the script CNN_Genre.py, which is our primary script where the data is sorted, and the model is trained and evaluated. In addition, the script GridSearchCV.py is also in the src folder. This is the script we used to estimate the results of various parameters for the model. In the output folder two csv files are saved. One contains the data frame before the final genre categories are selected (genre_df.csv) and the other contains the data frame we use in the model (sort-ed_genre_df.csv. The output folder also contains the images of the model’s architecture and perfor-mance. 

I have created a bash script that creates and activates a virtual environment, retrieves necessary libraries from the requirements.txt file and runs the script for the assignment.

**How to run**

The following shows how to set up the virtual environment and run the script step by step.

**Step 1: Clone repo**
- Open terminal
- Navigate to destination you want the repo
- Type the following command:

```console
git clone https://github.com/JohanneBW/cds_visual_assignments.git
```  
**Step 2: Set up environment and run scripts**
- Type the following command to navigate to the folder "Project":

```console
cd cds_visual_assignments
cd Project
``` 
- Use the bash script _run_project.sh_ to set up environment and run the script in the src folder
- Type the following command:

```console
bash run_project.sh
```


