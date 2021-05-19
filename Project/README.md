## Project – Genres by movie posters ##
**Johanne Brandhøj Würtz**

__Contribution__

I work with Emil Buus Thomsen and Rasmus Vesti Hansen in this assignment. All contributed equal-ly to every stage of this project. Subsequently, I have modified and corrected the code myself in rela-tion to in-depth comments and further corrections in the code. In the coding process itself, everyone has contributed equally (33%/33%33%)


__Assignment description__

This is the self-assigned project. For our project, we have chosen to make a CNN model to recognize movie genre based on movie posters. We have found a csv file based on IMDB's genre categorizations of movies as well as URL for the movie posters. 
  The question we wanted to answer through the project was: can the CNN model successfully categorize the six genre categories based on the posters?
The task ended up being a lot more challenging than we first assumed. Our first idea was to make our CNN model with 6 genre categories. This led to some unexpected results where some of the categories had 0 in their f1 score. It seemed that the model was not able to categorize our data. We therefore used a Grid Search optimizer to see if we could change some of the parameters for the better. This did not help the results. Then we tried again but with 3 genre categories instead of 6. Here the model could categorize the genres, where all categories were taken into account. We were surprised that this could be done. We therefore used the pretrained model VGG16, first with the 6 genres and subsequently with the 3 genres. Here the same thing happened again, where only the 3 genres suc-ceeded. The code we uploaded is our original CNN model, but with the 3 genres instead of the 6 we first planned, as this model worked best and ended with the highest accuracy. To answer our ques-tions, the model cannot categorize movie posters based on 6 genres but is able to categorize with fewer genres.


__Methods__

There has been a lot of preparation in relation to getting the desired genre categories and the associated posters. Here we have used simple Panda features as well as string manipulation using regex. In addi-tion, we have used error handling to find and subsequently remove errors from our data. Once the data was in place, we especially used features from Sklearn to split, train and evaluate our model. 

__Usage__

_Structure:_
The repository contains the folders data, src and output. The data folder contains the csv file MovieG-enre.csv, which contains information from IMDB regarding genre, URL etc. The csv file can be found here: https://www.kaggle.com/neha1703/movie-genre-from-its-poster?select=MovieGenre.csv. When you run the scripts, a folder with movie posters is created. This folder is going to be in the data folder. When running the first script (data_cleaning.py) a csv (sorted_df.csv) is saved to the data fold-er. This is the csv we use for the second script (CNN_Movie_Posters.py).
The src folder contains the script CNN_Movie_Posters.py, which is our primary script where the data is model is trained and evaluated. In addition, the script data_cleaning.py is also in the src folder. This is the script we used to fetch and clean the data. The output folder contains the images of the model’s architecture and performance. 

I have created a bash script that creates and activates a virtual environment, retrieves necessary librar-ies from the requirements.txt file and runs the script for the assignment. There were some version er-rors when I tried to install the packages from the requirements file, these packages will be installed directly from the bash script.

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
__Discussion of results__

When we run our CNN model with 3 genres, we get an accuracy of 64%. If we look at the performance itself, we can see that the model overfits. In addition, we can see that it is the genre with the most posters associated that scores highest on accuracy (Horror with 75%). At the same time, we can see that it is the genre with the fewest associated posters that has the lowest accuracy (Western with 31%). To counteract this trend of overfitting, the size of the dropout layer can be increased. Another option that could make sure that most posters do not equal the greatest accuracy is to make sure that there are equal numbers of posters in each category. That our model was not able to categorize a higher number of genres may be due to the fact that our data that has not been optimal. We base this, among other things, on the fact that a pretrained model was also unable to categorize the genres. So even though we have changed the various parameters, the number of images and dimensions as well as the model we have not managed to get completely in house with it. Despite not going quite as ex-pected, it has been quite interesting to try to solve the problems and end up with fewer genre categories as a possible solution, even though it was not what we expected.


