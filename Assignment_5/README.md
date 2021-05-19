## Assignment 5 - CNNs on cultural image data
**Johanne Brandhøj Würtz**

__Contribution__

I work with Emil Buus Thomsen, Christoffer Mondrup Kramer and Rasmus Vesti Hansen in this assignment. All contributed equally to every stage of this project. Subsequently, I have modified and corrected the code myself in relation to in-depth comments and further corrections in the code. In the coding process itself, everyone has contributed equally (25%/25%/25%/25%).


__Assignment description__

This is assignment 5. In this assignment we make a multi-class classification of impressionist painters. This is the first time we are working with actual cultural data and for this assignment we will use what we have learned so far to build a classifier which can predict artists from paintings. The purposes of this assignment are to build and train deep convolutional neural networks, preprocess and prepare image data for use in these models and how to handle more complex, cultural image data. 

__Methods__

We have built a deep learning model using a convolutional neural network which classify paintings by their respective artists. We have used the LeNet model for this assignment. The pre-processing of the data was the most challenging aspect of this assignment. The images needed to be split into training and test data and made the dimensions the same shape. Furthermore, we needed to make the images into arrays and extract the labels from the names of each painter.  

__Usage__

_Structure:_
The repository contains the folders data, src and output. We have made a smaller data set based on the original. The smaller dataset can be found in the data folder. The code for how we did this appears in the script itself but is commented out. We used the both the full dataset and the sample when we ran the script. The original data for the assignment can be found and downloaded here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data. The src folder contains the script for the assignment and the output folder contains images of the model’s architecture and performance. 

_Parameters:_
It is possible to change the number of epochs where 20 is the default value. The number of epochs can be specified by “-epo” or “--epochs_number” followed by the number as an integer.

I have created a bash script that creates and activates a virtual environment, retrieves necessary libraries and runs the script for the assignment. There were some version errors when I tried to install the packages from the requirements file, these packages will be installed directly from the bash script.


## How to run
**Step 1: Clone repo**
- Open terminal
- Navigate to destination you want the repo
- type the following command
 ```console
 git clone https://github.com/JohanneBW/cds_visual_assignments.git
 ```
**Step 2: Set up enviroment and run script:**
- Type the following command to navigate to the folder "Assignment_5":
```console
cd cds_visual_assignments
cd Assignment_5
```  
- Use the bash script _run_assignment5.sh_ to set up environment and run the script in the src folder
- Type the following command:
```console
bash run_assignment5.sh
```  
**Else: Run with other parameters**
- Activate the virtual environment by typing the following:
```console
Assignment5_venv/bin/activate 
```
- Run the script with specified parameters. 
- Example: Run with 50 epochs. Type the following:
```console
python3 cnn-artists.py -epo 50
```

__Discussion of results__

If we run the script with the full dataset and the default number of epochs, which is 20 we get an accu-racy of 33%. If we run the script with the same number of epochs, we get an accuracy of 22%. As a starting point, I thought that the model did not perform well, but if the performance is put in relation to what you actually ask the model to do, I would argue that it actually performs quite well. To elaborate, we wanted the model to categorize a series of paintings based on the painter. Here it is worth bearing in mind that there are a limited number of paintings. First of all, these are paintings in the form of cul-tural data, of which there are simply only a certain number. If we try to manipulate the data so that we can create 'new' false data based on the original paintings, we risk undermining the very basis of the study. It is therefore not possible to continue training the model with new data, as there is simply no more. This assignment is a good example of how you often work with data, as it is a balancing act between what you want to examine, what you have available and how you want to examine it.
