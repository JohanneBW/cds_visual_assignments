## Assignment 4 - Classification benchmarks
**Johanne Brandhøj Würtz**

__Assignment description__

This is assignment 4. For this assignment we create two command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal. 
These scripts can then be used to provide easy-to-understand benchmark scores for evaluating these models. The purposes of this assignment are to train classification models using machine learning and neural networks, create simple models that can be used as statistical benchmarks and how to do so using scripts which can be executed from the command line.
 

__Methods__

In this assignment we made two models respectively a logistic regression model and a neural network model. We primarily used elements from Sklearn. The lr-mnist.py script uses a logistic regression model and the nn-mnist.py script uses a neural network model. The task was mostly to process the data so that it could be trained on the pre-made functions we have worked with in class imported from the utils folder. The data processing involved dividing the full data set in training and test data and labels, scaling the features to a value between 0 and 1 as well as binarizing the labels. 

__Usage__

_Structure:_
The repository contains the folders src and utils. For this assignment we use the MNIST data which can be found by the following link. The data is imported in the scripts. The src folder contains the source code for the assignment. In this assignment we have used two scripts (lr-mnist.py and nn-mnist.py) to solve the tasks. The utils folder contains utility functions we have used in the lessons for our CDS/Visual Analytics class. The output for this assignment is printed to the terminal after running the scripts.

_Parameters:_
It is possible to change the size of the training and test data in the two scripts from the command-line. Both scripts have 80 percent as the size of their training data and 20 percent as the size of their test data as their default values. In addition, for the nn-mnist.py script it is possible to change the number of epocs where 20 is the default value. The size of the test data can be specified by "-tes" or "--test_size" followed by the size as a float. The number of epochs can be specified by “-epo” or “--epochs_number” followed by the number as an integer. 

I have created a bash script that creates and activates a virtual environment, retrieves necessary libraries from the requirements.txt file and run the scripts for the assignment.


## How to run
**Step 1: Clone repo**
- Open terminal
- Navigate to destination you want the repo
- Type the following command:
 ```console
 git clone https://github.com/JohanneBW/cds_visual_assignments.git
 ```
**Step 2: Set up enviroment and run scrips:**
- Type the following command to navigate to the folder "Assignment_4":
```console
cd cds_visual_assignments
cd Assignment_4
```  
- Use the bash script _run_lr_nn_scripts.sh_ to set up environment and run the two scripts in the src folder
- Type the following command: 
```console
bash run_lr_nn_scripts.sh
```  
**Else: Run with other parameters**
- Activate the virtual environment by typing the following:
```console
Assignment4_venv/bin/activate
```  
- Run the script with specified parameters. 
- Example: The nn-mnist.py with 50 epochs. Type the following:
```console
python3 nn-mnist.py -epo 50
``` 
