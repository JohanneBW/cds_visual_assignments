## Assignment 4 - Classification benchmarks
**Johanne BW**

## Classifier benchmarks using Logistic Regression and a Neural Network

__Description__ 

Create two command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal. 
These scripts can then be used to provide easy-to-understand benchmark scores for evaluating these models. 

__data__

For this assignment we use the MNIST data which can be found by the following link: http://yann.lecun.com/exdb/mnist/. The data is importet in the scripts. 

__output__

The output for this assignment is printed to the terminal after running the scripts.

__src__

This folder contains the source code for the assignment. In this assignment I have used two script (lr-mnist.py and nn-mnist.py) to solve the tasks. The lr-mnist.py script uses a logistic regression model and the nn-mnist.py script uses a neural network model. 

__utils__

This folder contains Utility functions we have used in the lessons for our CDS/Visual Analytics class. 

__Parameters__

It is possible to change the size of the training and test data in the two scripts from the command-line. Both scripts have 80 percent as the size of their training data and 20 percent as the size of their test data as their default values. In addition, for the nn-mnist.py script it is possible to change the number of epocs where 100 is the default value. 

## How to run
**Step 1: Clone repo**
- open terminal
- Navigate to destination you want the repo
- type the following command
 ```console
 git clone https://github.com/JohanneBW/cds_visual_assignments.git
 ```
**step 2: Set up enviroment:**
- Navigate to the folder "Assignment_4".
```console
cd cds_visual_assignments
cd Assignment_4
```  
- Use the bash script _create_assignment4_venv.sh_ to set up environment:  
```console
bash create_assignment4_venv.sh
```  
**step 3: Run the program:**
- Navigate to the folder "Assignment_4" if you are not already there
- As a shortcut you can run the programs by running the run_lr_nn_scripts.sh where the enviroment activates and the two scripts in the src folder run. 
```console
bash run_lr_nn_scripts.sh
``` 

