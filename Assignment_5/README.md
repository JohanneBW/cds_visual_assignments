## Assignment 5 - CNNs on cultural image data
**Johanne BW**
DESCRIPTION
Multi-class classification of impressionist painters

data:
I have made a smaller data set based on the original. The smaller dataset can be found under the data folder. The code for how I did this appears in the script itself, but is commented out. I just want to point out that this is a small sample set and that it can be seen on the results.
You can find the original data for the assignment here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data

If you run the script from the terminal it is possible to choose the number of epochs yourself. A default value of 20 epocs has been inserted, so it is easier to test.


model = LeNet


## How to run
**Step 1: Clone repo**
- open terminal
- Navigate to destination you want the repo
- type the following command
 ```console
 git clone https://github.com/JohanneBW/cds_visual_assignments.git
 ```
**step 2: Set up enviroment and run program:**
- Navigate to the folder "Assignment_5".
```console
cd cds_visual_assignments
cd Assignment_5
```  
- Use the bash script _run_assignment5.sh_ to run the program and set up environment:  
```console
bash run_assignment5.sh
```  
- Else: Run the script directly from the terminal
```console
python3 cnn-artists.py 
```
