## Assignment 3 - Edge detection
**Johanne Brandhøj Würtz**

__Contribution__

I did not work with others on this assignment.


__Assignment description__

This is assignment 3. The purpose of this assignment is to use computer vision to extract specific features from images. In this assignment we want to find language-like objects, such as letters and punc-tuation by using edge detection. 
The First thing we want to do, is to draw a green rectangular box to show a region of in-terest (ROI) around the main body of text in the middle of the image. The next thing we want to do, is to crop the original image to create a new image containing only the ROI in the rectangle. Then we use Canny edge detection to 'find' every letter in the image and draw a green contour around each letter in the cropped image.

__Methods__

In this assignment I primarily used elements from the numpy and cv2 libraries. The actual scaling of the image in relation to finding ROI as well as color manipulation and eventually edge detection was solved using different cv2 methods such as grayscaling, blurring and Canny edge detection. The cropping I solved using numpy.

__Usage__

Structure:
The repository contains the folders data, src and output. The data folder contains the data for the script. The image for this assignment is already located in the data folder but can also be found by the following link:https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG. The src folder contains the source code for the assignment. In this assignment I have used one script (edge_detection.py) to solve the tasks. The output folder contains the outputs from the script. In this assignment the output is four images.

I have created a bash script that creates and activates a virtual environment, retrieves necessary libraries from the requirements.txt file (two of the packages are installed directly in the bash script) and runs the script for the assignment. 


## How to run
**Step 1: Clone repo**
- Open terminal
- Navigate to destination you want the repo
- Type the following command:
 ```console
 git clone https://github.com/JohanneBW/cds_visual_assignments.git
 ```
**Step 2: Set up enviroment and run script:**
- Type the following command to navigate to the folder "Assignment_3".:
```console
cd cds_visual_assignments
cd Assignment_3
```  
- Use the bash script _run_assignment3.sh_  to set up environment and run the program
- Type the following command:
 
```console
bash run_assignment3.sh
```  
__Discussion of results__

I managed to apply Canny edge detection to the image of the Jefferson Memorial. However, I ended up with a script that is difficult to apply to other images for similar tasks. One of the reasons for this is that both the image and the regions of interest can be in vastly different sizes. in addition, the regions of interest are not necessarily in the center of the image, nor can they necessarily be enclosed by a rec-tangle of the same format. Therefore, there are many elements that must be met in order to have a successful edge detection. Since the code itself is not easy to apply directly to other images, I have instead tried to explain the different steps needed to make a successful edge detection. That way, it's easy to figure out what's needed and how to do it.
