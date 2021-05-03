## Assignment 3 - Edge detection
**Johanne BW**

__Finding text using edge detection__

The purpose of this assignment is to use computer vision to extract specific features from images. For now, we want to find language-like objects, such as letters and punctuation.

__Data__

This folder contains the data for the script. The image for this assignment is already located in the data folder. 
The data used for this asignment can be found by the following link:
https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG

__output__

The folder contains the outputs from the script. In this assignment the output is four images.

__src__

This folder contains the source code for the assignment. In this assignment I have used one script (edge_detection.py) to solve the tasks. 

__The tasks for the assignment__

- Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. 
- Save this as image_with_ROI.jpg.
- Crop the original image to create a new image containing only the ROI in the rectangle. 
- Save this as image_cropped.jpg.
- Using this cropped image, use Canny edge detection to 'find' every letter in the image
- Draw a green contour around each letter in the cropped image. 
- Save this as image_letters.jpg


## How to run
**Step 1: Clone repo**
- open terminal
- Navigate to destination you want the repo
- type the following command
 ```console
 git clone https://github.com/JohanneBW/cds_visual_assignments.git
 ```
**step 2: Set up enviroment and run program:**
- Navigate to the folder "Assignment_3".
```console
cd cds_visual_assignments
cd Assignment_3
```  
- Use the bash script _run_assignment3.sh_ to set up environment and run the program:  
```console
bash run_assignment3.sh
```  
