#!/usr/bin/env python

"""
---------- Import libraries ----------
"""
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
---------- Main function ----------
"""

def main():

    
    """
    ---------- Path ----------
    """
    # Define the path were the image is located
    image_path = os.path.join("..", data", "We_Hold_These_Truths_at_Jefferson_Memorial.JPG")
    
    # image is now defined by reading the path
    image = cv2.imread(image_path)
    
    
    """
    ---------- Region of interest  ----------
    """
    # I check the image shape. This is not nessecary but I like to have an idea about the dimensions
    image.shape
    # I now find the centers of the x and y axes. 
    # The centers are defined by dividing the width(x) and height(y) by two.
    (centre_x, centre_y) = (image.shape[1]//2, image.shape[0]//2)
    # I find the region of interest by by having the two centers as a starting point. 
    
    # From here, the distance from the center and out to the frame of the ROI is subtracted or added depending on whether it is on the right or left side of the ROI.
    ROI = cv2.rectangle(image, (centre_x-1000, centre_y-1000), (centre_x+1000, centre_y+1500), (0,255,0), 3)
    #The last two elements refer to the color, namely green, and the thickness of the line that makes up the ROI, namely 3 pixels.
    
    """
    ---------- Save ROI  ----------
    """    
    # I define the path were the image is going to be stored. 
    outfile_ROI = os.path.join("..", "output", "image_with_ROI.jpg")
    # Save the image 
    cv2.imwrite(outfile_ROI, ROI)
    
    
    """
    ---------- Save ROI  ----------
    """  
    # Read the image with the region of interest as ROI
    ROI = cv2.imread(outfile_ROI)
    # I have tried some different approaches but ended up cropping the image by using a feature from np.array
    # NB: the order of values is different from when I found the ROI. 
    #This time I had to combine it as: start_y:end_y and start_x:end_x
    cropped = ROI[centre_y-1000:centre_y+1500, centre_x-1000:centre_x+1000]
    
    """
    ---------- Save cropped  ----------
    """     
    # Define the path were the cropped image will be saved
    outfile_cropped = os.path.join("..", "output", "image_cropped.jpg")
    # Save the cropped image
    cv2.imwrite(outfile_cropped, cropped)
    
    """
    ---------- Canny edge detection  ----------
    """     
    # Read the cropped image
    cropped = cv2.imread(outfile_cropped)
    # Making the cropped image grey scale.
    grey_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # Blur the image
    # I use the Gaussian blur to blur the cropped, grey image
    blurred = cv2.GaussianBlur(grey_image, (5,5), 0)
    # I use canny edge detection to detect the letters 
    canny = cv2.Canny(blurred, 115, 150) 
    # I tried some differnt min and max values. 
    # If the values are lower, the outlines of the bricks are clearer
    # If the values are higher, some of the letters was ignorred 
    
    """
    ---------- Contours  ----------
    """      
    # I find the contours on the canny edge detected image 
    (contours, _) = cv2.findContours(canny.copy(),
                     cv2.RETR_EXTERNAL,
                     cv2.CHAIN_APPROX_SIMPLE)
    # I apply the contours
    cropped_contour = cv2.drawContours(cropped.copy(), # The contours are applied on the cropped image 
                                       contours, -1, # We want all the "letters" to be marked, so we choose the value -1, where a possitive value are refering to an 'index'/an individual letter
                                       (0,255,0) # The contours will be green
                                       , 2)  # the thickness of the markings will be two pixels
    
    """
    ---------- Save contoured image  ----------
    """  
    # Define the path were the contoured image will be saved
    outfile_contoured = os.path.join("..", "output", "image_letters.jpg")
    # Save the contoured image
    cv2.imwrite(outfile_contoured, cropped_contour)
    
#Define behaviour when called from command line
if __name__ == "__main__":
    main()   
        
