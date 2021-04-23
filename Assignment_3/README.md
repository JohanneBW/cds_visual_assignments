## Assignment 3 - Edge detection

__Finding text using edge detection__

The purpose of this assignment is to use computer vision to extract specific features from images. In particular, we're going to see if we can find text. We are not interested in finding whole words right now; we'll look at how to find whole words in a coming class. For now, we only want to find language-like objects, such as letters and punctuation.

__Data__
The data used for this asignment can be found by the following link:
https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG


Using the skills you have learned up to now, do the following tasks:
Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. Save this as image_with_ROI.jpg.
Crop the original image to create a new image containing only the ROI in the rectangle. Save this as image_cropped.jpg.
Using this cropped image, use Canny edge detection to 'find' every letter in the image
Draw a green contour around each letter in the cropped image. Save this as image_letters.jpg
