# ===================================================================

# Solution to the Durham Logo problem from the Image Processing practical

# Author : Amir Atapour Abarghouei, amir.atapour-abarghouei@durham.ac.uk

# Copyright (c) 2024 Amir Atapour Abarghouei

# based on : https://github.com/tobybreckon/python-examples-ip/blob/master/skeleton.py
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# ===================================================================

import cv2

# ===================================================================

# load the images

img = cv2.imread('peppers.png')
logo = cv2.imread('DULogo.png')
mask = cv2.imread('DULogoMask.png')

# Read the dimensions of the logo

rows, cols, channels = logo.shape

# Extract the region of interest (ROI) of the original image
# This is where the logo will be placed

roi = img[0:rows, 0:cols]

# at this point, we have the ROI as an intermediary result

# Save the ROI as an intermediate result
# this is just to see the progress of the approach
 
cv2.imwrite("step1.png", roi)

# the pixels in the ROI where the logo will be placed are turned to zero
# we will use logical transforms

roi = cv2.bitwise_and(roi,mask)

# save an intermediate result

cv2.imwrite("step2.png", roi)

# make the background pixels of the logo black

logo = cv2.bitwise_and(logo, cv2.bitwise_not(mask))

# save an intermediate result

cv2.imwrite("step3.png", logo)

# bitwise OR the ROI and the logo (as have been processed by the mask)

roi = cv2.bitwise_or(roi,logo)

# save an intermediate result

cv2.imwrite("step4.png", roi)

# copy the ROI which now has the logo on it on to the original image 

img[0:rows, 0:cols] = roi

# save the final result 

cv2.imwrite("final-result.png", img)

# ===================================================================
